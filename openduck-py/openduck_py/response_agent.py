import asyncio
from time import time
from typing import Literal, Optional, List, AsyncGenerator
import re
from io import BytesIO

from litellm import acompletion
from sqlalchemy import select
import numpy as np
import httpx

from openduck_py.configs.tts_config import TTSConfig
from openduck_py.settings import (
    CHUNK_SIZE,
    WS_SAMPLE_RATE,
    CHAT_MODEL_GPT4,
    CHAT_MODEL,
    LOG_TO_SLACK,
    ML_API_URL,
)
from openduck_py.configs.tts_config import TTSConfig
from openduck_py.db import AsyncSession, SessionAsync
from openduck_py.prompts import prompt
from openduck_py.models import DBChatHistory, DBChatRecord
from openduck_py.logging.db import log_event
from openduck_py.utils.third_party_tts import (
    aio_elevenlabs_tts,
    ELEVENLABS_VIKRAM,
    ELEVENLABS_CHRIS,
)


async def _normalize_text(text: str) -> str:
    url = f"{ML_API_URL}/ml/normalize"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json={"text": text})

    if response.status_code == 200:
        return response.json()["text"]
    else:
        raise Exception(f"Normalization failed with status code {response.status_code}")


async def _inference(sentence: str) -> AsyncGenerator[bytes, None]:
    url = f"{ML_API_URL}/ml/tts"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json={"text": sentence})

        async for chunk in response.aiter_bytes(CHUNK_SIZE):
            yield chunk


async def _transcribe(audio_data: np.ndarray) -> str:
    assert audio_data.dtype == np.float32
    wav_io = BytesIO(audio_data.tobytes())
    wav_data = wav_io.getvalue()

    # Send the POST request to the endpoint
    url = f"{ML_API_URL}/ml/transcribe"
    files = {"audio": ("audio.wav", wav_data, "application/octet-stream")}
    async with httpx.AsyncClient() as client:
        response = await client.post(url, files=files)

    if response.status_code == 200:
        return response.json()["text"]
    else:
        raise Exception(f"Transcription failed with status code {response.status_code}")


class SileroVad:
    model = None
    # Default window size in examples at https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies#examples
    window_size = 512

    def reset_states(self):
        if hasattr(self, "vad_iterator"):
            self.vad_iterator.reset_states()

    def __call__(self, audio_data):
        # Lazy import torch so that it doesn't slow down creating a new Daily room for the user
        import torch

        if not isinstance(audio_data, torch.Tensor):
            audio_data = torch.tensor(audio_data)
        if self.model is None:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad"
            )
            self.model = model
            self.utils = utils
            (
                get_speech_timestamps,
                save_audio,
                read_audio,
                VADIterator,
                collect_chunks,
            ) = utils
            self.vad_iterator = VADIterator(self.model)
        speech_dict = self.vad_iterator(audio_data, return_seconds=True)
        return speech_dict


class WavAppender:
    def __init__(self, wav_file_path="output.wav"):
        self.wav_file_path = wav_file_path
        self.file = None
        self.params_set = False

    def open_file(self):
        self.file = wave.open(
            self.wav_file_path,
            "wb" if not os.path.exists(self.wav_file_path) else "r+b",
        )

    def append(self, audio_data: np.ndarray):
        if audio_data.dtype == np.float32:
            audio_data = np.round(audio_data * 32767).astype(np.int16)
        assert audio_data.dtype == np.int16
        if not self.file:
            self.open_file()
        if not self.params_set:
            self.file.setnchannels(1)  # Mono
            self.file.setsampwidth(2)  # 16 bits
            self.file.setframerate(WS_SAMPLE_RATE)
            self.params_set = True
        self.file.writeframes(audio_data.tobytes())

    def close_file(self):
        if self.file:
            self.file.close()
            self.file = None
            self.params_set = False

    def log(self):
        if LOG_TO_SLACK:
            log_audio_to_slack(self.wav_file_path)


class ResponseAgent:
    def __init__(
        self,
        session_id: str,
        input_audio_format: Literal["float32", "int32", "int16"] = "float32",
        record=False,
        tts_config: Optional[TTSConfig] = None,
        system_prompt: str = "system-prompt",
        context: Optional[dict] = None,
    ):
        self.session_id = session_id
        self.response_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.is_responding = False
        self.input_audio_format = input_audio_format
        self.audio_data: List[np.ndarray] = []
        self.recorder = WavAppender(wav_file_path=f"{session_id}.wav")
        self.vad = SileroVad()
        self.record = record
        self.time_of_last_activity = time()
        self.response_task: Optional[asyncio.Task] = None
        self.system_prompt = system_prompt

        if context is None:
            context = {}
        self.context = context

        if tts_config is None:
            tts_config = TTSConfig()
        self.tts_config = tts_config

    def interrupt(self, task: asyncio.Task):
        assert self.is_responding
        print("interrupting!")
        task.cancel()
        self.is_responding = False

    async def receive_audio(self, message: bytes):
        # Convert the input audio from input_audio_format to float32
        # Silero VAD and Whisper require float32
        if self.input_audio_format == "float32":
            audio_16k_np = np.frombuffer(message, dtype=np.float32)
        elif self.input_audio_format == "int32":
            audio_16k_np = np.frombuffer(message, dtype=np.int32)
            audio_16k_np = audio_16k_np.astype(np.float32) / np.iinfo(np.int32).max
            audio_16k_np = audio_16k_np.astype(np.float32)
        elif self.input_audio_format == "int16":
            audio_16k_np = np.frombuffer(message, dtype=np.int16)
            audio_16k_np = audio_16k_np.astype(np.float32) / np.iinfo(np.int16).max
            audio_16k_np = audio_16k_np.astype(np.float32)

        self.audio_data.append(audio_16k_np)
        if self.record:
            self.recorder.append(audio_16k_np)
        i = 0
        while i < len(audio_16k_np):
            upper = i + self.vad.window_size
            if len(audio_16k_np) - self.vad.window_size < upper:
                upper = len(audio_16k_np)
            audio_16k_chunk = audio_16k_np[i:upper]
            vad_result = self.vad(audio_16k_chunk)
            if vad_result:
                # TODO (Matthew): Can we send telemetry via an API instead of saving to a database?
                async with SessionAsync() as db:
                    if "end" in vad_result:
                        print("end of speech detected.")
                        self.time_of_last_activity = time()
                        await log_event(db, self.session_id, "detected_end_of_speech")
                        if self.response_task is None or self.response_task.done():
                            self.response_task = asyncio.create_task(
                                self.start_response(
                                    np.concatenate(self.audio_data),
                                )
                            )
                        else:
                            print("already responding")
                    if "start" in vad_result:
                        print("start of speech detected.")
                        self.time_of_last_activity = time()
                        await log_event(db, self.session_id, "detected_start_of_speech")
                        if self.response_task and not self.response_task.done():
                            if self.is_responding:
                                await log_event(
                                    db, self.session_id, "interrupted_response"
                                )
                                self.interrupt(self.response_task)
            i = upper

    async def _generate_and_speak(
        self,
        db: SessionAsync,
        t_whisper=None,
        new_message=None,
        system_prompt=None,
        chat_model=CHAT_MODEL_GPT4,
    ):
        if system_prompt is None:
            system_prompt = prompt(
                f"most-interesting-bot/{self.system_prompt}", self.context
            )
        system_prompt = {
            "role": "system",
            "content": system_prompt,
        }

        chat = (
            await db.execute(
                select(DBChatHistory).where(DBChatHistory.session_id == self.session_id)
            )
        ).scalar_one_or_none()
        if chat is None:
            chat = DBChatHistory(
                session_id=self.session_id,
                history_json={"messages": [system_prompt]},
            )
            db.add(chat)
        messages = chat.history_json["messages"]
        if new_message is not None:
            messages.append(new_message)

        # NOTE(zach): retries
        response = None
        for _retry in range(3):
            try:
                response = await acompletion(
                    chat_model,
                    messages,
                    temperature=1.2,
                    stream=True,
                )
            except Exception:
                if _retry == 2:
                    raise
            else:
                break
        complete_sentence = ""
        full_response = ""
        if response is None:
            return
        async for chunk in response:
            chunk_text = chunk.choices[0].delta.content
            if not chunk_text:
                break
            complete_sentence += chunk_text
            full_response += chunk_text
            # TODO: Smarter sentence detection - this will split sentences on cases like "Mr. Kennedy"
            if re.search(r"(?<!\d)[.!?](?!\d)", chunk_text):
                await self.speak_response(complete_sentence, db, t_whisper)
                complete_sentence = ""

        messages.append({"role": "assistant", "content": full_response})
        chat.history_json["messages"] = messages
        await db.commit()

    async def start_response(
        self,
        audio_data: np.ndarray,
    ):
        self.is_responding = True
        async with SessionAsync() as db:
            await log_event(db, self.session_id, "started_response", audio=audio_data)
            t_0 = time()

            transcription = await _transcribe(audio_data)
            print("TRANSCRIPTION: ", transcription, flush=True)
            t_whisper = time()
            await log_event(
                db,
                self.session_id,
                "transcribed_audio",
                meta={"text": transcription},
                latency=t_whisper - t_0,
            )
            if not transcription or len(audio_data) < 100:
                return

            await self._generate_and_speak(
                db,
                t_whisper,
                {"role": "user", "content": transcription},
                chat_model=CHAT_MODEL_GPT4,
            )

    async def speak_response(
        self,
        response_text: str,
        db: AsyncSession,
        start_time: float,
    ):
        t_chat = time()
        await log_event(
            db,
            self.session_id,
            "generated_completion",
            meta={"text": response_text},
            latency=t_chat - start_time,
        )
        if "$ECHO" in response_text:
            print("Echo detected, not sending response.")
            return

        if self.tts_config.provider == "local":
            normalized = await _normalize_text(response_text)
            t_normalize = time()
            await log_event(
                db,
                self.session_id,
                "normalized_text",
                meta={"text": normalized},
                latency=t_normalize - t_chat,
            )

            audio_bytes_iter = _inference(normalized)
        elif self.tts_config.provider == "elevenlabs":
            t_normalize = time()
            await log_event(
                db,
                self.session_id,
                "normalized_text",
                meta={"text": response_text},
                latency=t_normalize - t_chat,
            )
            audio_bytes_iter = aio_elevenlabs_tts(
                response_text, voice_id=self.tts_config.voice_id
            )

        t_styletts = time()

        audio_chunk_bytes = bytes()
        async for chunk in audio_bytes_iter:
            await self.response_queue.put(chunk)
            audio_chunk_bytes += chunk
        await log_event(
            db,
            self.session_id,
            "generated_tts",
            audio=np.frombuffer(audio_chunk_bytes, dtype=np.int16),
            latency=t_styletts - t_normalize,
        )
