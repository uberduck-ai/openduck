import asyncio
from time import time
from typing import Literal, Optional, List, AsyncGenerator
import re
from io import BytesIO
import tempfile
import os
import wave

from litellm import acompletion
from sqlalchemy import select
import numpy as np
import httpx
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from scipy.io.wavfile import write

from openduck_py.settings import (
    CHUNK_SIZE,
    WS_SAMPLE_RATE,
    CHAT_MODEL,
    LOG_TO_SLACK,
    ML_API_URL,
    DEEPGRAM_API_SECRET,
    ASR_METHOD,
    TEMPERATURE,
)
from openduck_py.configs.tts_config import TTSConfig
from openduck_py.db import AsyncSession, SessionAsync
from openduck_py.prompts import prompt
from openduck_py.models import DBChatHistory
from openduck_py.logging.db import log_event
from openduck_py.logging.slack import log_audio_to_slack
from openduck_py.utils.third_party_tts import (
    aio_azure_tts,
    aio_elevenlabs_tts,
    aio_gptsovits_tts,
    aio_openai_tts,
)


deepgram = DeepgramClient(DEEPGRAM_API_SECRET)


async def _completion_with_retry(chat_model, messages):
    # NOTE(zach): retries
    response = None
    for _retry in range(3):
        try:
            response = await acompletion(
                chat_model,
                messages,
                temperature=TEMPERATURE,
                stream=True,
            )
        except Exception:
            if _retry == 2:
                raise
        else:
            break
    return response


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

    if ASR_METHOD == "whisper":
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
            raise Exception(
                f"Transcription failed with status code {response.status_code}"
            )
    elif ASR_METHOD == "deepgram":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            write(tmp_file, WS_SAMPLE_RATE, audio_data)
            tmp_file_path = tmp_file.name
        payload: FileSource = {
            "buffer": open(tmp_file_path, "rb"),
        }
        options = PrerecordedOptions(
            model="nova-2",
        )
        try:
            response = deepgram.listen.prerecorded.v("1").transcribe_file(
                payload, options
            )
            transcript = response.results.channels[0].alternatives[0].transcript
        finally:
            os.remove(tmp_file_path)

        return transcript

    else:
        raise ValueError


class SileroVad:
    model = None
    # Default window size in examples at https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies#examples
    window_size = 512

    def init(self) -> None:
        # Lazy import torch so that it doesn't slow down creating a new Daily room for the user
        import torch

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
    def __init__(self, local_path="output.wav", remote_path="output.wav"):
        self.wav_file_path = local_path
        self.remote_path = remote_path
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
        print("Logging audio to Slack", LOG_TO_SLACK, self.wav_file_path, flush=True)
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
        self.recorder = WavAppender(
            local_path=os.path.join(
                os.path.dirname(__file__),
                f"../logs/{session_id}",
                f"{session_id}.wav",
            ),
            remote_path=f"recordings/{session_id}.wav",
        )
        self.vad = SileroVad()
        self.transcript = ""
        self.record = record
        self.time_of_last_activity = time()
        self.response_task: Optional[asyncio.Task] = None
        self.interrupt_event = asyncio.Event()
        self.system_prompt = system_prompt

        if context is None:
            context = {}
        self.context = context

        if tts_config is None:
            tts_config = TTSConfig()
        self.tts_config = tts_config

        self._time_of_last_record = None

        if ASR_METHOD == "deepgram":
            self.dg_connection = deepgram.listen.live.v("1")
            options = LiveOptions(
                model="nova-2",
                punctuate=True,
                language="en-US",
                encoding="linear16",
                channels=1,
                sample_rate=WS_SAMPLE_RATE,
                interim_results=True,
                utterance_end_ms="1000",
                vad_events=True,
            )

            self.dg_connection.on(
                LiveTranscriptionEvents.Transcript,
                lambda x, result, **kwargs: self.on_deepgram_message(result),
            )
            self.dg_connection.start(options)

    def _reset_transcription(self):
        self.audio_data = []

    async def _clear_queue(self):
        while not self.response_queue.empty():
            await self.response_queue.get()

    async def interrupt(self, task: asyncio.Task):
        assert self.is_responding
        print("interrupting!")
        self.interrupt_event.set()
        task.cancel()
        await self._clear_queue()
        self.interrupt_event.clear()
        self.is_responding = False

    async def receive_audio(self, message: bytes):
        if ASR_METHOD == "deepgram":
            self.dg_connection.send(message)

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
            if self._time_of_last_record is None:
                self._time_of_last_record = time()
            elif time() - self._time_of_last_record > 0.2:
                print("TOOOOO SLOW: ", time() - self._time_of_last_record)
            self._time_of_last_record = time()
            self.recorder.append(audio_16k_np)

        i = 0
        while i < len(audio_16k_np):
            upper = i + self.vad.window_size
            if len(audio_16k_np) - self.vad.window_size < upper:
                upper = len(audio_16k_np)
            audio_16k_chunk = audio_16k_np[i:upper]
            vad_result = self.vad(audio_16k_chunk)
            if vad_result:
                async with SessionAsync() as db:
                    if "end" in vad_result:
                        print("end of speech detected.")
                        self.time_of_last_activity = time()
                        await log_event(db, self.session_id, "detected_end_of_speech")

                        audio_data = np.concatenate(self.audio_data)
                        transcription = await _transcribe(audio_data)
                        if not transcription:
                            continue

                        # Interrupt the current task if it's still running
                        if (
                            self.response_task
                            and not self.response_task.done()
                            and self.is_responding
                        ):
                            await log_event(db, self.session_id, "interrupted_response")
                            await self.interrupt(self.response_task)

                        await log_event(
                            db, self.session_id, "started_response", audio=audio_data
                        )
                        self.response_task = asyncio.create_task(
                            self.start_response(transcription)
                        )

                    if "start" in vad_result:
                        print("start of speech detected.")
                        self.time_of_last_activity = time()
                        await log_event(db, self.session_id, "detected_start_of_speech")
            i = upper

    async def _generate_and_speak(
        self,
        db: SessionAsync,
        t_asr=None,
        new_message=None,
        system_prompt=None,
        chat_model=CHAT_MODEL,
    ):
        t_previous = t_asr
        if system_prompt is None:
            system_prompt = prompt(self.system_prompt, self.context)
        system_prompt = {
            "role": "system",
            "content": system_prompt,
        }
        print("system_prompt: ", system_prompt, flush=True)

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
        else:
            messages = chat.history_json["messages"]
            if messages[0]["role"] == "system":
                messages[0] = system_prompt
            chat.history_json["messages"] = messages
        messages = chat.history_json["messages"]
        if new_message is not None:
            messages.append(new_message)

        chat.history_json["messages"] = messages
        await db.commit()
        self._reset_transcription()

        response = await _completion_with_retry(chat_model, messages)
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
                await self.speak_response(complete_sentence, db, t_previous)
                t_previous = time()
                inprogress_messages = messages + [
                    {"role": "assistant", "content": full_response}
                ]
                chat.history_json["messages"] = inprogress_messages
                await db.commit()
                complete_sentence = ""

        messages.append({"role": "assistant", "content": full_response})
        chat.history_json["messages"] = messages
        await db.commit()

    async def start_response(self, transcription):
        try:
            async with SessionAsync() as db:
                t_0 = time()

                print("TRANSCRIPTION: ", transcription, flush=True)

                if transcription and self.is_responding:
                    await log_event(db, self.session_id, "interrupted_response")
                    await self.interrupt(self.response_task)

                self.is_responding = True

                t_asr = time()
                await log_event(
                    db,
                    self.session_id,
                    "transcribed_audio",
                    meta={"text": transcription},
                    latency=t_asr - t_0,
                )
                if not transcription:
                    return

                await self._generate_and_speak(
                    db,
                    t_asr,
                    {"role": "user", "content": transcription},
                    chat_model=CHAT_MODEL,
                )
        finally:
            self.is_responding = False

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

        if self.tts_config.provider == "styletts2":
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
        else:
            t_normalize = time()
            await log_event(
                db,
                self.session_id,
                "normalized_text",
                meta={"text": response_text},
                latency=t_normalize - t_chat,
            )
            print("NORMALIZE LATENCY: ", t_normalize - t_chat, flush=True)
            if self.tts_config.provider == "elevenlabs":
                audio_bytes_iter = aio_elevenlabs_tts(
                    response_text, voice_id=self.tts_config.voice_id
                )
            elif self.tts_config.provider == "openai":
                audio_bytes_iter = aio_openai_tts(response_text)
            elif self.tts_config.provider == "azure":
                audio_bytes_iter = aio_azure_tts(response_text)
            elif self.tts_config.provider == "gptsovits":
                audio_bytes_iter = aio_gptsovits_tts(
                    response_text, voice_ref=self.tts_config.voice_id
                )

        audio_chunk_bytes = bytes()
        _idx = 0
        async for chunk in audio_bytes_iter:
            if _idx == 0:
                # Measure time to first byte.
                t_tts = time()
            if self.interrupt_event.is_set():
                break
            await self.response_queue.put(chunk)
            audio_chunk_bytes += chunk
            _idx += 1
        await log_event(
            db,
            self.session_id,
            "generated_tts",
            audio=np.frombuffer(audio_chunk_bytes, dtype=np.int16),
            latency=t_tts - t_normalize,
        )

    def on_deepgram_message(self, result):
        transcript = result.channel.alternatives[0].transcript
        if transcript:
            self.transcript = transcript
        print("DEEPGRAM TRANSCRIPT: ", self.transcript, flush=True)
