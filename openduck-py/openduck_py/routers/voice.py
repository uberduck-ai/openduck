import asyncio
import os
import re
from time import time
from typing import Optional, Dict, Literal
import wave
import requests
from pathlib import Path

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
import numpy as np
from scipy.io import wavfile
from sqlalchemy import select
import torch
from whisper import load_model

from openduck_py.models import DBChatHistory, DBChatRecord
from openduck_py.models.chat_record import EventName
from openduck_py.db import get_db_async, AsyncSession
from openduck_py.prompts import prompt
from openduck_py.voices.styletts2 import styletts2_inference, STYLETTS2_SAMPLE_RATE
from openduck_py.settings import IS_DEV, WS_SAMPLE_RATE
from openduck_py.routers.templates import generate
from openduck_py.utils.speaker_identification import (
    segment_audio,
    load_pipelines,
)

if IS_DEV:
    normalize_text = lambda x: x
else:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    normalizer = Normalizer(input_case="cased", lang="en")
    normalize_text = normalizer.normalize


pipeline, inference = load_pipelines()

with open("aec-cartoon-degraded.wav", "wb") as f:
    f.write(
        requests.get(
            "https://s3.us-west-2.amazonaws.com/quack.uberduck.ai/aec-cartoon-degraded.wav"
        ).content
    )

speaker_embedding = inference("aec-cartoon-degraded.wav")
whisper_model = load_model("tiny.en")

audio_router = APIRouter(prefix="/audio")


def _transcribe(audio_data):
    return whisper_model.transcribe(audio_data)["text"]


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


class SileroVad:
    model = None
    # Default window size in examples at https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies#examples
    window_size = 512

    def reset_states(self):
        if hasattr(self, "vad_iterator"):
            self.vad_iterator.reset_states()

    def __call__(self, audio_data):
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


class ResponseAgent:

    def __init__(self, session_id: str):
        self.is_responding = False
        self.session_id = session_id

    def __init__(self, output_audio_format="int16", output_sample_rate=24_000):
        self.output_audio_format = output_audio_format
        self.output_sample_rate = output_sample_rate

    def interrupt(self, task: asyncio.Task):
        assert self.is_responding
        print("interrupting!")
        task.cancel()
        self.is_responding = False

    async def start_response(
        self,
        websocket: WebSocket,
        db: AsyncSession,
        audio_data: np.ndarray,
    ):
        print("starting response")
        await log_event(db, self.session_id, "started_response", audio=audio_data)
        self.is_responding = True

        def _inference(sentence: str, audio_format: str = "int16"):
            audio_chunk = styletts2.styletts2_inference(
                text=sentence,
                output_sample_rate=self.output_sample_rate,
            )
            if audio_format == "int32":
                # NOTE(zach): max value is 2147483647
                audio_chunk = np.int32(audio_chunk * 2_000_000_000)
            else:  # Default to int16
                audio_chunk = np.int16(audio_chunk * 32767)
            audio_chunk_bytes = audio_chunk.tobytes()
            return audio_chunk_bytes

        loop = asyncio.get_running_loop()
        audio_data = await loop.run_in_executor(
            None,
            segment_audio,
            audio_data,
            16000,
            speaker_embedding,
            pipeline,
            inference,
        )
        await log_event(db, self.session_id, "removed_echo", audio=audio_data)
        if len(audio_data) < 100:
            print(f"All audio has been filtered out. Not responding.")

        t0 = time()

        transcription = await loop.run_in_executor(None, _transcribe, audio_data)
        print("TRANSCRIPTION: ", transcription)

        if not transcription:
            return

        t0 = time()

        transcription = await loop.run_in_executor(None, _transcribe, audio_data)
        print("transcription", transcription)
        await log_event(
            db, self.session_id, "transcribed_audio", meta={"text": transcription}
        )
        t_whisper = time()
        if not transcription:
            return

        system_prompt = {
            "role": "system",
            "content": prompt("system-prompt"),
        }
        new_message = {"role": "user", "content": transcription}

        chat = (
            await db.execute(
                select(DBChatHistory).where(DBChatHistory.session_id == self.session_id)
            )
        ).scalar_one_or_none()
        if chat is None:
            chat = DBChatHistory(
                session_id=self.session_id, history_json={"messages": [system_prompt]}
            )
            db.add(chat)
        messages = chat.history_json["messages"]
        messages.append(new_message)
        response = await generate({"messages": messages}, [], "gpt-35-turbo-deployment")
        response_message = response.choices[0].message
        completion = response_message.content
        await log_event(
            db, self.session_id, "generated_completion", meta={"text": completion}
        )

        t_gpt = time()

        print(
            f"Used {response.usage.prompt_tokens} prompt tokens and {response.usage.completion_tokens} completion tokens"
        )

        if "$ECHO" in completion:
            print("Echo detected, not sending response.")
            return

        messages.append({"role": response_message.role, "content": completion})
        chat.history_json["messages"] = messages
        await db.commit()

        normalized = normalize_text(response_message.content)
        await log_event(
            db, self.session_id, "normalized_text", meta={"text": normalized}
        )
        t_normalize = time()
        sentences = re.split(r"(?<=[.!?]) +", normalized)
        for sentence in sentences:
            audio_chunk_bytes = await loop.run_in_executor(
                None, _inference, sentence, self.output_audio_format
            )
            await log_event(
                db,
                self.session_id,
                "generated_tts",
                audio=np.frombuffer(audio_chunk_bytes, dtype=np.int16),
            )
            for i in range(0, len(audio_chunk_bytes), 1024):
                await websocket.send_bytes(audio_chunk_bytes[i : i + 1024])

            print("DONE SENDING BYTES")

        t_styletts = time()

        print("Whisper", t_whisper - t0)
        print("GPT", t_gpt - t_whisper)
        print("Normalizer", t_normalize - t_gpt)
        print("StyleTTS2 + sending bytes", t_styletts - t_normalize)


def _check_for_exceptions(response_task: Optional[asyncio.Task]) -> bool:
    reset_state = False
    if response_task and response_task.done():
        try:
            response_task.result()
        except asyncio.CancelledError:
            print("response task was cancelled")
        except Exception as e:
            print("response task raised an exception:", e)
        else:
            print(
                "response task completed successfully. Resetting audio_data and response_task"
            )
            reset_state = True
        return reset_state


async def log_event(
    db: AsyncSession,
    session_id: str,
    event: EventName,
    meta: Optional[Dict[str, str]] = None,
    audio: Optional[np.ndarray] = None,
):
    if audio is not None:
        log_path = f"logs/{session_id}/{event}_{time()}.wav"
        abs_path = Path(__file__).resolve().parents[2] / log_path
        session_folder = abs_path.parent
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)

        sample_rate = WS_SAMPLE_RATE
        if event == "generated_tts":
            sample_rate = STYLETTS2_SAMPLE_RATE
        wavfile.write(abs_path, sample_rate, audio)
        print(f"Wrote wavfile to {abs_path}")

        meta = {"audio_url": log_path}
    record = DBChatRecord(session_id=session_id, event_name=event, meta_json=meta)
    db.add(record)
    await db.commit()


@audio_router.websocket("/response")
async def audio_response(
    websocket: WebSocket,
    session_id: str,
    record: bool = False,
    input_audio_format: Literal["float32", "int32"] = "float32",
    output_audio_format: Literal["int16", "int32"] = "int16",
    output_sample_rate: int = 24_000,
    db: AsyncSession = Depends(get_db_async),
):
    await websocket.accept()
    time_of_last_activity = time()
    await log_event(db, session_id, "started_session")

    vad = SileroVad()
    responder = ResponseAgent(
        output_audio_format=output_audio_format, output_sample_rate=output_sample_rate
    )
    recorder = WavAppender(wav_file_path=f"{session_id}.wav")

    audio_data = []
    response_task = None
    try:
        while True:
            if time() - time_of_last_activity > 30:
                print("closing websocket due to inactivity")
                break
            if _check_for_exceptions(response_task):
                audio_data = []
                response_task = None
                time_of_last_activity = time()
            try:
                message = await websocket.receive_bytes()

            except WebSocketDisconnect:
                print("websocket disconnected")
                return

            # NOTE(zach): Client records at 16khz sample rate.
            if input_audio_format == "float32":
                audio_16k_np = np.frombuffer(message, dtype=np.float32)
            elif input_audio_format == "int32":
                audio_16k_np = np.frombuffer(message, dtype=np.int32)
                audio_16k_np = audio_16k_np.astype(np.float32) / np.iinfo(np.int32).max
                audio_16k_np = audio_16k_np.astype(np.float32)

            audio_16k: torch.Tensor = torch.tensor(audio_16k_np)
            audio_data.append(audio_16k_np)
            if record:
                recorder.append(audio_16k_np)
            i = 0
            while i < len(audio_16k):
                upper = i + vad.window_size
                if len(audio_16k) - vad.window_size < upper:
                    upper = len(audio_16k)
                audio_16k_chunk = audio_16k[i:upper]
                vad_result = vad(audio_16k_chunk)
                if vad_result:
                    if "end" in vad_result:
                        print("end of speech detected.")
                        time_of_last_activity = time()

                        await log_event(db, session_id, "detected_end_of_speech")
                        if response_task is None or response_task.done():
                            response_task = asyncio.create_task(
                                responder.start_response(
                                    websocket,
                                    db,
                                    np.concatenate(audio_data),
                                )
                            )
                        else:
                            print("already responding")
                    if "start" in vad_result:
                        print("start of speech detected.")
                        time_of_last_activity = time()
                        await log_event(db, session_id, "detected_start_of_speech")
                        if response_task and not response_task.done():
                            if responder.is_responding:
                                await log_event(db, session_id, "interrupted_response")
                                responder.interrupt(response_task)
                i = upper
    finally:
        recorder.close_file()

    # TODO(zach): We never actually close it right now, we wait for the client
    # to close. But we should close it based on some timeout.
    await websocket.close()
    await log_event(db, session_id, "ended_session")
