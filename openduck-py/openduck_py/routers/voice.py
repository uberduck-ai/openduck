import asyncio
import os
import re
from time import time
from typing import Optional, Dict, Literal
import wave
import requests
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
import numpy as np
from scipy.io import wavfile
from sqlalchemy import select
import torch
from whisper import load_model
from daily import *

from openduck_py.models import DBChatHistory, DBChatRecord
from openduck_py.models.chat_record import EventName
from openduck_py.db import get_db_async, AsyncSession, SessionAsync
from openduck_py.prompts import prompt
from openduck_py.voices.styletts2 import styletts2_inference, STYLETTS2_SAMPLE_RATE
from openduck_py.settings import (
    IS_DEV,
    WS_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    CHUNK_SIZE,
    LOG_TO_SLACK,
    CHAT_MODEL,
)
from openduck_py.routers.templates import chat_continuation
from openduck_py.utils.speaker_identification import (
    segment_audio,
    load_pipelines,
)
from openduck_py.logging.slack import log_audio_to_slack

if IS_DEV:
    normalize_text = lambda x: x
else:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    normalizer = Normalizer(input_case="cased", lang="en")
    normalize_text = normalizer.normalize


try:
    pipeline, inference = load_pipelines()
except OSError:
    pipeline, inference = load_pipelines()

with open("aec-cartoon-degraded.wav", "wb") as f:
    f.write(
        requests.get(
            "https://s3.us-west-2.amazonaws.com/quack.uberduck.ai/aec-cartoon-degraded.wav"
        ).content
    )

speaker_embedding = inference("aec-cartoon-degraded.wav")
whisper_model = load_model("base.en")

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

    def log(self):
        if LOG_TO_SLACK:
            log_audio_to_slack(self.wav_file_path)


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
    def __init__(
        self,
        session_id: str,
        input_audio_format: Literal["float32", "int32", "int16"] = "float32",
        record=False,
    ):
        self.session_id = session_id
        self.response_queue = asyncio.Queue()
        self.is_responding = False
        self.input_audio_format = input_audio_format
        self.audio_data = []
        self.recorder = WavAppender(wav_file_path=f"{session_id}.wav")
        self.vad = SileroVad()
        self.record = record
        self.time_of_last_activity = time()
        self.response_task = None

    def interrupt(self, task: asyncio.Task):
        assert self.is_responding
        print("interrupting!")
        task.cancel()
        self.is_responding = False

    async def receive_audio(self, message: np.ndarray):
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

        audio_16k: torch.Tensor = torch.tensor(audio_16k_np)

        self.audio_data.append(audio_16k_np)
        if self.record:
            self.recorder.append(audio_16k_np)
        i = 0
        while i < len(audio_16k):
            upper = i + self.vad.window_size
            if len(audio_16k) - self.vad.window_size < upper:
                upper = len(audio_16k)
            audio_16k_chunk = audio_16k[i:upper]
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

    async def start_response(
        self,
        audio_data: np.ndarray,
    ):
        self.is_responding = True
        async with SessionAsync() as db:
            await log_event(db, self.session_id, "started_response", audio=audio_data)
            t_0 = time()

            def _inference(sentence: str):
                audio_chunk = styletts2_inference(
                    text=sentence,
                    output_sample_rate=OUTPUT_SAMPLE_RATE,
                )
                audio_chunk = np.int16(audio_chunk * 32767).tobytes()
                return audio_chunk

            # Remove echo
            audio_data = await asyncio.to_thread(
                segment_audio,
                audio_data,
                WS_SAMPLE_RATE,
                speaker_embedding,
                pipeline,
                inference,
            )

            t_echo = time()
            await log_event(
                db,
                self.session_id,
                "removed_echo",
                audio=audio_data,
                latency=t_echo - t_0,
            )

            transcription = await asyncio.to_thread(_transcribe, audio_data)
            print("TRANSCRIPTION: ", transcription, flush=True)
            t_whisper = time()
            await log_event(
                db,
                self.session_id,
                "transcribed_audio",
                meta={"text": transcription},
                latency=t_whisper - t_echo,
            )
            if not transcription or len(audio_data) < 100:
                return

            system_prompt = {
                "role": "system",
                "content": prompt("system-prompt"),
            }
            new_message = {"role": "user", "content": transcription}

            chat = (
                await db.execute(
                    select(DBChatHistory).where(
                        DBChatHistory.session_id == self.session_id
                    )
                )
            ).scalar_one_or_none()
            if chat is None:
                chat = DBChatHistory(
                    session_id=self.session_id,
                    history_json={"messages": [system_prompt]},
                )
                db.add(chat)
            messages = chat.history_json["messages"]
            messages.append(new_message)

            response = await chat_continuation(messages)
            response_message = response.choices[0].message
            completion = response_message.content
            t_chat = time()
            await log_event(
                db,
                self.session_id,
                "generated_completion",
                meta={"text": completion},
                latency=t_chat - t_whisper,
            )
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
            t_normalize = time()
            await log_event(
                db,
                self.session_id,
                "normalized_text",
                meta={"text": normalized},
                latency=t_normalize - t_chat,
            )
            sentences = re.split(r"(?<=[.!?]) +", normalized)
            for sentence in sentences:
                audio_chunk_bytes = await asyncio.to_thread(_inference, sentence)
                t_styletts = time()
                await log_event(
                    db,
                    self.session_id,
                    "generated_tts",
                    audio=np.frombuffer(audio_chunk_bytes, dtype=np.int16),
                    latency=t_styletts - t_normalize,
                )

                for i in range(0, len(audio_chunk_bytes), CHUNK_SIZE):
                    chunk = audio_chunk_bytes[i : i + CHUNK_SIZE]
                    await self.response_queue.put(chunk)

            print("Whisper", t_whisper - t_0, flush=True)
            print(CHAT_MODEL, t_chat - t_whisper, flush=True)
            print("Normalizer", t_normalize - t_chat, flush=True)
            print("StyleTTS2 + sending bytes", t_styletts - t_normalize, flush=True)


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
    latency: Optional[float] = None,
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
    record = DBChatRecord(
        session_id=session_id, event_name=event, meta_json=meta, latency=latency
    )
    db.add(record)
    await db.commit()


async def daily_consumer(queue: asyncio.Queue, mic: VirtualMicrophoneDevice):
    while True:
        chunk = await queue.get()  # Dequeue a chunk
        if chunk:
            mic.write_frames(chunk)
            queue.task_done()


async def websocket_consumer(queue: asyncio.Queue, websocket: WebSocket):
    """Dequeue audio chunks and send them through the websocket."""
    while True:
        chunk = await queue.get()  # Dequeue a chunk
        if chunk:
            await websocket.send_bytes(chunk)  # Send the chunk through the websocket
            queue.task_done()


@audio_router.websocket("/response")
async def audio_response(
    websocket: WebSocket,
    session_id: str,
    record: bool = False,
    db: AsyncSession = Depends(get_db_async),
):
    await websocket.accept()

    responder = ResponseAgent(session_id=session_id, record=record)
    asyncio.create_task(websocket_consumer(responder.response_queue, websocket))

    try:
        while True:
            if time() - responder.time_of_last_activity > 300:
                print("closing websocket due to inactivity")
                break
            if _check_for_exceptions(responder.response_task):
                responder.audio_data = []
                responder.response_task = None
                responder.time_of_last_activity = time()
            try:
                message = await websocket.receive_bytes()
                await responder.receive_audio(message)

            except WebSocketDisconnect:
                print("websocket disconnected")
                return

    finally:
        responder.recorder.close_file()
        responder.recorder.log()

    await responder.response_queue.join()
    await websocket.close()
    await log_event(db, session_id, "ended_session")


async def connect_daily():
    session_id = str(uuid4())

    Daily.init()
    mic = Daily.create_microphone_device(
        "my-mic", sample_rate=OUTPUT_SAMPLE_RATE, channels=1, non_blocking=True
    )
    speaker = Daily.create_speaker_device(
        "my-speaker", sample_rate=WS_SAMPLE_RATE, channels=1
    )
    Daily.select_speaker_device("my-speaker")
    client = CallClient()
    client.update_subscription_profiles(
        {"base": {"camera": "unsubscribed", "microphone": "subscribed"}}
    )
    client.join(
        meeting_url="https://matthewkennedy5.daily.co/Od7ecHzUW4knP6hS5bug",
        client_settings={
            "inputs": {
                "camera": False,
                "microphone": {
                    "isEnabled": True,
                    "settings": {"deviceId": "my-mic"},
                },
                "speaker": {
                    "isEnabled": True,
                    "settings": {"deviceId": "my-speaker"},
                },
            }
        },
    )
    responder = ResponseAgent(
        session_id=session_id, record=False, input_audio_format="int16"
    )
    asyncio.create_task(daily_consumer(responder.response_queue, mic))
    while True:
        if _check_for_exceptions(responder.response_task):
            responder.audio_data = []
            responder.response_task = None
            responder.time_of_last_activity = time()

        message = speaker.read_frames(WS_SAMPLE_RATE // 10)
        if len(message) > 0:
            await responder.receive_audio(message)
        await asyncio.sleep(0.01)

    responder.recorder.close_file()
    responder.recorder.log()
    await responder.response_queue.join()
    await log_event(db, session_id, "ended_session")


if __name__ == "__main__":
    asyncio.run(connect_daily())
