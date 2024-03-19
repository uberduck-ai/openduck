import asyncio
import os
import re
import multiprocessing
from time import time
from typing import Optional, Dict, Literal, AsyncGenerator
import wave
import requests
from pathlib import Path
from uuid import uuid4
from io import BytesIO

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
import numpy as np
from scipy.io import wavfile
from sqlalchemy import select
import torch
from daily import *
from litellm import acompletion
import httpx

from openduck_py.configs.tts_config import TTSConfig
from openduck_py.models import DBChatHistory, DBChatRecord
from openduck_py.models.chat_record import EventName
from openduck_py.db import get_db_async, AsyncSession, SessionAsync
from openduck_py.prompts import prompt
from openduck_py.settings import (
    CHAT_MODEL,
    CHUNK_SIZE,
    LOG_TO_SLACK,
    ML_API_URL,
    OUTPUT_SAMPLE_RATE,
    WS_SAMPLE_RATE,
)
from openduck_py.utils.daily import create_room, RoomCreateResponse, CustomEventHandler
from openduck_py.utils.speaker_identification import load_pipelines
from openduck_py.utils.third_party_tts import aio_elevenlabs_tts
from openduck_py.logging.slack import log_audio_to_slack


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
audio_router = APIRouter(prefix="/audio")

Daily.init()

processes = {}


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


async def _inference(sentence: str) -> AsyncGenerator[bytes, None]:
    url = f"{ML_API_URL}/ml/tts"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json={"text": sentence})

        async for chunk in response.aiter_bytes(CHUNK_SIZE):
            yield chunk


async def _normalize_text(text: str) -> str:
    url = f"{ML_API_URL}/ml/normalize"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json={"text": text})

    if response.status_code == 200:
        return response.json()["text"]
    else:
        raise Exception(f"Normalization failed with status code {response.status_code}")


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
        tts_config: TTSConfig = None,
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

        if tts_config is None:
            tts_config = TTSConfig()
        self.tts_config = tts_config

    def interrupt(self, task: asyncio.Task):
        assert self.is_responding
        print("interrupting!")
        task.cancel()
        self.is_responding = False

    async def receive_audio(self, message: np.ndarray):
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

            response = await acompletion(
                CHAT_MODEL, messages, temperature=0.3, stream=True
            )

            complete_sentence = ""
            full_response = ""
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
            audio_bytes_iter = aio_elevenlabs_tts(response_text)

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


def _check_for_exceptions(response_task: Optional[asyncio.Task]) -> bool:
    reset_state = False
    if response_task and response_task.done():
        try:
            response_task.result()
        except asyncio.CancelledError:
            print("response task was cancelled")
        # except Exception as e:
        #     print("response task raised an exception:", e)
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
            sample_rate = OUTPUT_SAMPLE_RATE
        wavfile.write(abs_path, sample_rate, audio)
        print(f"Wrote wavfile to {abs_path}")

        meta = {"audio_url": log_path}
    record = DBChatRecord(
        session_id=session_id, event_name=event, meta_json=meta, latency_seconds=latency
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


@audio_router.post("/start")
async def create_room_and_start():
    room_info = await create_room()
    print("created room")

    process = multiprocessing.Process(
        target=run_connect_daily, args=(room_info["url"],)
    )
    process.start()
    print("started process: ", process.pid)
    processes[process.pid] = process
    return RoomCreateResponse(
        url=room_info["url"],
        name=room_info["name"],
        privacy=room_info["privacy"],
    )


@audio_router.websocket("/response")
async def audio_response(
    websocket: WebSocket,
    session_id: str,
    record: bool = False,
    db: AsyncSession = Depends(get_db_async),
):
    await websocket.accept()

    responder = ResponseAgent(
        session_id=session_id,
        record=record,
    )
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


async def connect_daily(
    room="https://matthewkennedy5.daily.co/Od7ecHzUW4knP6hS5bug",
):
    session_id = str(uuid4())
    mic = Daily.create_microphone_device(
        "my-mic", sample_rate=OUTPUT_SAMPLE_RATE, channels=1, non_blocking=True
    )
    speaker = Daily.create_speaker_device(
        "my-speaker", sample_rate=WS_SAMPLE_RATE, channels=1
    )
    Daily.select_speaker_device("my-speaker")

    event_handler = CustomEventHandler()
    client = event_handler.client
    client.update_subscription_profiles(
        {"base": {"camera": "unsubscribed", "microphone": "subscribed"}}
    )
    client.join(
        meeting_url=room,
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
        session_id=session_id,
        record=False,
        input_audio_format="int16",
        tts_config=TTSConfig(provider="elevenlabs"),
    )
    asyncio.create_task(daily_consumer(responder.response_queue, mic))
    while True:
        if _check_for_exceptions(responder.response_task):
            responder.audio_data = []
            responder.response_task = None
            responder.time_of_last_activity = time()
        if event_handler.left:
            print("left the call")
            break

        message = speaker.read_frames(WS_SAMPLE_RATE // 10)
        if len(message) > 0:
            await responder.receive_audio(message)
        await asyncio.sleep(0.01)

    responder.recorder.close_file()
    responder.recorder.log()
    await responder.response_queue.join()
    async with SessionAsync() as db:
        await log_event(db, session_id, "ended_session")


def run_connect_daily(room_url: str):
    asyncio.run(connect_daily(room=room_url))


if __name__ == "__main__":
    asyncio.run(connect_daily())
