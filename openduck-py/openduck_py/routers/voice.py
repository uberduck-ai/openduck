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
from daily import *
from litellm import acompletion
import httpx

from openduck_py.response_agent import ResponseAgent
from openduck_py.configs.tts_config import TTSConfig
from openduck_py.models import DBChatHistory, DBChatRecord
from openduck_py.models.chat_record import EventName
from openduck_py.db import get_db_async, AsyncSession, SessionAsync
from openduck_py.prompts import prompt
from openduck_py.settings import (
    CHAT_MODEL,
    CHAT_MODEL_GPT4,
    CHUNK_SIZE,
    LOG_TO_SLACK,
    ML_API_URL,
    OUTPUT_SAMPLE_RATE,
    WS_SAMPLE_RATE,
)
from openduck_py.utils.daily import create_room, RoomCreateResponse, CustomEventHandler
from openduck_py.utils.third_party_tts import (
    aio_elevenlabs_tts,
    ELEVENLABS_VIKRAM,
    ELEVENLABS_CHRIS,
)
from openduck_py.logging.slack import log_audio_to_slack
from openduck_py.logging.db import log_event

with open("aec-cartoon-degraded.wav", "wb") as f:
    f.write(
        requests.get(
            "https://s3.us-west-2.amazonaws.com/quack.uberduck.ai/aec-cartoon-degraded.wav"
        ).content
    )

audio_router = APIRouter(prefix="/audio")

Daily.init()

processes = {}


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
        target=run_connect_daily,
        kwargs=dict(
            room_url=room_info["url"],
            username="Vikram (AI)",
            prompt="podcast_host",
            voice_id=ELEVENLABS_VIKRAM,
            speak_first=True,
        ),
    )
    process.start()
    print("started process: ", process.pid)
    processes[process.pid] = process
    print("number of running processes: ", len(processes))

    return RoomCreateResponse(
        url=room_info["url"],
        name=room_info["name"],
        privacy=room_info["privacy"],
    )


@audio_router.post("/start/podcast")
async def create_room_and_start_podcast():
    room_info = await create_room()
    print("created room")

    # Podcast host
    process = multiprocessing.Process(
        target=run_connect_daily,
        kwargs=dict(
            room_url=room_info["url"],
            username="Vikram (AI)",
            prompt="podcast_host",
            voice_id=ELEVENLABS_VIKRAM,
            speak_first=True,
        ),
    )
    process.start()
    print("started process: ", process.pid)
    processes[process.pid] = process

    # Podcast guest
    process = multiprocessing.Process(
        target=run_connect_daily,
        kwargs=dict(
            room_url=room_info["url"],
            username="Chris (AI)",
            prompt="podcast_guest",
            voice_id=ELEVENLABS_CHRIS,
        ),
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
    username: str = "host (AI)",
    system_prompt=None,
    voice_id=None,
    speak_first=False,
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

    assert username.endswith(" (AI)"), "Username must end with ' (AI)'"
    client.set_user_name(username)
    client.update_subscription_profiles(
        {"base": {"camera": "unsubscribed", "microphone": "subscribed"}}
    )
    print("PARTICIPANTS: ", client.participants())
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
    my_name = username.split(" (AI)")[0]
    context = {
        "my_name": my_name,
    }
    responder = ResponseAgent(
        session_id=session_id,
        record=False,
        input_audio_format="int16",
        tts_config=TTSConfig(provider="elevenlabs", voice_id=voice_id),
        system_prompt=system_prompt,
        context=context,
    )
    asyncio.create_task(daily_consumer(responder.response_queue, mic))
    if speak_first:
        async with SessionAsync() as db:
            participants = client.participants()
            participant_names = [
                p["info"]["userName"].split(" (AI)")[0]
                for p in participants.values()
                if not p["info"]["isLocal"]
            ]
            await responder._generate_and_speak(
                db,
                t_whisper=time(),
                new_message=None,
                system_prompt=prompt(
                    f"most-interesting-bot/intro-prompt",
                    {
                        "my_name": my_name,
                        "participant_names": " and ".join(participant_names),
                    },
                ),
                chat_model=CHAT_MODEL,
            )
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


def run_connect_daily(
    room_url: str,
    username: str,
    prompt: str,
    voice_id: Optional[str] = None,
    speak_first=False,
):
    asyncio.run(
        connect_daily(
            room=room_url,
            username=username,
            system_prompt=prompt,
            voice_id=voice_id,
            speak_first=speak_first,
        )
    )


if __name__ == "__main__":
    asyncio.run(connect_daily())
