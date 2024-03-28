import asyncio
import os
import multiprocessing
from time import time
from typing import Optional, Dict
import requests
from uuid import uuid4

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Request
from daily import *

from openduck_py.response_agent import ResponseAgent
from openduck_py.configs.tts_config import TTSConfig
from openduck_py.db import get_db_async, AsyncSession, SessionAsync
from openduck_py.logging.slack import log_audio_to_slack
from openduck_py.prompts import prompt
from openduck_py.settings import (
    CHAT_MODEL,
    OUTPUT_SAMPLE_RATE,
    WS_SAMPLE_RATE,
)
from openduck_py.utils.daily import (
    create_room,
    RoomCreateResponse,
    CustomEventHandler,
    start_recording,
    stop_and_download_recording,
)
from openduck_py.utils.third_party_tts import (
    ELEVENLABS_VIKRAM,
    ELEVENLABS_CHRIS,
)
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


def _check_for_exceptions(response_task: Optional[asyncio.Task]):
    if response_task and response_task.done():
        try:
            return response_task.result()
        except asyncio.CancelledError:
            print("response task was cancelled")
        except Exception as e:
            print("response task raised an exception:", e)
        else:
            print("response task completed successfully.")


async def daily_consumer(
    queue: asyncio.Queue, interrupt: asyncio.Event, mic: VirtualMicrophoneDevice
):

    buffer_estimate = 0
    buffer_estimate_t0 = None

    while True:
        if interrupt.is_set():
            await asyncio.sleep(0.1)
            continue

        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=0.1)  # Dequeue a chunk
        except asyncio.TimeoutError:
            buffer_estimate = 0
            continue

        if buffer_estimate == 0:
            buffer_estimate_t0 = time()
        assert buffer_estimate_t0 is not None
        buffer_estimate += len(chunk)
        buffered_time_seconds = buffer_estimate / 2 / OUTPUT_SAMPLE_RATE
        clock_time_seconds = time() - buffer_estimate_t0

        MAX_LAG = 1

        if chunk:
            if buffered_time_seconds > clock_time_seconds + MAX_LAG:
                await asyncio.sleep(
                    buffered_time_seconds - clock_time_seconds - MAX_LAG
                )
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
async def create_room_and_start(request: Request):
    request_data = await request.json()
    context = request_data.get("context", {})

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
            context=context,
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
            _check_for_exceptions(responder.response_task)
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
    system_prompt="system-prompt",
    voice_id=None,
    speak_first=False,
    context: Optional[Dict[str, str]] = None,
    record=False,
):
    session_id = str(uuid4())
    mic = Daily.create_microphone_device(
        "my-mic",
        sample_rate=OUTPUT_SAMPLE_RATE,
        channels=1,
        non_blocking=True,
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
    base_context = {
        "my_name": my_name,
    }
    if context is not None:
        base_context.update(context)
    daily_recording_id = await start_recording(room)

    responder = ResponseAgent(
        session_id=session_id,
        record=record,
        input_audio_format="int16",
        # tts_config=TTSConfig(provider="elevenlabs", voice_id=voice_id),
        # tts_config=TTSConfig(provider="openai", voice_id=voice_id),
        tts_config=TTSConfig(provider="azure"),
        system_prompt=system_prompt,
        context=base_context,
    )
    asyncio.create_task(
        daily_consumer(responder.response_queue, responder.interrupt_event, mic)
    )
    # NOTE(zach): this is the first greeting
    if speak_first:
        async with SessionAsync() as db:
            participants = client.participants()
            while len(participants) < 2:
                await asyncio.sleep(0.1)
                participants = client.participants()
            participant_names = [
                p["info"].get("userName", "unknown").split(" (AI)")[0]
                for p in participants.values()
                if not p["info"]["isLocal"]
            ]
            await responder._generate_and_speak(
                db,
                t_asr=time(),
                new_message=None,
                system_prompt=prompt(
                    "intros/greeting.txt",
                    dict(
                        responder.context,
                        my_name=my_name,
                        participant_names=" and ".join(participant_names),
                    ),
                ),
                chat_model=CHAT_MODEL,
            )
    responder.vad.init()
    # NOTE(zach): The main loop of receiving audio and sending it to the agent.
    while True:
        _check_for_exceptions(responder.response_task)
        if event_handler.left:
            print("left the call")
            break

        message = speaker.read_frames(WS_SAMPLE_RATE // 1)
        if len(message) > 0:
            asyncio.create_task(responder.receive_audio(message))
        await asyncio.sleep(0.01)

    try:
        print("closing and logging to slack")
        responder.recorder.close_file()
        responder.recorder.log()
        room_id = room.split("/")[-1]
        daily_recording_path = await stop_and_download_recording(
            room_id, daily_recording_id
        )
        log_audio_to_slack(daily_recording_path)

        async with SessionAsync() as db:
            await log_event(db, session_id, "ended_session")
    except Exception as e:
        print("Error logging to slack:", e)

    finally:
        print("exiting the process")
        os._exit(0)


def run_connect_daily(
    room_url: str,
    username: str,
    prompt: str,
    voice_id: Optional[str] = None,
    speak_first=False,
    context: Optional[Dict[str, str]] = None,
):
    asyncio.run(
        connect_daily(
            room=room_url,
            username=username,
            system_prompt=prompt,
            voice_id=voice_id,
            speak_first=speak_first,
            context=context,
        )
    )


if __name__ == "__main__":
    asyncio.run(connect_daily())
