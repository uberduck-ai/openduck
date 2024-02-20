import re
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, Depends, Query, WebSocket
from sqlalchemy import select
import whisper
from time import time

import numpy as np

from openduck_py.models import DBChatHistory
from openduck_py.db import get_db_async, AsyncSession
from openduck_py.voices import styletts2
from openduck_py.routers.templates import generate

model = whisper.load_model("tiny")  # Fastest possible whisper model

audio_router = APIRouter(prefix="/audio")


@audio_router.websocket("/response")
async def audio_response(
    websocket: WebSocket,
    session_id: str,
    db: AsyncSession = Depends(get_db_async),
):
    await websocket.accept()

    print("Session ID", session_id)
    audio_data = await websocket.receive_bytes()
    assert session_id is not None
    t0 = time()

    def _transcribe():
        with NamedTemporaryFile() as temp_file:
            temp_file.write(audio_data)
            transcription = model.transcribe(temp_file.name)["text"]
            return transcription

    from asgiref.sync import sync_to_async

    _async_transcribe = sync_to_async(_transcribe)
    transcription = await _async_transcribe()

    if not transcription:
        return

    t_whisper = time()

    system_prompt = {
        "role": "system",
        "content": "You are a children's toy which can answer educational questions. You want to help your user and support them. Give short concise responses no more than 2 sentences.",
    }
    new_message = {"role": "user", "content": transcription}

    chat = (
        await db.execute(
            select(DBChatHistory).where(DBChatHistory.session_id == session_id)
        )
    ).scalar_one_or_none()
    if chat is None:
        chat = DBChatHistory(
            session_id=session_id, history_json={"messages": [system_prompt]}
        )
        db.add(chat)

    messages = chat.history_json["messages"]
    messages.append(new_message)
    response = await generate({"messages": messages}, [], "gpt-35-turbo-deployment")

    t_gpt = time()

    response_message = response.choices[0].message

    messages.append(
        {"role": response_message.role, "content": response_message.content}
    )
    chat.history_json["messages"] = messages
    await db.commit()

    sentences = re.split(r"(?<=[.!?]) +", response_message.content)
    for sentence in sentences:
        # TODO: deal with asyncio
        audio_chunk = styletts2.styletts2_inference(text=sentence)
        audio_chunk_bytes = np.int16(
            audio_chunk * 32767
        ).tobytes()  # Scale to 16-bit integer values
        await websocket.send_bytes(audio_chunk_bytes)

    t_styletts = time()

    print("Whisper", t_whisper - t0)
    print("GPT", t_gpt - t_whisper)
    print("StyleTTS2", t_styletts - t_gpt)

    # await websocket.send_text("done")
    await websocket.close()
