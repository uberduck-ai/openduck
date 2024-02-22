import re
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, Depends, Query, WebSocket
from sqlalchemy import select
import nemo.collections.asr as nemo_asr
from time import time
from torchaudio.functional import resample

import numpy as np
from asgiref.sync import sync_to_async
import torch
from torchaudio.functional import resample

from openduck_py.models import DBChatHistory
from openduck_py.db import get_db_async, AsyncSession
from openduck_py.voices import styletts2
from openduck_py.routers.templates import generate

asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/stt_en_fastconformer_ctc_large")

audio_router = APIRouter(prefix="/audio")


def _transcribe(audio_data):
    # resampled = resample(
    #     torch.tensor(audio_data), orig_freq=24000, new_freq=16000
    # ).numpy()
    # return asr_model.transcribe([resampled])[0]
    with NamedTemporaryFile() as temp_file:
        temp_file.write(audio_data)
        transcription = asr_model.transcribe([resampled])[0]
    return transcription


_async_transcribe = sync_to_async(_transcribe)

SILENCE_THRESHOLD = 1.0


@audio_router.websocket("/response")
async def audio_response(
    websocket: WebSocket,
    session_id: str,
    db: AsyncSession = Depends(get_db_async),
):
    await websocket.accept()

    while True:
        audio_data = []
        started_talking = False
        while True:
            message = await websocket.receive_bytes()
            print("Received audio!")
            audio_chunk = np.frombuffer(message, dtype=np.float32)
            audio_data.append(audio_chunk)
            volume = np.linalg.norm(audio_chunk)
            print("Norm:", volume)
            if volume < SILENCE_THRESHOLD and started_talking:
                print("[INFO] Silence! My turn.")
                break
            elif volume > SILENCE_THRESHOLD:
                print("I'm hearing you load and clear...")
                started_talking = True

        audio_data = np.concatenate(audio_data)

        t0 = time()

        transcription = await _async_transcribe(audio_data)
        print("transcription", transcription)

        if not transcription:
            continue

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

    await websocket.close()
