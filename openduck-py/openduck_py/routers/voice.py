import re
from fastapi import APIRouter, Depends, WebSocket
from sqlalchemy import select
import nemo.collections.asr as nemo_asr
from tempfile import NamedTemporaryFile
from time import time
from torchaudio.functional import resample
from scipy.io import wavfile

import numpy as np
from asgiref.sync import sync_to_async
import torch
from nemo_text_processing.text_normalization.normalize import Normalizer

from openduck_py.models import DBChatHistory
from openduck_py.db import get_db_async, AsyncSession
from openduck_py.voices import styletts2
from openduck_py.routers.templates import generate

asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    model_name="nvidia/stt_en_fastconformer_ctc_large"
)
normalizer = Normalizer(input_case="cased", lang="en")

audio_router = APIRouter(prefix="/audio")


def _transcribe(audio_data):
    with NamedTemporaryFile(suffix=".wav", mode="wb+") as temp_file:
        wavfile.write(temp_file.name, 16000, audio_data)
        temp_file.flush()
        temp_file.seek(0)
        transcription = asr_model.transcribe([temp_file.name])[0]
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
            # print("Received audio!")
            audio_chunk = np.frombuffer(message, dtype=np.float32)
            audio_data.append(audio_chunk)
            volume = np.linalg.norm(audio_chunk)
            # print("Norm:", volume)
            if volume < SILENCE_THRESHOLD and started_talking:
                # print("[INFO] Silence! My turn.")
                break
            elif volume > SILENCE_THRESHOLD:
                # print("I'm hearing you load and clear...")
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

        def normalize_text(text):
            normalized_text = normalizer.normalize(text)
            print("Original response:", text)
            print("Normalized response:", normalized_text)
            return normalized_text

        normalized = normalize_text(response_message.content)
        sentences = re.split(r"(?<=[.!?]) +", normalized)
        for sentence in sentences:
            # TODO: deal with asyncio
            audio_chunk = styletts2.styletts2_inference(text=sentence)
            audio_chunk_bytes = np.int16(
                audio_chunk * 32767
            ).tobytes()  # Scale to 16-bit integer values
            await websocket.send_bytes(audio_chunk_bytes)

        t_styletts = time()

        print("Fastconformer", t_whisper - t0)
        print("GPT", t_gpt - t_whisper)
        print("StyleTTS2", t_styletts - t_gpt)

    await websocket.close()
