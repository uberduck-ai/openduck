import io
import re
from tempfile import NamedTemporaryFile
from uuid import uuid4
from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy import select
from starlette.responses import StreamingResponse
import whisper
import base64
from scipy.io.wavfile import read, write
import numpy as np

from openduck_py.utils.third_party_tts import aio_polly_tts
from openduck_py.utils.s3 import download_file
from openduck_py.models import DBVoice, DBUser, DBChatHistory
from openduck_py.db import get_db_async, AsyncSession
from openduck_py.voices import styletts2
from pydantic import BaseModel
from openduck_py.routers.templates import generate


voice_router = APIRouter(prefix="/voice")


@voice_router.post("/text-to-speech", include_in_schema=False)
async def text_to_speech(
    db: AsyncSession = Depends(get_db_async),
):

    raise NotImplementedError

    styletts2.styletts2_inference(
        text="Hello, my name is Matthew. How are you today?",
        model_path="styletts2/rap_v1.pt",
        model_bucket="uberduck-models-us-west-2",
        config_path="styletts2/rap_v1_config.yml",
        config_bucket="uberduck-models-us-west-2",
        output_bucket="uberduck-audio-outputs",
        output_path="test.wav",
        style_prompt_path="511f17d1-8a30-4be8-86aa-4cdd8b0aed70.wav",
        style_prompt_bucket="uberduck-audio-files",
    )

    voice_uuid = "906471f3-efa1-4410-978e-c105ac4fad61"
    voice = await db.execute(
        select(DBVoice).where(DBVoice.voice_uuid == voice_uuid).limit(1)
    )
    request_id = str(uuid4())
    upload_path = f"{request_id}/output.mp3"
    text = "Il était une fois, dans un petit village pittoresque en France, deux âmes solitaires dont les chemins étaient destinés à se croiser. Juliette, une jeune fleuriste passionnée par les couleurs et les parfums de ses fleurs, passait ses journées à embellir la vie des villageois avec ses bouquets enchanteurs. De l'autre côté du village vivait Étienne, un poète timide dont les vers capturaient la beauté et la mélancolie de la vie, mais qui gardait ses poèmes pour lui, craignant qu'ils ne soient pas à la hauteur du monde extérieur."
    await aio_polly_tts(
        text=text,
        voice_id="Mathieu",
        language_code="fr-FR",
        engine="standard",
        upload_path=upload_path,
        output_format="mp3",
    )
    return dict(
        uuid=request_id,
        path=f"https://uberduck-audio-outputs.s3-us-west-2.amazonaws.com/{upload_path}",
    )


model = whisper.load_model("base")

audio_router = APIRouter(prefix="/audio")


@audio_router.post("/response", include_in_schema=False)
async def audio_response(
    session_id: str = Form(None),
    audio: UploadFile = File(None),
    db: AsyncSession = Depends(get_db_async),
    response_class=StreamingResponse,
):
    with NamedTemporaryFile() as temp_file:
        data = await audio.read()
        temp_file.write(data)
        transcription = model.transcribe(temp_file.name)["text"]

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

    response_message = response.choices[0].message

    messages.append(
        {"role": response_message.role, "content": response_message.content}
    )
    chat.history_json["messages"] = messages
    await db.commit()

    audio_chunks = []
    sentences = re.split(r"(?<=[.!?]) +", response_message.content)
    for i in range(0, len(sentences), 2):
        chunk_text = " ".join(sentences[i : i + 2])
        audio_chunk = styletts2.styletts2_inference(text=chunk_text)
        audio_chunks.append(audio_chunk)
    audio = np.concatenate(audio_chunks)
    audio = np.int16(audio * 32767)  # Scale to 16-bit integer values
    output = StreamingResponse(io.BytesIO(audio), media_type="application/octet-stream")
    return output
