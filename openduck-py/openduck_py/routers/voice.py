from uuid import uuid4
from fastapi import APIRouter, Depends
from sqlalchemy import select
from openduck_py.utils.third_party_tts import aio_polly_tts
from openduck_py.models import DBVoice, DBUser
from openduck_py.db import get_db_async, AsyncSession
import whisper

voice_router = APIRouter(prefix="/voice")


@voice_router.post("/text-to-speech", include_in_schema=False)
async def text_to_speech(
    db: AsyncSession = Depends(get_db_async),
):
    voice_uuid = "906471f3-efa1-4410-978e-c105ac4fad61"
    voice = await db.execute(
        select(DBVoice).where(DBVoice.voice_uuid == voice_uuid).limit(1)
    )
    print(voice)

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


# @voice_router.post("/audio/completion", include_in_schema=False)
# async def plushy_response(
#     db: AsyncSession = Depends(get_db_async),
# ):
#     # TODO: get the audio from the HTTP request
#     model = whisper.load_model("base")
#     text = model.transcribe(audio)
#     response_text = deployment_generate(
#         text,
#     )
#     response_audio = styletts2(response_text)
#     return dict(response_audio=response_audio)
