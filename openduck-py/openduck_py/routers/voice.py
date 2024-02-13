import io
from uuid import uuid4
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from openduck_py.utils.third_party_tts import (
    aio_polly_tts,
    aio_azure_tts,
    aio_google_tts,
)
from openduck_py.models import DBVoice
from openduck_py.db import get_db_async, AsyncSession
from openduck_py.db import aio_first_scalar
from openduck_py.utils.s3 import upload_to_s3_bucket

voice_router = APIRouter(prefix="/voice")


class TTSRequest(BaseModel):
    # TODO (Matthew): Add checks for string length
    text: str
    voice_uuid: str


class TTSResponse(BaseModel):
    path: str
    uuid: str


@voice_router.post("/text-to-speech", include_in_schema=False)
async def text_to_speech(
    request: TTSRequest,
    db: AsyncSession = Depends(get_db_async),
) -> TTSResponse:
    """Generate speech from text"""
    voice = await aio_first_scalar(db, DBVoice.get(voice_uuid=request.voice_uuid))
    voice_uuid = request.voice_uuid
    # TODO(Matthew): Add mixpanel
    request_id = str(uuid4())
    if voice_uuid.startswith("polly_"):
        upload_path = f"{request_id}/output.mp3"
        await aio_polly_tts(
            request.text,
            voice.meta_json["Id"],
            voice.meta_json["LanguageCode"],
            "neural" if "neural" in voice.meta_json["SupportedEngines"] else "standard",
            upload_path,
            "mp3",
        )
    elif voice_uuid.startswith("azure_"):
        upload_path = f"{request_id}/output.wav"
        await aio_azure_tts(request.text, voice.meta_json["short_name"], upload_path)
    else:
        language_code = voice.meta_json["language_codes"][0]
        result = await aio_google_tts(
            request.text, voice.meta_json["name"], language_code
        )
        bio = io.BytesIO(result.audio_content)
        upload_path = f"{request_id}/output.mp3"
        await upload_to_s3_bucket(bio, "uberduck-audio-outputs", upload_path)
    return TTSResponse(
        uuid=request_id,
        path=f"https://uberduck-audio-outputs.s3-us-west-2.amazonaws.com/{upload_path}",
    )
