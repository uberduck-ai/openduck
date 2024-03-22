import os
from typing import AsyncGenerator

import httpx

elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")


def elevenlabs_tts():
    raise NotImplementedError("Elevenlabs TTS is not implemented yet")


ELEVENLABS_VIKRAM = "gKhGpodmvg3JEngzD7eI"
ELEVENLABS_CHRIS = "iP95p4xoKVk53GoZ742B"


async def aio_elevenlabs_tts(
    text, voice_id="gKhGpodmvg3JEngzD7eI"
) -> AsyncGenerator[bytes, None]:
    if elevenlabs_api_key is None:
        raise ValueError("ELEVENLABS_API_KEY is not set")
    async with httpx.AsyncClient() as client:
        result = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=pcm_24000",
            headers={
                "xi-api-key": elevenlabs_api_key,
            },
            json={"text": text},
        )
        result.raise_for_status()
        async for chunk in result.aiter_bytes(chunk_size=1024):
            yield chunk
