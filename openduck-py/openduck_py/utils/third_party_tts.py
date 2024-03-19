import os
from typing import AsyncGenerator

import httpx

elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")


def elevenlabs_tts():
    raise NotImplementedError("Elevenlabs TTS is not implemented yet")


async def aio_elevenlabs_tts(text) -> AsyncGenerator[bytes, None]:
    if elevenlabs_api_key is None:
        raise ValueError("ELEVENLABS_API_KEY is not set")
    async with httpx.AsyncClient() as client:
        result = await client.post(
            "https://api.elevenlabs.io/v1/text-to-speech/gKhGpodmvg3JEngzD7eI?output_format=pcm_24000",
            headers={
                "xi-api-key": elevenlabs_api_key,
            },
            json={"text": text},
        )
        result.raise_for_status()
        async for chunk in result.aiter_bytes(chunk_size=1024):
            yield chunk
