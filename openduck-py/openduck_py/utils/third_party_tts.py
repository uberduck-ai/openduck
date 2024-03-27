import os
from typing import AsyncGenerator

import httpx

import openai

elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")


def elevenlabs_tts():
    raise NotImplementedError("Elevenlabs TTS is not implemented yet")


ELEVENLABS_VIKRAM = "gKhGpodmvg3JEngzD7eI"
ELEVENLABS_CHRIS = "iP95p4xoKVk53GoZ742B"

CHUNK_SIZE = 8096


async def aio_elevenlabs_tts(
    text, voice_id=ELEVENLABS_VIKRAM
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
        async for chunk in result.aiter_bytes(chunk_size=CHUNK_SIZE):
            yield chunk


async def aio_openai_tts(
    text, model="tts-1", voice="alloy"
) -> AsyncGenerator[bytes, None]:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    async with httpx.AsyncClient() as client:
        result = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": text,
                "voice": voice,
                "response_format": "pcm",
            },
        )
        result.raise_for_status()
        async for chunk in result.aiter_bytes(chunk_size=CHUNK_SIZE):
            yield chunk
