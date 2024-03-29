"""Synthesize speech with third-party TTS services.

All functions in this module return an async generator that yields chunks 24khz pcm audio.
"""

import asyncio
import io
import librosa
import numpy as np
import os
from typing import AsyncGenerator

import httpx
import azure.cognitiveservices.speech as azure_speechsdk
import openai

elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
GPT_SOVITS_API_URL = os.environ.get("GPT_SOVITS_API_URL")


def elevenlabs_tts():
    raise NotImplementedError("Elevenlabs TTS is not implemented yet")


ELEVENLABS_VIKRAM = "gKhGpodmvg3JEngzD7eI"
ELEVENLABS_CHRIS = "iP95p4xoKVk53GoZ742B"

CHUNK_SIZE = 8192


async def aio_gptsovits_tts(text, voice_ref) -> AsyncGenerator[bytes, None]:
    result = httpx.get(
        GPT_SOVITS_API_URL,
        params={
            "refer_wav_path": voice_ref,
            "prompt_text": "Abandon all aspirations for any kind of cohesive architecture,",
            "prompt_language": "en",
            "text": text,
            "text_language": "en",
        },
    )
    result.raise_for_status()
    wav, _ = librosa.load(io.BytesIO(result.content), sr=24000)
    bytes = np.int16(wav * 32767).tobytes()
    chunk_size = 16384
    for chunk in [bytes[i : i + chunk_size] for i in range(0, len(bytes), chunk_size)]:
        yield chunk


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


AZURE_ABEO = "en-NG-AbeoNeural"


async def aio_azure_tts(
    text: str,
    voice_name: str = AZURE_ABEO,
    chunk_size=CHUNK_SIZE,
) -> AsyncGenerator[bytes, None]:
    speech_config = azure_speechsdk.SpeechConfig(
        subscription=os.environ["AZURE_SPEECH_KEY"], region="westus"
    )
    speech_config.speech_synthesis_voice_name = voice_name
    speech_config.set_speech_synthesis_output_format(
        azure_speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
    )

    # Create an instance of a speech synthesizer using the default speaker as audio output.
    synthesizer = azure_speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=None
    )

    def _start_speaking():
        return synthesizer.start_speaking_text_async(text).get()

    result = await asyncio.get_event_loop().run_in_executor(None, _start_speaking)
    stream = azure_speechsdk.AudioDataStream(result)
    audio_buffer = bytes(chunk_size)
    total_size = 0
    filled_size = stream.read_data(audio_buffer)
    while filled_size > 0:
        total_size += filled_size
        yield bytes(bytearray(audio_buffer[:filled_size]))
        filled_size = stream.read_data(audio_buffer)
