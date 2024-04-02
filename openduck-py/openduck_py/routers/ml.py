from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io

from whisper import load_model
import numpy as np
from nemo_text_processing.text_normalization.normalize import Normalizer

from openduck_py.voices.styletts2 import styletts2_inference
from openduck_py.settings import OUTPUT_SAMPLE_RATE, IS_DEV, NO_SPEECH_PROB_THRESHOLD

ml_router = APIRouter(prefix="/ml")

whisper_model = load_model("medium.en")

# TODO (Matthew): Load the normalizer on IS_DEV but change the docker-compose to only reload the ML
# service if this file is changed
if IS_DEV:
    normalize_text_fn = lambda x: x
else:
    normalizer = Normalizer(input_case="cased", lang="en")
    normalize_text_fn = normalizer.normalize


class TextInput(BaseModel):
    text: str


@ml_router.post("/normalize")
async def normalize_text(text: TextInput):
    return {"text": normalize_text_fn(text.text)}


@ml_router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(..., media_type="application/octet-stream")
):
    """
    Transcribes the given audio using OpenAI Whisper base.en. Audio should be a float32 bytes array
    sampled at 16 kHz.
    """
    try:
        audio_bytes = await audio.read()
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        response = whisper_model.transcribe(audio_data)
        if len(response["segments"]) == 0:
            return {"text": ""}
        no_speech_prob = response["segments"][0]["no_speech_prob"]
        print("No speech prob:", no_speech_prob)
        transcription = response["text"]
        if no_speech_prob > NO_SPEECH_PROB_THRESHOLD:
            transcription = ""
        return {"text": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.post("/tts")
async def text_to_speech(tts_input: TextInput):
    try:
        audio_chunk = styletts2_inference(
            text=tts_input.text,
            output_sample_rate=OUTPUT_SAMPLE_RATE,
        )
        audio_chunk = np.int16(audio_chunk * 32767).tobytes()
        return StreamingResponse(
            io.BytesIO(audio_chunk), media_type="application/octet-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
