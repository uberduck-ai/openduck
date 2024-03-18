from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io

from whisper import load_model
import numpy as np

from openduck_py.voices.styletts2 import styletts2_inference
from openduck_py.settings import OUTPUT_SAMPLE_RATE

ml_router = APIRouter(prefix="/ml")

whisper_model = load_model("base.en")


@ml_router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(..., media_type="application/octet-stream")
):
    try:
        audio_bytes = await audio.read()
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        transcription = whisper_model.transcribe(audio_data)["text"]
        return {"text": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TTSInput(BaseModel):
    text: str


@ml_router.post("/tts")
async def text_to_speech(tts_input: TTSInput):
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


app = FastAPI(title="ML Services for Openduck")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml_router)


@app.get("/status")
def status():
    return dict(status="OK")
