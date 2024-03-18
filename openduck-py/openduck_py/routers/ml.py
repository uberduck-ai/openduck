from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import io

ml_router = APIRouter(prefix="/ml")


@ml_router.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(..., media_type="audio/wav")):
    # Placeholder for actual audio transcription logic
    try:
        audio_data = await audio.read()
        transcribed_text = "This is a simulated transcription."
        return {"text": transcribed_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.post("/tts")
async def text_to_speech(text: str):
    # Placeholder for actual TTS logic
    try:
        # Simulate TTS process
        tts_audio = b"This is simulated TTS audio data."
        return StreamingResponse(
            io.BytesIO(tts_audio), media_type="application/octet-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ML Services for OpenDuck", servers=[{"url": "http://localhost:8001"}]
)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include the ML router
app.include_router(ml_router)


@app.get("/status")
def status():
    return dict(status="OK")
