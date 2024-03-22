from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openduck_py.routers.voice import audio_router
from openduck_py.routers.rooms import router as rooms_router
from openduck_py.routers.ml import ml_router
from openduck_py.settings import IS_DEV

if IS_DEV:
    servers = [{"url": "http://localhost:8000"}]
else:
    servers = [{"url": "https://api.uberduck.ai"}]
app = FastAPI(
    title="Uberduck Text To Speech API",
    servers=servers,
    debug=IS_DEV,
)

# Add CORS middleware to allow requests from localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://openduck.vercel.app",
        "https://openduck.ai",
        "https://b898bccd7757.ngrok.app",
        "https://uberduck.ngrok.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(audio_router)
app.include_router(rooms_router)
app.include_router(ml_router)


@app.get("/status")
def status():
    return dict(status="OK")
