from fastapi import FastAPI
from openduck_py.routers.voice import voice_router

# TODO (Matthew): Change
openapi_url = None
IS_DEV = True
if IS_DEV:
    servers = [{"url": "http://localhost:8003"}]
else:
    servers = [{"url": "https://api.uberduck.ai"}]
app = FastAPI(
    title="Uberduck Text To Speech API",
    openapi_url=openapi_url,
    servers=servers,
    debug=IS_DEV,
)
app.include_router(voice_router)
