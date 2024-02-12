from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from openduck_py.routers.voice import voice_router
from openduck_py.routers.call import call_router

# TODO (Matthew): Change
openapi_url = None
IS_DEV = True
if IS_DEV:
    servers = [{"url": "http://localhost:8000"}]
else:
    servers = [{"url": "https://api.uberduck.ai"}]

app = FastAPI(
    title="Openduck: The easiest way to build interactive multimodal AI applications.",
    openapi_url=openapi_url,
    servers=servers,
    debug=IS_DEV,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(call_router)
app.include_router(voice_router)

@app.get("/status")
def status():
    return dict(status="OK")