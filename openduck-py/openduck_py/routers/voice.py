from fastapi import APIRouter

voice_router = APIRouter(prefix="/voice")


@voice_router.post("/hello-world", include_in_schema=False)
async def hello_world():
    return {"message": "Hello World"}
