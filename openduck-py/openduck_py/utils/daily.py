import os
import time

import httpx
from pydantic import BaseModel

DAILY_API_KEY = os.environ.get("DAILY_API_KEY")
print("DAILY_API_KEY", DAILY_API_KEY)


class RoomCreateResponse(BaseModel):
    url: str
    name: str
    privacy: str


async def create_room(exp=None) -> dict:
    headers = {
        "Authorization": f"Bearer {DAILY_API_KEY}",
        "Content-Type": "application/json",
    }
    if exp is None:
        # Default room lifetime is 1 hour.
        exp = int(time.time()) + 3600
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.daily.co/v1/rooms",
            headers=headers,
            json={
                "properties": {
                    "enable_chat": True,
                    "start_video_off": True,
                    "start_audio_off": False,
                    "exp": exp,
                }
            },
        )
    response.raise_for_status()
    return response.json()
