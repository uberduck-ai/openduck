import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from openduck_py.utils.daily import create_room


class RoomCreateResponse(BaseModel):
    url: str
    name: str
    privacy: str


router = APIRouter(prefix="/rooms")


@router.post("/", response_model=RoomCreateResponse)
async def create():
    try:
        room_info = await create_room()
        return RoomCreateResponse(
            url=room_info["url"],
            name=room_info["name"],
            privacy=room_info["privacy"],
        )
    except Exception as e:
        logging.exception("Failed to create room: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to create room")
