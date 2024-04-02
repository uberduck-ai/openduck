import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select


from openduck_py.db import get_db_async
from openduck_py.utils.daily import create_room


class RoomCreateResponse(BaseModel):
    url: str
    name: str
    privacy: str


class GetRecordingsResponse(BaseModel):
    recordings: List[str]


router = APIRouter(prefix="/daily")


@router.post("/rooms", response_model=RoomCreateResponse)
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
