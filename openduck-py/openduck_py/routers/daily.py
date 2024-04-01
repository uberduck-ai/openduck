import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select


from openduck_py.db import get_db_async
from openduck_py.utils.daily import create_room
from openduck_py.models import DBChatRecording


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


@router.get("/recordings/{user_session_id}", response_model=GetRecordingsResponse)
async def get_recordings(
    user_session_id: str, async_db: AsyncSession = Depends(get_db_async)
):
    result = await async_db.execute(
        select(DBChatRecording).where(
            DBChatRecording.chat_session_id == user_session_id
        )
    )
    recordings = result.scalars().all()
    recording_urls = [recording.url for recording in recordings]
    return GetRecordingsResponse(
        recordings=recording_urls,
    )
