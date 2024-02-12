from uuid import uuid4
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Integer,
    BigInteger,
    DateTime,
    Column,
    ForeignKey,
    Text,
    select,
    Boolean,
    update,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import nullslast

from openduck_py.db import Base

# source_provider
YOUTUBE = "YOUTUBE"
TWITCH = "TWITCH"
FILE = "FILE"

# Statuses
RECEIVED = "RECEIVED"
DOWNLOADING = "DOWNLOADING"
CONVERTING = "CONVERTING"
TRANSCRIBING = "TRANSCRIBING"
CHOPPING = "CHOPPING"
CLEANING = "CLEANING"
FINISHED = "FINISHED"
ERROR = "ERROR"
PERMITTED_STATUS_ARR = [
    RECEIVED,
    DOWNLOADING,
    CONVERTING,
    TRANSCRIBING,
    CHOPPING,
    CLEANING,
    FINISHED,
    ERROR,
]


class DBDataSource(Base):
    __tablename__ = "data_source"
    id = Column(BigInteger, primary_key=True)
    uuid = Column(Text, nullable=False, unique=True, index=True)
    name = Column(Text)
    source_url = Column(Text)
    source_provider = Column(Text, index=True)  # YOUTUBE, TWITCH, FILE
    source_video_id = Column(
        Text, index=True
    )  # https://youtu.be/abcde-12fda -> abcde-12fda
    upload_path = Column(Text)
    meta_json = Column(MutableDict.as_mutable(JSONB), index=True)
    created_at = Column(DateTime, nullable=False)
    status = Column(Text)
    transcript_format = Column(Text)
    transcript_id = Column(Text)
    transcript_json = Column(MutableDict.as_mutable(JSONB))
    sentences_json = Column(MutableDict.as_mutable(JSONB))
    n_speakers = Column(Integer)

    __table_args__ = (
        UniqueConstraint(
            source_provider, source_video_id, name="source_provider_source_video_id_uc"
        ),
    )

    @classmethod
    async def create(
        cls,
        db: AsyncSession,
        name: str,
        source_url: str,
        source_provider: str,
        source_video_id: Optional[str],
        uuid: str,
        status: str,
        upload_path: Optional[str] = None,
        meta_json: Optional[dict] = None,
    ):
        item = cls(
            name=name,
            source_url=source_url,
            source_provider=source_provider,
            source_video_id=source_video_id,
            upload_path=upload_path,
            meta_json=meta_json,
            created_at=datetime.utcnow(),
            uuid=uuid,
            status=status,
        )
        db.add(item)
        await db.commit()
        return item

    @classmethod
    def get(cls, limit: int = 10, offset: int = 0):
        return (
            select(cls, func.count(cls.id).over().label("total"))
            .order_by(nullslast(cls.created_at.desc()))
            .limit(limit)
            .offset(offset)
            .group_by(cls.id)
        )

    @classmethod
    def get_by_id(cls, id: int):
        return select(cls).where(cls.id == id)

    @classmethod
    def get_by_uuid(cls, uuid: str):
        return select(cls).where(cls.uuid == uuid)

    @classmethod
    def get_by_voiced_entity_id(cls, voiced_entity_id: int):
        return select(cls).where(cls.voiced_entity_id == voiced_entity_id)

    @classmethod
    def update_status_by_id(cls, id: int, status: str):
        assert status in PERMITTED_STATUS_ARR
        return update(cls).where(cls.id == id).values(status=status)

    @classmethod
    def update_by_id(cls, id: int, **values):
        allowed_keys = [
            "transcript_format",
            "transcript_json",
            "sentences_json",
            "n_speakers",
            "status",
            "transcript_id",
        ]
        updates = {}
        for k in allowed_keys:
            if k in values:
                updates[k] = values[k]
        return update(cls).where(cls.id == id).values(**updates)

    def __str__(self):
        return f"{self.name}, {self.id}, {self.uuid}"


data_sources = DBDataSource.__table__
