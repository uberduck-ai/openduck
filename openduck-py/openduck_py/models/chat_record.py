from datetime import datetime
from typing import Literal

from sqlalchemy import Column, DateTime, ForeignKey, Integer, Text, Float
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped
from openduck_py.db import Base

EventName = Literal[
    "started_session",
    "ended_session",
    "received_audio",
    "sent_audio",
    "detected_start_of_speech",
    "detected_end_of_speech",
    "started_response",
    "interrupted_response",
    "transcribed_audio",
    "generated_completion",
    "normalized_text",
    "generated_tts",
    "removed_echo",
]


class DBChatRecord(Base):
    """An event that occurred during a chat session."""

    __tablename__ = "chat_record"
    id = Column(Integer, primary_key=True)
    session_id = Column(
        Text,
        ForeignKey("chat_history.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_name = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    meta_json = Column(MutableDict.as_mutable(JSON))
    latency_seconds = Column(Float)


chat_records = DBChatRecord.__table__
