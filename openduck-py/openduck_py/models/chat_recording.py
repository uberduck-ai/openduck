from datetime import datetime
from typing import Literal

from sqlalchemy import Column, DateTime, ForeignKey, Integer, Text, Float
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.mutable import MutableDict
from openduck_py.db import Base


class DBChatRecording(Base):
    """An event that occurred during a chat session."""

    __tablename__ = "chat_recording"
    id = Column(Integer, primary_key=True)
    uuid = Column(
        Text, nullable=False, unique=True, index=True
    )  # NOTE(wrl): Daily room ID
    url = Column(Text, nullable=False)
    chat_session_id = Column(
        Text,
        ForeignKey("chat_history.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


chat_recordings = DBChatRecording.__table__
