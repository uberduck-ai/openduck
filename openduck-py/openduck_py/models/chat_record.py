from datetime import datetime
from openduck_py.db import Base

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Text,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.mutable import MutableDict


class DBChatRecord(Base):
    __tablename__ = "chat_record"
    id = Column(Integer, primary_key=True)
    session_id = Column(Text, ForeignKey("chat_history.session_id", ondelete="CASCADE"), nullable=False, index=True)
    event_name = Column(Text)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    metadata = Column(MutableDict.as_mutable(JSON))
    audio = Column(Text)    # Audio recording URL in S3
    # Do I need to add relationship("DBChatHistory")?


chat_records = DBChatRecord.__table__
