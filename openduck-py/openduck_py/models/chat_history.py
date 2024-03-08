from datetime import datetime
from openduck_py.db import Base

from sqlalchemy import Column, DateTime, Integer, Text, ForeignKey
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.mutable import MutableDict


class DBChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True)
    session_id = Column(Text, nullable=False, index=True)
    history_json = Column(MutableDict.as_mutable(JSON))
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    user_id = Column(
        Integer,
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
    )


chat_histories = DBChatHistory.__table__
