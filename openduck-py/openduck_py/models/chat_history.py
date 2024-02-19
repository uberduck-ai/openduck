
from datetime import datetime
from uuid import uuid4
from openduck_py.db import Base

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.mutable import MutableDict



class DBChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), index=True)
    session_id = Column(Text, nullable=False)
    history_json = Column(MutableDict.as_mutable(JSON))
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    user = relationship(
        "DBUser", backref="created_chats", overlaps="chat_histories", viewonly=True
    )


chat_histories = DBChatHistory.__table__
