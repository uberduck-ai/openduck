from datetime import datetime
from typing import Literal
from uuid import uuid4
from openduck_py.db import Base

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.dialects.postgresql import JSONB

TTS_BASIC = "tts-basic"
TTS_REFERENCE = "tts-reference"
TTS_ALL = "tts-all"
V2V = "v2v"
ALL = "all"
TTS_RAP = "tts-rap"
TTS_OPTIONS = Literal["tts-basic", "tts-reference", "tts-all", "v2v", "all", "tts-rap"]


class DBVoice(Base):
    __tablename__ = "voice"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), index=True)
    ml_model_id = Column(
        BigInteger,
        ForeignKey("ml_model.id", ondelete="CASCADE"),
        index=True,
    )
    name = Column(Text, nullable=False, index=True)
    voicemodel_uuid = Column(
        Text, nullable=False, default=lambda: str(uuid4()), unique=True
    )
    display_name = Column(Text)
    category = Column(Text)
    model_type = Column(Text)
    is_private = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    deleted_at = Column(DateTime, default=None)
    language = Column(Text, default="english", nullable=False, index=True)
    meta_json = Column(MutableDict.as_mutable(JSONB))
    age = Column(Text, index=True)
    gender = Column(Text, index=True)
    accent = Column(Text, index=True)
    mood = Column(Text, index=True)
    description = Column(Text, index=True)
    style = Column(Text, index=True)
    sample_url = Column(Text, index=True)
    image_url = Column(Text, index=True)

    user = relationship(
        "DBUser", backref="created_voices", overlaps="voices", viewonly=True
    )
    ml_model = relationship("DBMLModel", backref="voices")

    def __str__(self):
        active = "active" if self.is_active else "not active"
        hidden = "hidden" if self.is_hidden else "visible"
        return f"{self.name}_{self.model_id} ({active}) ({hidden})"


voices = DBVoice.__table__
