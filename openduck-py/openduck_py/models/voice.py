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
    # TODO (Sam): let's think of these as voicemodel types
    MODEL_TYPE_TACOTRON2 = "tacotron2"
    MODEL_TYPE_TALKNET = "talknet"
    __tablename__ = "voice"
    id = Column(Integer, primary_key=True)
    author_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), index=True)
    dataset_id = Column(
        BigInteger, ForeignKey("dataset.id", ondelete="CASCADE"), index=True
    )
    ml_model_id = Column(
        BigInteger,
        ForeignKey("ml_model.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    voiced_entity_id = Column(
        BigInteger, ForeignKey("voiced_entity.id", ondelete="CASCADE"), index=True
    )
    name = Column(Text, nullable=False, index=True)
    # TODO (Matthew): The uuid4 function returns a UUID object, which is not compatible with Text
    voicemodel_uuid = Column(Text, nullable=False, default=uuid4, unique=True)
    display_name = Column(Text)
    category = Column(Text)
    voice_actor = Column(Text)
    model_id = Column(Text, nullable=True, default=uuid4, index=True)
    model_type = Column(Text, nullable=True, default="tacotron2")
    hifi_gan_vocoder = Column(Text, nullable=True, default="")
    is_arpabet = Column(Boolean, nullable=True, default=False)
    is_primary = Column(Boolean)
    is_active = Column(Boolean, default=False)
    is_hidden = Column(Boolean, default=False)
    is_priority = Column(Boolean, default=False)
    is_commercial = Column(
        Boolean, default=False, server_default="false", nullable=False
    )
    gate_threshold = Column(Float)
    speaker_id = Column(BigInteger)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    added_at = Column(DateTime, default=None)
    deleted_at = Column(DateTime, default=None)
    language = Column(Text, default="english", nullable=False, index=True)
    meta_json = Column(MutableDict.as_mutable(JSONB))
    user = relationship(
        "DBUser", backref="created_voices", overlaps="voices", viewonly=True
    )
    ml_model = relationship("DBMLModel", backref="voices")
    voiced_entity = relationship("DBEntity", backref="voices")

    age = Column(Text, index=True)
    gender = Column(Text, index=True)
    accent = Column(Text, index=True)
    mood = Column(Text, index=True)
    description = Column(Text, index=True)
    style = Column(Text, index=True)
    sample_url = Column(Text, index=True)
    image_url = Column(Text, index=True)

    def __str__(self):
        active = "active" if self.is_active else "not active"
        hidden = "hidden" if self.is_hidden else "visible"
        return f"{self.name}_{self.model_id} ({active}) ({hidden})"


voices = DBVoice.__table__
