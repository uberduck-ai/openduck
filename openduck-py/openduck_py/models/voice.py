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
    # NOTE(Matthew): These will have prefixes assigned in get_voice_uuid()
    voice_uuid = Column(Text, nullable=False, default=lambda: str(uuid4()), unique=True)
    display_name = Column(Text)
    category = Column(Text)
    model_type = Column(Text)
    is_private = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    deleted_at = Column(DateTime, default=None)
    language = Column(Text, default="english", nullable=False, index=True)
    meta_json = Column(JSON)
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

    def get_voice_uuid(self):
        pass


voices = DBVoice.__table__
