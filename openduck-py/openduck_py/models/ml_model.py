from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    Integer,
    Text,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship
from openduck_py.db import Base


class DBMLModel(Base):
    __tablename__ = "ml_model"
    MODEL_TYPE_RADTTS = "radtts"
    MODEL_TYPE_RVC = "rvc"
    MODEL_TYPE_STYLETTS2 = "styletts2"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), index=True)
    uuid = Column(Text, nullable=False, unique=True, default=uuid4)
    model_type = Column(Text, nullable=False)
    path = Column(Text, nullable=False, unique=True)
    # NOTE(zach): keys that are used in config include:
    # - n_speakers
    # - gst_type
    # - gst_dim
    # - symbol_set
    # - has_speaker_embedding
    # TODO (Matthew): Let's make nullable columns for the keys we need and move them out of the JSON
    config = Column(JSON)

    user = relationship("DBUser", backref="models")

    def __str__(self):
        return self.uuid


ml_models = DBMLModel.__table__

TEXT_TO_VOICE = "text-to-voice"
VOICE_TO_VOICE = "voice-to-voice"
MODEL_TYPE_TO_FEATURES = {
    DBMLModel.MODEL_TYPE_RADTTS: [TEXT_TO_VOICE],
    # NOTE (Sam): we had FREESTYLE_V2 on RVC here but it often failed to produce high fidelity voices due to mismatch in pitch between the base model and the skin
    # I think that it would be preferable for the voice (i.e. voicemodel) table to store voice generation workflows (e.g. tacotron-hifigan, bark->rvc, tacotron->hifigan->rvc)
    # those workflows would then have "features" like freestyle-v1, freestyle-v2 that indicated for example which endpoints they were suitable for.
    DBMLModel.MODEL_TYPE_RVC: [VOICE_TO_VOICE],
    DBMLModel.MODEL_TYPE_STYLETTS2: [TEXT_TO_VOICE],
    # NOTE (Sam): I think that MLModel is ultimately the wrong place to store these features - instead they should be stored on Voice (i.e. voicemodel).
    "styletts2->rvc": [],
}
