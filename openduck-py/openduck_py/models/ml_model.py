from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    Integer,
    Text,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship
import torch
from openduck_py.db import Base


class DBMLModel(Base):
    __tablename__ = "ml_model"

    # TODO (Matthew): Delete unused voices here?
    MODEL_TYPE_TACOTRON2 = "tacotron2"
    MODEL_TYPE_TALKNET = "talknet"
    MODEL_TYPE_HIFI_GAN = "hifi_gan"
    MODEL_TYPE_TORCHMOJI = "torchmoji"
    MODEL_TYPE_RADTTS = "radtts"
    MODEL_TYPE_SOVITS = "sovits_4_0"
    MODEL_TYPE_RVC = "rvc"
    MODEL_TYPE_STYLETTS2 = "styletts2"
    MODEL_TYPE_RVC_DISCRIMINATOR = "rvc_discriminator"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(
        Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=True, index=True
    )
    dataset_id = Column(BigInteger, ForeignKey("dataset.id"), nullable=True, index=True)
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
    config = Column(MutableDict.as_mutable(JSONB))

    user = relationship("DBUser", backref="models")
    dataset = relationship("DBDataset", backref="models")

    def __str__(self):
        return self.uuid

    @classmethod
    def get_by_uuid(cls, uuid):
        return select(cls).where(cls.uuid == uuid)

    @classmethod
    def get_by_ids(cls, model_ids):
        return select(cls).where(cls.id.in_(model_ids))

    @classmethod
    def _base_model_filters(cls):
        model_filters = []
        return model_filters

    @classmethod
    def _base_models_query(
        cls,
        model_filters,
    ):
        from openduck_py.models import DBMLModelContributor, DBUser

        query = select(
            cls,
            func.array_agg(func.distinct(DBUser.username)).label("contributors"),
        )
        query = (
            query.join(
                DBMLModelContributor,
                # NOTE (Sam): is_outer = True since some models have no contributor
                DBMLModel.id == DBMLModelContributor.ml_model_id,
                isouter=True,
            )
            .join(DBUser, DBMLModelContributor.user_id == DBUser.id, isouter=True)
            .group_by(cls.id)
            .where(*model_filters)
        )
        return query

    @classmethod
    def get_models_query(cls, dbfilters=[]):
        # NOTE (Sam): I think these imports are necessary for the arguments we are using
        from openduck_py.models import DBMLModelContributor, DBUser

        """Return models for whom user is contributor"""
        model_filters = cls._base_model_filters()
        for i, v in enumerate(dbfilters):
            model_filters += [
                getattr(dbfilters[i][0], dbfilters[i][1]).in_(dbfilters[i][2])
            ]
        query = cls._base_models_query(model_filters)
        return query

    # TODO (Matthew): Should we be using this instead of loading.py in openduck_py/tts?
    def load(self, device=None, cache=None):
        assert self.model_type in [
            "talknet",
            "radtts",
            "sovits",
            "sovits_4_0",
            "rvc",
        ], "Only talknet, radtts, sovits, rvc are supported"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        from openduck_py.tts.loading import (
            load_talknet,
            load_radtts,
            load_sovits,
            load_rvc,
        )

        if self.model_type == "talknet":
            if cache is not None and self.uuid in cache.talknet_models:
                ml_model, _ = cache.talknet_models[self.uuid]
            else:
                ml_model = load_talknet(self.path, device=device)
            if cache is not None:
                cache.talknet_models[self.uuid] = (ml_model, ())
        elif self.model_type == "radtts":
            if cache is not None and self.uuid in cache.radtts_models:
                ml_model, _ = cache.radtts_models[self.uuid]
            else:
                ml_model = load_radtts(
                    self.path,
                    config_overrides=self.config,
                    device=device,
                )
            if cache is not None:
                cache.radtts_models[self.uuid] = (ml_model, ())
        elif self.model_type in ["sovits", "sovits_4_0"]:
            if cache is not None and self.uuid in cache.sovits_models:
                ml_model, _ = cache.sovits_models[self.uuid]
            else:
                ml_model = load_sovits(
                    self.path,
                    config_overrides=self.config,
                    device=device,
                )
            if cache is not None:
                cache.sovits_models[self.uuid] = (ml_model, ())
        elif self.model_type in ["rvc"]:
            if cache is not None and self.uuid in cache.rvc_models:
                ml_model, _ = cache.rvc_models[self.uuid]
            else:
                ml_model = load_rvc(
                    self.path,
                    config_overrides=self.config,
                    device=device,
                )
            if cache is not None:
                cache.rvc_models[self.uuid] = (ml_model, ())
        return ml_model


ml_models = DBMLModel.__table__


# TODO (Matthew): Instead of storing MODEL_TYPE_TO_FEATURES, let's consolidate freestyle-v1 and
# freestyle-v2, and add a column to DBMLModel store "modality" (e.g. text-to-voice, voice-to-voice)
TEXT_TO_VOICE = "text-to-voice"
VOICE_TO_VOICE = "voice-to-voice"
FREESTYLE_V1 = "freestyle-v1"
FREESTYLE_V2 = "freestyle-v2"
MODEL_TYPE_TO_FEATURES = {
    DBMLModel.MODEL_TYPE_TACOTRON2: [TEXT_TO_VOICE],
    DBMLModel.MODEL_TYPE_TALKNET: [FREESTYLE_V1],
    DBMLModel.MODEL_TYPE_RADTTS: [TEXT_TO_VOICE, FREESTYLE_V1],
    # NOTE (Sam): we had FREESTYLE_V2 on RVC here but it often failed to produce high fidelity voices due to mismatch in pitch between the base model and the skin
    # I think that it would be preferable for the voice (i.e. voicemodel) table to store voice generation workflows (e.g. tacotron-hifigan, bark->rvc, tacotron->hifigan->rvc)
    # those workflows would then have "features" like freestyle-v1, freestyle-v2 that indicated for example which endpoints they were suitable for.
    DBMLModel.MODEL_TYPE_RVC: [VOICE_TO_VOICE],
    DBMLModel.MODEL_TYPE_SOVITS: [VOICE_TO_VOICE],
    DBMLModel.MODEL_TYPE_STYLETTS2: [TEXT_TO_VOICE, FREESTYLE_V2],
    # NOTE (Sam): I think that MLModel is ultimately the wrong place to store these features - instead they should be stored on Voice (i.e. voicemodel).
    "bark": [],
    "styletts2->rvc": [FREESTYLE_V2],
}
