from datetime import datetime
import io
import math

# from scipy.io import wavfile
from sqlalchemy import Column, DateTime, Integer, Text, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql.schema import ForeignKey

from openduck_py.db import Base

# from openduck_py.utils.s3 import upload_fileobj

# from openduck_py.settings import AUDIO_OUTPUTS_BUCKET


class DBAudio(Base):
    __tablename__ = "audio"
    id = Column(Integer, primary_key=True)
    author_id = Column(
        Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=True
    )
    s3_path = Column(Text, nullable=True)
    uuid = Column(Text, unique=True, nullable=False)
    meta = Column(MutableDict.as_mutable(JSONB))
    started_at = Column(DateTime, nullable=False)
    failed_at = Column(DateTime)
    finished_at = Column(DateTime)

    @classmethod
    def get_by_uuid(cls, uuid):
        return select(cls).where(cls.uuid == uuid)

    # def finalize_audio(
    #     self, session, audio_numpy, sr=22050, meta=None, user_id=None, text: str = ""
    # ):
    #     from openduck_py.models.user_usage import DBUserUsage

    #     bio = io.BytesIO()
    #     wavfile.write(bio, sr, audio_numpy)
    #     upload_fileobj(f"{self.uuid}/audio.wav", AUDIO_OUTPUTS_BUCKET, bio)
    #     s3_path = f"https://uberduck-audio-outputs.s3-us-west-2.amazonaws.com/{self.uuid}/audio.wav"
    #     self.s3_path = s3_path
    #     self.finished_at = datetime.utcnow()
    #     self.meta = meta
    #     session.commit()
    #     if user_id:
    #         duration = len(audio_numpy) / sr
    #         DBUserUsage.upsert(
    #             session,
    #             user_id,
    #             audio_seconds=math.ceil(duration),
    #             character_count=len(text),
    #         )

    @classmethod
    def get_by_uuid_bulk(cls, uuid_list):
        return select(cls).where(cls.uuid.in_(uuid_list))


audio = DBAudio.__table__
