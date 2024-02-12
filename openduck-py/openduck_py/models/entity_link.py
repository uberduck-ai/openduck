from datetime import datetime

from sqlalchemy import (
    and_,
    Column,
    BigInteger,
    DateTime,
    ForeignKey,
    Text,
    select,
    UniqueConstraint,
)
from sqlalchemy_searchable import search
from sqlalchemy.orm import relationship
from openduck_py.db import Base
from openduck_py.models import DBEntity, DBVoice, DBVoiceImage

IMAGE = "image"
VOICE = "voice"
VALID_CONTENT_TYPES = {
    IMAGE,
    VOICE,
    "reference_audio",
}


class DBEntityLink(Base):
    __tablename__ = "entity_link"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("user.id", ondelete="SET NULL"), index=True)
    content_type = Column(Text, index=True)
    content_type_id = Column(BigInteger, index=True)
    entity_id = Column(
        BigInteger, ForeignKey("voiced_entity.id", ondelete="CASCADE"), index=True
    )
    approved_at = Column(DateTime, default=None, index=True)
    rejected_at = Column(DateTime, default=None, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    deleted_at = Column(DateTime, default=None)

    entity = relationship("DBEntity", backref="entity_links")
    user = relationship("DBUser", backref="entity_links")

    __table_args__ = (
        UniqueConstraint(
            content_type, content_type_id, entity_id, name="content_entity_uc"
        ),
    )

    @classmethod
    def get_by_id(cls, id_):
        return select(DBEntityLink).where(DBEntityLink.id == id_)

    @classmethod
    def create(cls, entity_id, content_type, content_type_id, user_id=None):
        if content_type not in VALID_CONTENT_TYPES:
            raise Exception(f"Invalid content type: {content_type}")
        entity_links = cls.__table__
        query = entity_links.insert().values(
            entity_id=entity_id,
            content_type=content_type,
            content_type_id=content_type_id,
            created_at=datetime.utcnow(),
            user_id=user_id,
        )
        return query

    @classmethod
    def get_for_voice(cls, voice_id):
        query = (
            select(DBEntity, cls, DBVoice)
            .join(cls, cls.entity_id == DBEntity.id)
            .join(
                DBVoice,
                and_(cls.content_type == VOICE, cls.content_type_id == DBVoice.id),
            )
            .where(DBVoice.id == voice_id, cls.content_type == VOICE)
        )
        return query

    @classmethod
    def get_images_by_entity_ids(cls, entity_ids, limit=10):
        query = (
            select(cls, DBVoiceImage)
            .join(
                DBVoiceImage,
                and_(cls.content_type == IMAGE, cls.content_type_id == DBVoiceImage.id),
            )
            .where(cls.entity_id.in_(entity_ids), cls.approved_at != None)
            .order_by(cls.created_at.desc())
            .limit(limit)
        )
        return query

    @classmethod
    def search_images_by_tag(cls, query, limit, offset):
        results = (
            select(DBEntity, DBVoiceImage)
            .join(cls, DBEntity.id == cls.entity_id)
            .join(
                DBVoiceImage,
                and_(cls.content_type == IMAGE, cls.content_type_id == DBVoiceImage.id),
            )
            .where(cls.content_type == IMAGE)
        )
        results = search(results, query, sort=True)
        results = results.limit(limit).offset(offset)
        return results
