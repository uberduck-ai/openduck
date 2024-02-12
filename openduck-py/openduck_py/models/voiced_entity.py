from sqlalchemy import Column, BigInteger, Text, select
from sqlalchemy_searchable import search
from sqlalchemy_utils.types import TSVectorType

from openduck_py.db import Base


class DBEntity(Base):
    __tablename__ = "voiced_entity"

    id = Column(BigInteger, primary_key=True)
    short_name = Column(Text, nullable=False, unique=True)
    name = Column(Text, nullable=False, unique=True)
    description = Column(Text)

    search_vector = Column(TSVectorType("short_name", "name"))

    def __str__(self):
        return self.name

    @classmethod
    def get_by_name(cls, name):
        return select(cls).where(cls.name == name)

    @classmethod
    def get_by_short_name(cls, name):
        return select(cls).where(cls.short_name == name)

    @classmethod
    def get_by_tags(cls, tags):
        return select(cls).where(cls.short_name.in_(tags))

    @classmethod
    def get_by_id(cls, id: int):
        return select(cls).where(cls.id == id)

    @classmethod
    def search(cls, query, limit):
        results = select(cls)
        results = search(results, query, sort=True).limit(limit)
        return results


voiced_entities = DBEntity.__table__
