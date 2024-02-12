from sqlalchemy import Column, DateTime, Integer, Text
from openduck_py.db import Base


class DBAudio(Base):
    __tablename__ = "audio"
    id = Column(Integer, primary_key=True)
    # NOTE(zach): Don't use a Foreign Key to User here so that it's easy to do
    # efficient deletions of User rows without operations on the very large
    # audio table.
    user_id = Column(Integer, index=True)
    path = Column(Text, nullable=False)
    uuid = Column(Text, unique=True, nullable=False)
    created_at = Column(DateTime, nullable=False)
    deleted_at = Column(DateTime)


audio = DBAudio.__table__
