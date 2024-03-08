from openduck_py.db import Base

from sqlalchemy import (
    Column,
    Integer,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.mutable import MutableDict


class DBUser(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    meta_json = Column(MutableDict.as_mutable(JSON))


users = DBUser.__table__
