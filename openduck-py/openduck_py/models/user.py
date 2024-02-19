from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Integer, Text
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship
from openduck_py.db import Base


class DBUser(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    propel_auth_id = Column(Text, unique=True)
    username = Column(Text, nullable=False, unique=True)
    email = Column(Text, nullable=False, unique=True)
    is_admin = Column(Boolean, default=False)
    stripe_customer_id = Column(Text, index=True)
    voices = relationship("DBVoice")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    deleted_at = Column(DateTime)

    def __str__(self):
        return self.username


users = DBUser.__table__
