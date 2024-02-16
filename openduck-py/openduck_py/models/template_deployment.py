from datetime import datetime
from uuid import uuid4
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Index
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import relationship
from openduck_py.db import Base


class DBTemplateDeployment(Base):
    __tablename__ = "template_deployment"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(Text, nullable=False, default=lambda: str(uuid4()), unique=True)
    user_id = Column(
        Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # URL compatible name like "test-table"
    url_name = Column(String, nullable=False, index=True)
    # human readable name like "Test Table"
    display_name = Column(String)
    prompt = Column(MutableDict.as_mutable(JSON))
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    deleted_at = Column(DateTime, default=None)
    meta_json = Column(MutableDict.as_mutable(JSON))
    model = Column(String)

    __table_args__ = (
        Index(
            "deployment_user_id_url_name_unique_not_deleted",
            "user_id",
            "url_name",
            unique=True,
            sqlite_where=deleted_at.is_(None),
        ),
    )

    user = relationship("DBUser", backref="user_template_deployments", viewonly=False)
