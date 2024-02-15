from datetime import datetime
from uuid import uuid4
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Index, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.dialects.sqlite import JSON
from openduck_py.db import Base


class DBTemplatePrompt(Base):
    __tablename__ = "template_prompt"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(
        Text, nullable=False, default=lambda: str(uuid4()), unique=True, index=True
    )
    user_id = Column(
        Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # URL compatible name like "test-table"
    url_name = Column(String, nullable=False, index=True)
    # human readable name like "Test Table"
    display_name = Column(String)
    prompt = Column(JSON)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    deleted_at = Column(DateTime)
    meta_json = Column(JSON)
    model = Column(String)

    __table_args__ = (
        Index(
            "prompt_user_id_url_name_unique_not_deleted",
            "user_id",
            "url_name",
            unique=True,
            sqlite_where=deleted_at.is_(None),
        ),
    )

    user = relationship("DBUser", backref="user_template_prompts")

    def __str__(self):
        return f"{self.url_name} {self.deleted_at}"
