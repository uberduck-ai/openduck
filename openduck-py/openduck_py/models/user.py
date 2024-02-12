from datetime import datetime
from random import randint
import numbers

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    Index,
    Text,
    select,
    update,
    func,
    or_,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship
from sqlalchemy_utils.types.password import PasswordType

from openduck_py.db import Base


class DBUser(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    propel_auth_id = Column(Text)
    username = Column(Text, nullable=False, unique=True)
    username_lower = Column(Text, nullable=False, unique=True)
    email = Column(Text, nullable=False, unique=True)
    password = Column(PasswordType(schemes=["argon2"]), nullable=True)
    is_admin = Column(Boolean, default=False)
    is_approved_submitter = Column(Boolean, default=False)
    is_priority = Column(Boolean, default=False)
    # An integer from 0 to 999, used for segmenting the userbase into random groups.
    cohort = Column(Integer, index=True, nullable=False)
    stripe_customer_id = Column(Text, nullable=True, index=True)
    voices = relationship("DBVoice")
    audio_files = relationship("DBAudio")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    banned_at = Column(DateTime)
    deleted_at = Column(DateTime)
    is_confirmed = Column(Boolean, default=False)
    flags = Column(MutableDict.as_mutable(JSONB))
    marketing_opt_in = Column(Boolean, default=False)
    # NOTE (Sam): see here (https://stackoverflow.com/questions/32383685/sqlalchemy-date-error-argument-arg-is-expected-str-but-got-int) for why this default must be a string
    stable_diffusion_use_count = Column(Integer, nullable=False, server_default="0")

    __table_args__ = (Index("unique_propel_auth_id", propel_auth_id, unique=True),)

    @classmethod
    def get_by_propel_auth_id(cls, propel_auth_id: str):
        return select(cls).where(cls.propel_auth_id == propel_auth_id)

    @classmethod
    def get_by_email(cls, email: str):
        return select(cls).where(cls.email == email)

    @classmethod
    def get_by_id(cls, id: int):
        return select(cls).where(cls.id == id)

    @classmethod
    def get_by_stripe_id(cls, stripe_customer_id):
        return select(cls).where(cls.stripe_customer_id == stripe_customer_id)

    @classmethod
    def get_by_username(cls, username: str):
        return select(cls).where(cls.username_lower == username.lower())

    @classmethod
    def soft_delete(cls, user_id):
        return (
            update(users)
            .where(users.c.id == user_id)
            .values(deleted_at=datetime.utcnow())
        )

    @classmethod
    def update_username(cls, id_, username):
        return cls.update_by_id(id_, username=username, username_lower=username.lower())

    @classmethod
    def update_by_id(cls, id_, **values):
        allowed_keys = [
            "email",
            "username",
            "stable_diffusion_use_count",
            "propel_auth_id",
        ]
        updates = {}
        for k in allowed_keys:
            if k in values:
                updates[k] = values[k]
                if k == "username":
                    updates["username_lower"] = values[k].lower()
        users = cls.__table__
        return update(users).where(users.c.id == id_).values(**updates)

    @classmethod
    def update_by_email(cls, email, **values):
        allowed_keys = [
            "is_confirmed",
            "password",
            "stripe_customer_id",
        ]
        updates = {}
        for k in allowed_keys:
            if k in values:
                updates[k] = values[k]
        users = cls.__table__
        return update(users).where(users.c.email == email).values(**updates)

    @classmethod
    def bulk_update_flags(cls, key, value, *where_filters):
        if isinstance(value, str):
            json_value = f'"{value}"'
        elif isinstance(value, numbers.Number):
            json_value = f"{value}"
        else:
            raise Exception(f"value must be string or number, got {type(value)}")
        return (
            update(cls)
            .values(
                flags=func.jsonb_set(
                    func.coalesce(cls.flags, "{}"),
                    "{%s}" % key,
                    json_value,
                )
            )
            .where(*where_filters)
        )

    @classmethod
    def get_flags_by_username(cls, username):
        return select(cls.flags).where(cls.username_lower == username.lower())

    @classmethod
    def set_flags_by_username(cls, username, flags):
        return (
            update(cls)
            .values(flags=flags)
            .where(cls.username_lower == username.lower())
        )

    @classmethod
    def get_flags_by_id(cls, id_):
        return select(cls.flags).where(cls.id == id_)

    @classmethod
    def set_flags_by_id(cls, id_, flags):
        return update(cls).values(flags=flags).where(cls.id == id_)

    @classmethod
    def get_marketplace_tasks(cls, user_id):
        # Avoid circular imports
        from uberduck_py.models.marketplace_task import DBMarketplaceTask
        from uberduck_py.models.marketplace_task_type import DBMarketplaceTaskType

        return (
            select(DBMarketplaceTask, DBMarketplaceTaskType.name.label("type"))
            .join(DBMarketplaceTaskType)
            .where(
                DBMarketplaceTask.deleted_at == None,
                DBMarketplaceTask.user_id == user_id,
            )
        )

    @property
    def select_current_usage(self):
        from uberduck_py.models.user_usage import DBUserUsage

        now = datetime.utcnow()
        usage = select(DBUserUsage).where(
            DBUserUsage.user_id == self.id,
            DBUserUsage.billing_period_start <= now,
            or_(
                DBUserUsage.billing_period_end >= now,
                DBUserUsage.billing_period_end == None,
            ),
        )
        return usage, now

    def __str__(self):
        return self.username


users = DBUser.__table__
