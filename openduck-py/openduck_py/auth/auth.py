import os
from typing import Optional

from asyncpg.exceptions import UniqueViolationError
from fastapi import Depends
from propelauth_fastapi import init_auth, User as PropelAuthUser
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from openduck_py.db import get_db_async
from openduck_py.models import DBUser

PROPEL_AUTH_API_KEY = os.environ.get("PROPEL_AUTH_API_KEY")
# PROPEL_AUTH_ENDPOINT = "https://auth.uberduck.ai"
PROPEL_AUTH_ENDPOINT = "https://5479410.propelauthtest.com"


async def get_by_propel_auth_user_id(user_id):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{PROPEL_AUTH_ENDPOINT}/api/backend/v1/user/{user_id}",
            headers=dict(Authorization=f"Bearer {PROPEL_AUTH_API_KEY}"),
        )
    return response.json()


async def propel_auth_user_to_user_sqlalchemy(
    db: AsyncSession, propel_auth_user: PropelAuthUser
):
    user = (
        await db.execute(DBUser.get(propel_auth_id=propel_auth_user.user_id))
    ).scalar()
    if user:
        return user
    else:
        # If the access_token is valid but no record exists, that means the user
        # has been created in PropelAuth but not yet in our database.
        user_data = await get_by_propel_auth_user_id(propel_auth_user.user_id)
        # NOTE(zach): Wrap in a try/catch to handle race conditions on user creation.
        try:
            user_id = (
                await db.execute(
                    DBUser.create(
                        user_data["email"],
                        username=user_data["username"],
                        propel_auth_id=propel_auth_user.user_id,
                    )
                )
            ).scalar()
            user = (await db.execute(DBUser.get(id=user_id))).scalar()
        except UniqueViolationError:
            user = (
                await db.execute(DBUser.get(propel_auth_id=propel_auth_user.user_id))
            ).scalar()
        return user


# Initialize PropelAuth
propel_auth = init_auth(
    PROPEL_AUTH_ENDPOINT,
    PROPEL_AUTH_API_KEY,
)


async def get_user_sqlalchemy(
    db: AsyncSession = Depends(get_db_async),
    propel_auth_user: Optional[PropelAuthUser] = Depends(propel_auth.optional_user),
):
    user = await propel_auth_user_to_user_sqlalchemy(db, propel_auth_user)
    return user
