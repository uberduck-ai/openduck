import os
from typing import Dict, Optional

from asyncpg.exceptions import UniqueViolationError
from fastapi import HTTPException, Request, status, Security, Depends
from fastapi.openapi.models import OAuthFlows
from fastapi.security import OAuth2, HTTPBasic, HTTPBasicCredentials
from fastapi.security.utils import get_authorization_scheme_param
from passlib.context import CryptContext
from propelauth_fastapi import init_auth, User as PropelAuthUser
from sqlalchemy.ext.asyncio import AsyncSession

from openduck_py.db import get_db_async
from openduck_py.models import DBUser
from openduck_py.utils.propel_auth import get_by_propel_auth_user_id
from openduck_py.utils.utils import track_async

SECRET_KEY = "9bcd18098bed0153617a2ee039ae9d8c827ebb9d5c259710f4aaccfb389fda46"
ALGORITHM = "HS256"

PROPEL_AUTH_API_KEY = os.environ.get("PROPEL_AUTH_API_KEY")
PROPEL_AUTH_ENDPOINT = "https://auth.uberduck.ai"


pwd_context = CryptContext(schemes=["argon2"])


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


async def propel_auth_user_to_user_sqlalchemy(
    db: AsyncSession, propel_auth_user: PropelAuthUser
):
    user = (
        await db.execute(DBUser.get_by_propel_auth_id(propel_auth_user.user_id))
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
                        is_confirmed=user_data["email_confirmed"],
                        propel_auth_id=propel_auth_user.user_id,
                    )
                )
            ).scalar()
            user = (await db.execute(DBUser.get_by_id(user_id))).scalar()
        except UniqueViolationError:
            user = (
                await db.execute(DBUser.get_by_propel_auth_id(propel_auth_user.user_id))
            ).scalar()
        return user


class OAuth2PasswordImplicit(OAuth2):
    def __init__(
        self,
        tokenUrl: str,
        authorizationUrl: str,
        scheme_name: Optional[str] = None,
        scopes: Optional[Dict[str, str]] = None,
        auto_error: bool = True,
    ):
        if not scopes:
            scopes = {}
        flows = OAuthFlows(
            password={"tokenUrl": tokenUrl, "scopes": scopes},
            implicit={"authorizationUrl": authorizationUrl},
        )
        super().__init__(flows=flows, scheme_name=scheme_name, auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        authorization: str = request.headers.get("Authorization")

        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != "bearer":
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                return None
        return param


class OAuth2PasswordBearerWithCookie(OAuth2):
    def __init__(
        self,
        tokenUrl: str,
        scheme_name: Optional[str] = None,
        scopes: Optional[Dict[str, str]] = None,
        auto_error: bool = True,
    ):
        if not scopes:
            scopes = {}
        flows = OAuthFlows(password={"tokenUrl": tokenUrl, "scopes": scopes})
        super().__init__(flows=flows, scheme_name=scheme_name, auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        authorization: str = request.cookies.get(
            "access_token"
        )  # changed to accept access token from httpOnly Cookie

        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != "bearer":
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                return None
        return param


oauth2_scheme = OAuth2PasswordImplicit(
    tokenUrl="/token",
    authorizationUrl="http://localhost:3000/login",
    auto_error=False,
)
oauth2_cookie_scheme = OAuth2PasswordBearerWithCookie(
    tokenUrl="/token", auto_error=False
)
basic_auth_scheme = HTTPBasic(auto_error=False)

# Initialize PropelAuth
propel_auth = init_auth(
    PROPEL_AUTH_ENDPOINT,
    PROPEL_AUTH_API_KEY,
)


class GetUserSQLAlchemy:
    """A GetUser implementation that does not depend on the databases module.

    The goal is to move away from databases and use pure SQLAlchemy.
    """

    def __init__(
        self,
        allow_basic: bool,
        auto_error: bool = True,
        allow_uncomfirmed: bool = False,
        internal: bool = False,
    ):
        self.allow_basic = allow_basic
        self.auto_error = auto_error
        self.allow_unconfirmed = allow_uncomfirmed
        self.internal = internal

    async def _api_key_to_user(self, db: AsyncSession, key: str, secret: str):
        db_user = (await db.execute(DBUser.get_by_api_key(key, secret))).scalar()
        return db_user

    async def _get_user(
        self,
        db: AsyncSession,
        cookie_token: str,
        token: str,
        api_key: HTTPBasicCredentials,
        propel_auth_user: Optional[PropelAuthUser],
    ):
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

        # TODO (Sam): consider adding exception for not (cookie_token ^ token ^ api_key)
        # NOTE(zach): Add this back later.
        # if self.allow_basic and api_key:
        #     user = await api_key_to_user(api_key.username, api_key.password)
        #     if not user or (not self.allow_unconfirmed and not user.is_confirmed):
        #         raise credentials_exception
        #     await track_async(user.email, "Used API Key")
        #     return user

        if propel_auth_user:
            user = await propel_auth_user_to_user_sqlalchemy(db, propel_auth_user)
            # NOTE (Sam): need something to avoid getting by paywall using cookie
            if api_key:
                user.api_user = True
            else:
                user.api_user = False
            if self.internal and not user.email.endswith("@uberduck.ai"):
                raise credentials_exception
            return user
        if not cookie_token and not token and not api_key:
            raise credentials_exception

        assert self.allow_basic and api_key
        user = await self._api_key_to_user(db, api_key.username, api_key.password)
        if user:
            await track_async(
                user.email,
                "Used API Key",
                {"apiPublicKey": api_key.username},
            )

        if (
            not user
            or user.deleted_at is not None
            or (self.internal and not user.email.endswith("@uberduck.ai"))
        ):
            raise credentials_exception

        if api_key:
            user.api_user = True
        else:
            user.api_user = False

        return user

    async def __call__(
        self,
        cookie_token: str = Security(oauth2_cookie_scheme),
        token: str = Security(oauth2_scheme),
        api_key: HTTPBasicCredentials = Security(basic_auth_scheme),
        db: AsyncSession = Depends(get_db_async),
        propel_auth_user: Optional[PropelAuthUser] = Depends(propel_auth.optional_user),
    ):
        try:
            user = await self._get_user(
                db, cookie_token, token, api_key, propel_auth_user
            )
        except HTTPException as e:
            if not self.auto_error and e.status_code == 401:
                return None
            raise e

        if user.banned_at is not None:
            await track_async(user.email, "Banned user denied access")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user


get_user_sqlalchemy = GetUserSQLAlchemy(allow_basic=True)
