import os
from urllib.parse import quote_plus
import databases
from sqlalchemy import select, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import configure_mappers, declarative_base, sessionmaker
from sqlalchemy_searchable import make_searchable


DB_HOST = os.environ.get(
    "POSTGRESQL_HOST", "uberduck-prod.ck70xzutqodt.us-west-2.rds.amazonaws.com"
)
DB_USER = "uberduck"
DB_PASSWORD = os.environ["POSTGRESQL_PASSWORD"]
DB_PORT = os.environ.get("POSTGRESQL_PORT", 5432)

connection_string = (
    f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}"
)
async_connection_string = (
    f"postgresql+asyncpg://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}"
)
database = databases.Database(
    f"postgresql://{DB_HOST}",
    user=DB_USER,
    password=DB_PASSWORD,
    port=DB_PORT,
)


class UberBase:
    @classmethod
    def get_filters(cls, *args):
        filter_tuples = args or []
        filters = []

        for obj, attr, values in filter_tuples:
            if values == [None]:
                filters.append(getattr(obj, attr).is_(None))
            else:
                filters.append(getattr(obj, attr).in_(values))

        return filters

    @classmethod
    def _base_query(cls, *args):
        filters = args or []
        query = select(cls).where(*filters)
        return query

    @classmethod
    def get(cls, **kwargs):
        filter_tuples = [(cls, k, [v]) for k, v in kwargs.items()]
        filters = cls.get_filters(*filter_tuples)
        query = cls._base_query(*filters)
        return query


Base = declarative_base(cls=UberBase)
make_searchable(Base.metadata)
configure_mappers()

echo = True
engine = create_engine(connection_string, echo=echo, pool_size=100, max_overflow=200)
Session = sessionmaker(bind=engine)
kwargs = {
    "pool_size": 100,
    "max_overflow": 200,
}
async_engine = create_async_engine(async_connection_string, echo=echo, **kwargs)
SessionAsync = sessionmaker(
    bind=async_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_db_async():
    async with SessionAsync() as session:
        yield session
        await session.commit()


def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()
