from sqlalchemy import select, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import configure_mappers, declarative_base, sessionmaker


DB_USER = "uberduck"

connection_string = "sqlite:///test.db"
async_connection_string = "sqlite+aiosqlite:///test.db"


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

    @property
    def uuid(self):
        return f"{self.PREFIX}_{self._uuid}"


Base = declarative_base(cls=UberBase)
configure_mappers()

echo = True
engine = create_engine(connection_string, echo=echo)
Session = sessionmaker(bind=engine)
async_engine = create_async_engine(async_connection_string, echo=echo)
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
