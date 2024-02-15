import asyncio
from datetime import datetime
from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import configure_mappers, scoped_session, sessionmaker
from propelauth_fastapi import User

from openduck_py.db import connection_string, Base
from openduck_py.models import DBUser, DBTemplatePrompt, DBTemplateDeployment
from openduck_py.routers.templates import DEFAULT_MODEL
from openduck_py.routers.main import app
from openduck_py.auth.auth import propel_auth

engine = create_engine(connection_string)
Session = scoped_session(sessionmaker(bind=engine))


@pytest.fixture(scope="session")
def event_loop(request) -> Generator:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def setup_database(db_connection):
    Base.metadata.bind = db_connection
    Base.metadata.create_all()
    yield
    Base.metadata.drop_all()


@pytest.fixture
def db_session():
    """NOTE(zach): This fixture handles connecting to the DB and resetting between tests.

    The way it is currently implemented is a very inefficient approach! It tears
    down the entire database and recreates it on every test. A much better
    approach is the one in db_session_v2, which creates the test database only
    once per *session*, then runs each test in a transaction which is rolled
    back after the test finishes. However, I haven't been able to make this
    approach work with the asyncio databases module that we use. When we
    eventually move to asyncio SQLAlchemy, I think this issue should go away.
    """
    configure_mappers()
    Base.metadata.create_all(engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def override_dependencies():
    def _dep():
        return User("test-propel-auth-id", None, "test@test.com")

    app.dependency_overrides[propel_auth.optional_user] = _dep


@pytest.fixture
def user_1(db_session):
    """Create a user."""
    user = DBUser(
        username="test-user",
        propel_auth_id="test-propel-auth-id",
        email="test@test.com",
        stripe_customer_id="test_stripe_customer_id",
    )
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def user_token(user_1):
    return "asdf"


@pytest.fixture
def user_2(db_session):
    """Create a user."""
    user = DBUser(
        username="test-user-2",
        propel_auth_id="test-propel-auth-id2",
        email="test2@test2.com",
    )
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def user_2_token(user_2):
    return "asdf2"


@pytest.fixture
def prompt_template(db_session, user_1):
    template = DBTemplatePrompt(
        user_id=user_1.id,
        uuid="uuid1",
        url_name="test-template",
        display_name="Test Template",
        prompt={
            "messages": [
                {
                    "role": "user",
                    "content": "This is a test prompt. {{test_var1}} {{test_var2}} {{test_var3}}",
                }
            ]
        },
        meta_json={
            "variables": ["test_var1", "test_var2", "test_var3"],
            "values": [{"test_var1": "", "test_var2": "", "test_var3": ""}],
        },
        model=DEFAULT_MODEL,
        deleted_at=None,
    )
    db_session.add(template)
    db_session.commit()
    return template


@pytest.fixture
def prompt_template2(db_session, user_1):
    template = DBTemplatePrompt(
        user_id=user_1.id,
        uuid="uuid1",
        url_name="test-template-2",
        display_name="Test Template 2",
        prompt={
            "messages": [
                {
                    "role": "user",
                    "content": "This is the second test prompt. {{test_var1}}",
                }
            ]
        },
        meta_json={
            "variables": ["test_var1"],
            "values": [{"test_var1": "hello"}],
        },
        model=DEFAULT_MODEL,
        deleted_at=None,
    )
    db_session.add(template)
    db_session.commit()
    return template


@pytest.fixture
def deployment_template(db_session, user_1):
    template = DBTemplateDeployment(
        user_id=user_1.id,
        uuid="uuid1",
        url_name="test-template",
        display_name="Test Template",
        prompt={
            "messages": [
                {
                    "role": "user",
                    "content": "This is a test prompt. {{test_var1}} {{test_var2}} {{test_var3}}",
                }
            ]
        },
        meta_json={
            "variables": ["test_var1", "test_var2", "test_var3"],
            "values": [{"test_var1": "", "test_var2": "", "test_var3": ""}],
        },
        model="gpt-4-deployment",
        deleted_at=None,
    )
    db_session.add(template)
    db_session.commit()
    return template


@pytest.fixture
def prompt_template_deleted(db_session, user_1):
    template = DBTemplatePrompt(
        user_id=user_1.id,
        uuid="uuid2",
        url_name="test-template-deleted",
        display_name="Test Template Deleted",
        prompt={
            "messages": [
                {
                    "role": "user",
                    "content": "This is a test prompt. {{test_var1}} {{test_var2}} {{test_var3}}",
                }
            ]
        },
        meta_json={
            "variables": ["test_var1", "test_var2", "test_var3"],
            "values": [{"test_var1": "", "test_var2": "", "test_var3": ""}],
        },
        model=DEFAULT_MODEL,
        deleted_at=datetime.utcnow(),
    )
    db_session.add(template)
    db_session.commit()
    return template


@pytest.fixture
def deployment_template_deleted(db_session, user_1):
    template = DBTemplateDeployment(
        user_id=user_1.id,
        uuid="uuid2",
        url_name="test-template-deleted",
        display_name="Test Template Deleted",
        prompt={
            "messages": [
                {
                    "role": "user",
                    "content": "This is a test prompt. {{test_var1}} {{test_var2}} {{test_var3}}",
                }
            ]
        },
        meta_json={
            "variables": ["test_var1", "test_var2", "test_var3"],
            "values": [{"test_var1": "", "test_var2": "", "test_var3": ""}],
        },
        model=DEFAULT_MODEL,
        deleted_at=datetime.utcnow(),
    )
    db_session.add(template)
    db_session.commit()
    return template
