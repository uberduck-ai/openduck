"""add session_id column

Revision ID: c4e71a1a96da
Revises: 42f7dfcde186
Create Date: 2024-02-23 15:35:32.585825+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c4e71a1a96da"
down_revision: Union[str, None] = "42f7dfcde186"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index(
        op.f("ix_chat_history_session_id"), "chat_history", ["session_id"], unique=False
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f("ix_chat_history_session_id"), table_name="chat_history")
    # ### end Alembic commands ###
