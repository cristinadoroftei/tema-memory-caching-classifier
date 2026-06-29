"""create sessions and conversation_messages tables

Revision ID: 004
Revises: 003
Create Date: 2026-06-28
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'sessions',
        sa.Column('id', sa.String(100), primary_key=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('metadata_json', sa.Text(), server_default='{}'),
    )

    op.create_table(
        'conversation_messages',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('session_id', sa.String(100), sa.ForeignKey('sessions.id'), nullable=False, index=True),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table('conversation_messages')
    op.drop_table('sessions')
