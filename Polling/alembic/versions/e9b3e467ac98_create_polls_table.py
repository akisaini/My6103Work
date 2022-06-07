"""'create_polls_table'

Revision ID: e9b3e467ac98
Revises: eeebf79dd115
Create Date: 2022-06-07 16:37:41.829772

"""
import enum
from alembic import op
from prometheus_client import Enum
import sqlalchemy as sa


class PollType(enum.Enum):
    text = 1
    image = 2


# revision identifiers, used by Alembic.
revision = 'e9b3e467ac98'
down_revision = 'eeebf79dd115'
branch_labels = None
depends_on = None 


def upgrade() -> None:
     op.create_table(
    'polls',
    sa.Column('id', sa.Integer, primary_key=True),
    sa.Column('title', sa.String(255), nullable=False),
    sa.Column('type', sa.Enum(PollType), nullable = False),
    sa.Column('is_voting_active', sa.Boolean, nullable=False),
    sa.Column('is_add_choices_active', sa.Boolean, nullable = False),
    sa.Column('created_by', sa.Integer, nullable=False),
    sa.Column('created_at', sa.DateTime, nullable = False),
    sa.Column('updated_at', sa.DateTime, nullable=False),    
    )
     pass


def downgrade() -> None:
    op.drop_table('polls')
    pass
