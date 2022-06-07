"""'create_user_table'

Revision ID: eeebf79dd115
Revises: 
Create Date: 2022-06-06 20:08:22.019659

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'eeebf79dd115'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
    'users',
    sa.Column('id', sa.Integer, primary_key=True),
    sa.Column('username', sa.String(30), nullable=False),
    sa.Column('email', sa.String(100), nullable = False),
    )
    pass


def downgrade() -> None:
    op.drop_table('users'),
    pass
