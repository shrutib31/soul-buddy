import os
from alembic.config import Config
from alembic import command

# Run Alembic migrations
alembic_cfg = Config(os.path.join(os.path.dirname(__file__), 'alembic.ini'))
command.upgrade(alembic_cfg, 'head')

# Add your seed logic here
# Example: Insert initial data using SQLAlchemy ORM
# from models import ...
# ...