"""Database initialization and migration management."""

from alembic.config import Config
from alembic import command
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
import os
import sys

from src.core.config import settings


def init_database():
    """Initialize database with migrations."""
    print("Initializing SSA database...")
    
    # Create alembic config
    alembic_cfg = Config("alembic.ini")
    
    # Override the sqlalchemy.url in the config
    alembic_cfg.set_main_option("sqlalchemy.url", settings.database.sqlalchemy_database_uri)
    
    try:
        # Upgrade to head (latest migration)
        command.upgrade(alembic_cfg, "head")
        print("✓ Database initialized successfully")
        return True
    except OperationalError as e:
        print(f"✗ Database connection failed: {e}")
        print("Make sure PostgreSQL is running and credentials are correct in .env")
        return False
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        return False


def create_migration(message):
    """Create a new migration."""
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", settings.database.sqlalchemy_database_uri)
    
    try:
        command.revision(alembic_cfg, message=message, autogenerate=True)
        print(f"✓ Migration created for: {message}")
    except Exception as e:
        print(f"✗ Failed to create migration: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "init":
            init_database()
        elif sys.argv[1] == "migrate" and len(sys.argv) > 2:
            create_migration(sys.argv[2])
        else:
            print("Usage: python db_init.py init | migrate <message>")
    else:
        init_database()