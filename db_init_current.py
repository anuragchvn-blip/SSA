#!/usr/bin/env python3
"""Initialize the database with current schema."""

import os
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data.database import Base
from src.core.config import settings

def init_db():
    """Initialize database with current schema."""
    # Use SQLite for simplicity in testing
    engine = create_engine(settings.database.sqlalchemy_database_uri, echo=True)
    
    # Create all tables according to the current model
    print("Creating tables...")
    Base.metadata.create_all(engine)
    print("Tables created successfully!")
    
    return engine

if __name__ == "__main__":
    init_db()
    print("Database initialized with current schema.")