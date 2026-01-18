#!/usr/bin/env python3
"""
Database initialization script for Railway deployment.
Creates all necessary tables in the PostgreSQL database.
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the src directory to the path so we can import the models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.data.models import Base
from src.core.config import settings


def create_database_tables():
    """Create all database tables."""
    try:
        # Get database URL from settings
        database_url = settings.database.sqlalchemy_database_uri
        print(f"Connecting to database: {database_url.replace(settings.database.postgres_password, '***')}")
        
        # Create engine
        engine = create_engine(database_url)
        
        # Create all tables
        print("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("Tables created successfully!")
        
        # Verify tables were created
        with engine.connect() as conn:
            # Check for TLE table
            result = conn.execute(text("""
                SELECT EXISTS (
                   SELECT FROM information_schema.tables 
                   WHERE table_schema = 'public' 
                   AND table_name = 'tle'
                );
            """))
            tle_exists = result.scalar()
            
            # Check for ConjunctionEvent table
            result = conn.execute(text("""
                SELECT EXISTS (
                   SELECT FROM information_schema.tables 
                   WHERE table_schema = 'public' 
                   AND table_name = 'conjunction_event'
                );
            """))
            conj_exists = result.scalar()
            
            print(f"TLE table exists: {tle_exists}")
            print(f"ConjunctionEvent table exists: {conj_exists}")
            
            if tle_exists and conj_exists:
                print("✅ Database tables created successfully!")
                return True
            else:
                print("❌ Some tables were not created properly")
                return False
                
    except Exception as e:
        print(f"❌ Error creating database tables: {str(e)}")
        return False


if __name__ == "__main__":
    success = create_database_tables()
    if success:
        print("\nDatabase initialization completed successfully!")
        sys.exit(0)
    else:
        print("\nDatabase initialization failed!")
        sys.exit(1)