#!/bin/bash
# Database initialization script for Railway deployment

echo "Waiting for database to be ready..."
# Give the database connection time to establish
sleep 15

echo "Testing database connection..."
python -c '
import sys
try:
    from src.core.config import settings
    db_url = settings.database.sqlalchemy_database_uri
    if settings.database.postgres_password and settings.database.postgres_password != "change_me_in_production":
        masked_url = db_url.replace(settings.database.postgres_password, "***")
    else:
        masked_url = db_url
    print(f"Database URL: {masked_url}")
    from sqlalchemy import create_engine, text
    engine = create_engine(settings.database.sqlalchemy_database_uri)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("Database connection successful!")
except Exception as e:
    print(f"Database connection failed: {e}")
    sys.exit(1)
'

echo "Creating database tables..."
python create_tables.py

echo "Populating TLE data..."
python railway_init.py

echo "Creating conjunction events..."
python create_conjunctions.py

echo "Starting application server..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000