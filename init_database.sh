#!/bin/bash
# Database initialization script for Railway deployment

echo "Initializing database schema..."
python -c "from src.data.database import init_db; init_db()" || echo "Schema already exists"

echo "Populating TLE data..."
python populate_tles.py

echo "Creating conjunction events..."
python create_conjunctions.py

echo "Starting application server..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000