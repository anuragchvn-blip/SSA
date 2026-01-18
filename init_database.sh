#!/bin/bash
# Database initialization script for Railway deployment

echo "Waiting for database to be ready..."
# Give the database connection time to establish
sleep 10

echo "Creating database tables..."
python create_tables.py

echo "Populating TLE data..."
python populate_tles.py

echo "Creating conjunction events..."
python create_conjunctions.py

echo "Starting application server..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000