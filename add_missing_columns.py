#!/usr/bin/env python3
"""Script to add missing columns to existing database."""

import sqlite3
from sqlalchemy import create_engine, inspect

def add_missing_columns():
    """Add missing velocity columns to conjunction_event table."""
    # Connect to the database
    engine = create_engine("sqlite:///./ssa_demo.db")
    
    # Check existing columns in conjunction_event table
    inspector = inspect(engine)
    columns = inspector.get_columns('conjunction_event')
    existing_column_names = [col['name'] for col in columns]
    
    print(f"Existing columns in conjunction_event: {existing_column_names}")
    
    # Define the missing columns we need to add
    required_columns = [
        'primary_vx_eci',
        'primary_vy_eci', 
        'primary_vz_eci',
        'secondary_vx_eci',
        'secondary_vy_eci',
        'secondary_vz_eci'
    ]
    
    # Connect with sqlite3 to add columns
    conn = sqlite3.connect('./ssa_demo.db')
    cursor = conn.cursor()
    
    missing_columns = [col for col in required_columns if col not in existing_column_names]
    
    print(f"Missing columns to add: {missing_columns}")
    
    for col in missing_columns:
        try:
            # Add the missing column (FLOAT type to match the model)
            cursor.execute(f"ALTER TABLE conjunction_event ADD COLUMN {col} FLOAT;")
            print(f"Added column: {col}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"Column {col} already exists")
            else:
                print(f"Error adding column {col}: {e}")
    
    conn.commit()
    conn.close()
    
    print("Completed adding missing columns.")

if __name__ == "__main__":
    add_missing_columns()