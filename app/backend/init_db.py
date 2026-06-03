#!/usr/bin/env python
"""
Database initialization script
Initialize SQLite database from schema.sql
"""

import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data")
DB_FILE = os.path.join(DB_PATH, "connect4.db")
SCHEMA_FILE = os.path.join(BASE_DIR, "schema.sql")

def init_db():
    """Initialize database from schema.sql"""
    # Create data directory if it doesn't exist
    os.makedirs(DB_PATH, exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Read and execute schema
    with open(SCHEMA_FILE, 'r') as f:
        schema = f.read()
    
    cursor.executescript(schema)
    conn.commit()
    conn.close()
    
    print(f"✓ Database initialized at: {DB_FILE}")

if __name__ == "__main__":
    init_db()
