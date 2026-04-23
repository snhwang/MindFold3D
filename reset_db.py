
import sqlite3
import os
import json

# Check if database exists and delete it
if os.path.exists("mindfold.db"):
    try:
        os.remove("mindfold.db")
        print("Existing database deleted")
    except Exception as e:
        print(f"Error deleting database: {e}")

# Create a new database with proper schema
conn = sqlite3.connect("mindfold.db")
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE,
    username TEXT UNIQUE,
    hashed_password TEXT,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE password_resets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    reset_code TEXT,
    is_used BOOLEAN DEFAULT 0,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE game_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    total_questions INTEGER DEFAULT 0,
    correct_answers INTEGER DEFAULT 0,
    feature_stats TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
''')

conn.commit()
conn.close()

print("Database reset complete. Empty database with schema created.")
