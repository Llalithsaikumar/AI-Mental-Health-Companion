"""
Script to create SQLite database for AI Mental Health Companion
Use this as a fallback when MongoDB is not available
"""

import os
import sys
import sqlite3
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Ensure data directory exists
data_dir = Path("../data")
data_dir.mkdir(exist_ok=True)

# Database path
DB_PATH = data_dir / "mental_health.db"

def create_sqlite_database():
    """Create SQLite database with necessary tables"""
    print(f"Creating SQLite database at {DB_PATH}...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE,
        full_name TEXT,
        hashed_password TEXT NOT NULL,
        is_active INTEGER DEFAULT 1,
        created_at TEXT NOT NULL
    )
    ''')
    
    # Create mood_entries table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mood_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        mood_score REAL NOT NULL,
        text_content TEXT,
        text_emotion TEXT,
        audio_emotion TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create interventions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interventions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        type TEXT NOT NULL,
        message TEXT NOT NULL,
        suggestions TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create intervention_feedback table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS intervention_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        intervention_id INTEGER NOT NULL,
        user_id TEXT NOT NULL,
        rating INTEGER NOT NULL,
        comments TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (intervention_id) REFERENCES interventions (id),
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create indexes for common queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mood_entries_user_id_timestamp ON mood_entries (user_id, timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_interventions_user_id ON interventions (user_id)')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("SQLite database created successfully!")
    
    # Print database file size
    db_size = DB_PATH.stat().st_size / 1024  # Size in KB
    print(f"Database size: {db_size:.2f} KB")

if __name__ == "__main__":
    create_sqlite_database()
    
    # Add sample data if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--sample-data":
        try:
            from datetime import datetime, timedelta
            import uuid
            import random
            import bcrypt
            
            print("Adding sample data...")
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Add a test user
            user_id = str(uuid.uuid4())
            username = "testuser"
            password = "password123"
            hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            
            try:
                cursor.execute(
                    "INSERT INTO users (id, username, email, full_name, hashed_password, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, username, "test@example.com", "Test User", hashed_password, datetime.now().isoformat())
                )
            except sqlite3.IntegrityError:
                print("Test user already exists, skipping...")
                
                # Get existing user_id
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                result = cursor.fetchone()
                if result:
                    user_id = result[0]
            
            # Add sample mood entries
            emotions = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
            
            for i in range(30):
                # Generate mood entries for the last 30 days
                entry_date = datetime.now() - timedelta(days=i)
                mood_score = random.uniform(3.0, 9.0)
                emotion = random.choice(emotions)
                
                cursor.execute(
                    "INSERT INTO mood_entries (user_id, mood_score, text_emotion, timestamp) VALUES (?, ?, ?, ?)",
                    (user_id, mood_score, emotion, entry_date.isoformat())
                )
            
            # Add sample interventions
            intervention_types = ["mindfulness", "cognitive", "behavioral", "social"]
            
            for i in range(5):
                # Generate interventions for the last 5 days
                intervention_date = datetime.now() - timedelta(days=i*3)
                intervention_type = random.choice(intervention_types)
                
                cursor.execute(
                    "INSERT INTO interventions (user_id, type, message, suggestions, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (
                        user_id, 
                        intervention_type, 
                        f"Sample {intervention_type} intervention", 
                        json.dumps(["Suggestion 1", "Suggestion 2", "Suggestion 3"]), 
                        intervention_date.isoformat()
                    )
                )
            
            conn.commit()
            conn.close()
            
            print("Sample data added successfully!")
            
        except ImportError:
            print("Could not add sample data. Make sure bcrypt is installed: pip install bcrypt")
        except Exception as e:
            print(f"Error adding sample data: {e}")
