#!/usr/bin/env python
"""
Minimal starter for the AI Mental Health Companion backend
This version uses a simple FastAPI app with mock data for demonstration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import random
from typing import Dict, List, Optional

# Initialize FastAPI app
app = FastAPI(title="AI Mental Health Companion API - Demo Version", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock database
users = {
    "demo": {"password": "demo123", "id": 1}
}

mood_entries = []
active_sessions = {}

# Models
class User(BaseModel):
    username: str
    password: str

class MoodEntry(BaseModel):
    text_input: Optional[str] = ""
    mood_score: float
    journal_text: Optional[str] = ""

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to AI Mental Health Companion API - Demo Version"}

@app.post("/api/auth/register")
async def register(user: User):
    if user.username in users:
        raise HTTPException(status_code=400, detail="Username already exists")
    users[user.username] = {"password": user.password, "id": len(users) + 1}
    return {"message": "User registered successfully"}

@app.post("/api/auth/login")
async def login(user: User):
    if user.username not in users or users[user.username]["password"] != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = f"demo_token_{random.randint(1000, 9999)}"
    active_sessions[token] = users[user.username]["id"]
    return {"access_token": token, "user_id": users[user.username]["id"]}

@app.post("/api/mood/entry")
async def create_mood_entry(entry: MoodEntry):
    # Simple mock emotion analysis
    emotions = {
        "joy": round(random.uniform(0.1, 0.9), 2),
        "sadness": round(random.uniform(0.1, 0.5), 2),
        "anger": round(random.uniform(0.1, 0.3), 2),
        "neutral": round(random.uniform(0.1, 0.7), 2)
    }
    
    # Normalize to sum to 1
    total = sum(emotions.values())
    emotions = {k: v/total for k, v in emotions.items()}
    
    # Find dominant emotion
    dominant = max(emotions.items(), key=lambda x: x[1])[0]
    
    # Create entry
    new_entry = {
        "id": len(mood_entries) + 1,
        "timestamp": datetime.now().isoformat(),
        "text_emotion": dominant,
        "emotions": emotions,
        "mood_score": entry.mood_score,
        "risk_level": "LOW" if entry.mood_score > 4 else "MEDIUM"
    }
    
    mood_entries.append(new_entry)
    return new_entry

@app.get("/api/mood/history")
async def get_mood_history():
    return {"history": mood_entries}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
