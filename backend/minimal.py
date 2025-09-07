from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, List, Optional
import random
import json

# Initialize FastAPI app
app = FastAPI(title="AI Mental Health Companion API - Minimal Version", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mock Data ---
users_db = {
    "user1": {"password": "pass123", "id": 1}
}

mood_entries = []

# --- Models ---
class UserLogin(BaseModel):
    username: str
    password: str

class MoodEntry(BaseModel):
    text_input: str
    mood_score: float
    journal_text: Optional[str] = None

class Intervention(BaseModel):
    emotion: str
    mood_score: float
    user_preferences: Optional[Dict] = None

# --- Helper Functions ---
def analyze_emotion(text: str) -> Dict[str, float]:
    """Simple emotion analysis - mock for demo"""
    emotions = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
    # Generate random probabilities that sum to 1
    probs = [random.random() for _ in range(len(emotions))]
    total = sum(probs)
    normalized = [p / total for p in probs]
    return {e: float(p) for e, p in zip(emotions, normalized)}

# --- API Routes ---
@app.post("/api/auth/register")
async def register(user: UserLogin):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    users_db[user.username] = {"password": user.password, "id": len(users_db) + 1}
    return {"message": "User registered successfully"}

@app.post("/api/auth/login")
async def login(user: UserLogin):
    if user.username not in users_db or users_db[user.username]["password"] != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user_id = users_db[user.username]["id"]
    return {"access_token": f"mock_token_{user_id}", "token_type": "bearer", "user_id": user_id}

@app.post("/api/mood/entry")
async def create_mood_entry(entry: MoodEntry):
    emotions = analyze_emotion(entry.text_input)
    dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
    
    new_entry = {
        "id": len(mood_entries) + 1,
        "mood_score": entry.mood_score,
        "text_emotion": dominant_emotion,
        "emotions": emotions,
        "timestamp": datetime.now().isoformat(),
        "crisis_assessment": {
            "risk_level": "LOW" if entry.mood_score > 4 else "MEDIUM"
        }
    }
    mood_entries.append(new_entry)
    
    return new_entry

@app.get("/api/mood/history")
async def get_mood_history():
    return {"history": mood_entries}

@app.post("/api/mood/predict")
async def predict_mood():
    if len(mood_entries) < 5:
        return {"message": "Need at least 5 mood entries for predictions. Keep logging your mood!"}
    
    return {
        "prediction": {
            "predicted_mood": round(random.uniform(3.5, 8.5), 1),
            "confidence": round(random.uniform(0.6, 0.9), 2)
        },
        "recommendations": "Focus on self-care and mindfulness exercises for the next few days."
    }

@app.post("/api/intervention/advanced")
async def get_advanced_intervention(intervention: Intervention):
    return [
        "Take a deep breath and count to 10 slowly.",
        "Try a 5-minute mindfulness meditation.",
        "Write down 3 things you're grateful for today."
    ]

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "minimal"
    }

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("minimal:app", host="0.0.0.0", port=8000, reload=True)
