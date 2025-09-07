"""
Minimal app for AI Mental Health Companion
With MongoDB and SQLite support
"""

# Standard library imports
import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third-party imports
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Try to import security packages
try:
    from passlib.context import CryptContext
    from jose import JWTError, jwt
    SECURITY_ENABLED = True
except ImportError:
    print("Warning: security packages not found. Install with: pip install python-jose passlib[bcrypt]")
    SECURITY_ENABLED = False

# Try to import the MongoDB handler
try:
    from mongodb_handler import db_handler
    print("MongoDB handler found and imported.")
    USE_MONGODB = True
except ImportError:
    print("MongoDB handler not available. Using SQLite as fallback.")
    import sqlite3
    USE_MONGODB = False

# Ensure data directory exists
data_dir = Path("../data")
data_dir.mkdir(exist_ok=True)

# Security configurations
SECRET_KEY = os.environ.get("SECRET_KEY", "this_is_a_secret_key_please_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password context for hashing
if SECURITY_ENABLED:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    # OAuth2 password bearer token
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")
else:
    # Fallback for when security packages are not available
    oauth2_scheme = None

# Pydantic models for auth
class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserBase(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: str
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class MoodEntry(BaseModel):
    mood_score: int
    text_content: Optional[str] = None
    text_emotion: Optional[str] = None
    audio_emotion: Optional[str] = None

# Mock user database - replace with MongoDB or SQLite later
MOCK_USERS_DB = {
    "testuser": {
        "id": "user1",
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "hashed_password": "hashed_password_placeholder",  # Will be updated after initialization
        "disabled": False
    }
}

MOCK_MOOD_HISTORY = {
    "user1": [
        {"timestamp": datetime.now().isoformat(), "mood_score": 7, "text_content": "I'm feeling good today.", "text_emotion": "joy", "audio_emotion": None},
        {"timestamp": (datetime.now() - timedelta(days=1)).isoformat(), "mood_score": 5, "text_content": "Just an average day.", "text_emotion": "neutral", "audio_emotion": None},
        {"timestamp": (datetime.now() - timedelta(days=2)).isoformat(), "mood_score": 3, "text_content": "Feeling a bit down today.", "text_emotion": "sadness", "audio_emotion": None}
    ]
}

# Helper functions for auth
if SECURITY_ENABLED:
    def verify_password(plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(password):
        return pwd_context.hash(password)
else:
    # Simple mock implementations for when security packages are not available
    def verify_password(plain_password, hashed_password):
        return plain_password == hashed_password

    def get_password_hash(password):
        return f"mock_hash_{password}"  # Not secure, just for testing

# User functions based on database type
def get_user(username: str):
    if USE_MONGODB:
        user_data = db_handler.get_user_by_username(username)
        if user_data:
            user_dict = dict(user_data)
            return UserInDB(**user_dict)
        return None
    else:
        # SQLite or mock implementation
        if username in MOCK_USERS_DB:
            user_dict = MOCK_USERS_DB[username]
            return UserInDB(**user_dict)
        return None

def get_user_by_id(user_id: str):
    if USE_MONGODB:
        user_data = db_handler.get_user_by_id(user_id)
        if user_data:
            user_dict = dict(user_data)
            return UserInDB(**user_dict)
        return None
    else:
        # SQLite or mock implementation
        for username, user_data in MOCK_USERS_DB.items():
            if user_data["id"] == user_id:
                return UserInDB(**user_data)
        return None

# JWT token functions
if SECURITY_ENABLED:
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    async def get_current_user(token: str = Depends(oauth2_scheme)):
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            token_data = TokenData(username=username)
        except Exception:  # Catch both JWTError and other exceptions
            raise credentials_exception
        user = get_user(username=token_data.username)
        if user is None:
            raise credentials_exception
        return user
else:
    # Simple mock implementations
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        return f"mock_token_{data.get('sub', 'unknown')}"

    async def get_current_user(token: str = None):
        if token and token.startswith("mock_token_"):
            username = token.split("_")[-1]
            return get_user(username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Try to import advanced interventions
try:
    from advanced_interventions import AdaptiveLearningSystem, AdvancedInterventionEngine
except ImportError:
    print("Warning: advanced_interventions module not found, using mock implementation")
    class AdaptiveLearningSystem:
        def track_intervention_effectiveness(self, user_id, intervention_id, feedback):
            pass
        
        def analyze_user_patterns(self, user_id):
            return {"insights": [], "recommendations": []}
    
    class AdvancedInterventionEngine:
        def get_personalized_intervention(self, user_id, current_mood, history=None, **kwargs):
            interventions = [
                {"type": "meditation", "title": "5-Minute Breathing", "content": "Focus on your breath..."},
                {"type": "cognitive", "title": "Thought Challenge", "content": "Identify negative thoughts..."},
                {"type": "behavioral", "title": "Pleasant Activity", "content": "Do something you enjoy..."}
            ]
            return interventions[0]

# Try to import predictive models
try:
    from predictive_models import MoodPredictor, InterventionRecommender
except ImportError:
    print("Warning: predictive_models module not found, using mock implementation")
    class MoodPredictor:
        def __init__(self):
            self.is_trained = False
        
        def train_models(self, data):
            self.is_trained = True
            
        def predict_mood(self, features):
            return {"predicted_mood": 5, "confidence": 0.7}
        
        def detect_anomaly(self, features):
            return {"is_anomaly": False}
    
    class InterventionRecommender:
        def get_recommendation(self, user_id, mood_history):
            return ["Take a walk", "Practice deep breathing", "Listen to calming music"]

# Mock crisis assessment system
class CrisisRiskAssessment:
    def assess_risk_level(self, user_id, mood_history, text_content):
        # Very basic assessment logic - would be more sophisticated in a real app
        recent_moods = [entry.get('mood_score', 5) for entry in mood_history[:3]]
        avg_mood = sum(recent_moods) / len(recent_moods) if recent_moods else 5
        
        risk_keywords = [
            "suicide", "kill myself", "end it all", "no reason to live", 
            "better off dead", "can't go on", "want to die"
        ]
        
        if text_content and any(keyword in text_content.lower() for keyword in risk_keywords):
            return {"risk_level": "HIGH", "recommendations": ["Contact crisis support"]}
        elif avg_mood < 3:
            return {"risk_level": "MEDIUM", "recommendations": ["Consider speaking with a therapist"]}
        else:
            return {"risk_level": "LOW", "recommendations": ["Practice self-care"]}

# Mock realtime features
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket, user_id: str):
        self.active_connections[user_id].remove(websocket)

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                await connection.send_text(message)

class RealTimeMonitoringSystem:
    async def start_monitoring(self, user_id):
        pass
        
    async def process_mood_update(self, user_id, data):
        return [{"type": "notification", "content": "Your mood has been tracked successfully."}]

class SmartNotificationSystem:
    pass

# Create FastAPI app instance
app = FastAPI(
    title="Mental Health Companion API", 
    version="1.0.0",
    description="AI-powered mental health companion API"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
try:
    app.mount("/static", StaticFiles(directory="./static"), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")

# Mock active tokens for authentication
active_tokens = {}

# Initialize systems
adaptive_learning = AdaptiveLearningSystem()
advanced_intervention_engine = AdvancedInterventionEngine()
mood_predictor = MoodPredictor()
intervention_recommender = InterventionRecommender()
crisis_assessor = CrisisRiskAssessment()
connection_manager = ConnectionManager()
realtime_monitor = RealTimeMonitoringSystem()
smart_notifications = SmartNotificationSystem()

# Update mock user password with proper hash
if SECURITY_ENABLED:
    MOCK_USERS_DB["testuser"]["hashed_password"] = pwd_context.hash("password123")
else:
    MOCK_USERS_DB["testuser"]["hashed_password"] = "password123"

# Basic routes
@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    db_info = "MongoDB" if USE_MONGODB else "SQLite"
    security_info = "Enabled" if SECURITY_ENABLED else "Disabled"
    return {
        "message": f"Welcome to the Mental Health Companion API",
        "database": db_info,
        "security": security_info,
        "status": "operational",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/logo.png")
async def get_logo():
    """Serve the logo image"""
    return FileResponse("../static/logo.png")

# Authentication routes
@app.post("/api/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Get access token for authentication"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Store token in active tokens
    user_id = getattr(user, "id", "user1")
    active_tokens[access_token] = {"user_id": user_id}
    
    return {"access_token": access_token, "token_type": "bearer", "user_id": user_id}

@app.post("/api/auth/register", response_model=User)
async def register_user(user: UserCreate):
    # Check if username already exists
    existing_user = get_user(user.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
        
    # Create new user
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user.password)
    
    user_dict = {
        "id": user_id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "disabled": False
    }
    
    if USE_MONGODB:
        db_handler.create_user(user_dict)
    else:
        # Add to mock database
        MOCK_USERS_DB[user.username] = user_dict
        
    return User(
        id=user_id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=False
    )

# Mood tracking endpoints
@app.post("/api/mood")
async def log_mood(mood_entry: MoodEntry, current_user: User = Depends(get_current_active_user)):
    timestamp = datetime.now().isoformat()
    
    # Map the emotion based on mood score if not provided
    if not mood_entry.text_emotion:
        if mood_entry.mood_score >= 8:
            mood_entry.text_emotion = "joy"
        elif mood_entry.mood_score >= 6:
            mood_entry.text_emotion = "neutral"
        elif mood_entry.mood_score >= 4:
            mood_entry.text_emotion = "neutral"
        elif mood_entry.mood_score >= 2:
            mood_entry.text_emotion = "sadness"
        else:
            mood_entry.text_emotion = "sadness"
    
    entry = {
        "timestamp": timestamp,
        "mood_score": mood_entry.mood_score,
        "text_content": mood_entry.text_content,
        "text_emotion": mood_entry.text_emotion,
        "audio_emotion": mood_entry.audio_emotion
    }
    
    if USE_MONGODB:
        # Add user_id to entry
        entry["user_id"] = current_user.id
        db_handler.create_mood_entry(entry)
    else:
        # Add to mock database
        if current_user.id not in MOCK_MOOD_HISTORY:
            MOCK_MOOD_HISTORY[current_user.id] = []
        
        # Insert at the beginning for latest entry
        MOCK_MOOD_HISTORY[current_user.id].insert(0, entry)
        
        # Keep only the latest 100 entries to avoid memory issues
        MOCK_MOOD_HISTORY[current_user.id] = MOCK_MOOD_HISTORY[current_user.id][:100]
    
    # Run crisis assessment if text content is provided
    risk_assessment = {"risk_level": "LOW"}
    if mood_entry.text_content:
        mood_history = get_mood_history(current_user.id)
        risk_assessment = crisis_assessor.assess_risk_level(
            current_user.id, 
            mood_history, 
            mood_entry.text_content
        )
    
    print(f"Mood entry logged for user {current_user.id}: {entry}")
    
    return {
        "status": "success",
        "timestamp": timestamp,
        "mood_score": mood_entry.mood_score,
        "text_emotion": mood_entry.text_emotion,
        "risk_assessment": risk_assessment
    }

# Helper function to get mood history
def get_mood_history(user_id, limit=30):
    if USE_MONGODB:
        return db_handler.get_mood_history(user_id, limit)
    else:
        # Return from mock database
        return MOCK_MOOD_HISTORY.get(user_id, [])

@app.get("/api/mood/history")
async def get_user_mood_history(current_user: User = Depends(get_current_active_user)):
    mood_history = get_mood_history(current_user.id)
    print(f"Retrieved {len(mood_history)} mood entries for user {current_user.id}")
    return {"history": mood_history}

@app.get("/api/mood/stats")
async def get_mood_statistics(days: int = 30, current_user: User = Depends(get_current_active_user)):
    if USE_MONGODB:
        return db_handler.get_mood_stats(current_user.id, days)
    
    # Simplified stats logic for mock data
    mood_history = get_mood_history(current_user.id)
    
    # Filter by days
    cutoff_date = datetime.now() - timedelta(days=days)
    filtered_history = []
    for entry in mood_history:
        entry_date = datetime.fromisoformat(entry["timestamp"])
        if entry_date >= cutoff_date:
            filtered_history.append(entry)
    
    if not filtered_history:
        return {
            "avg_mood": 0,
            "min_mood": 0,
            "max_mood": 0,
            "count": 0,
            "emotion_counts": {}
        }
    
    # Calculate statistics
    mood_scores = [entry["mood_score"] for entry in filtered_history]
    emotions = [entry["text_emotion"] for entry in filtered_history if entry["text_emotion"]]
    
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    return {
        "avg_mood": sum(mood_scores) / len(mood_scores),
        "min_mood": min(mood_scores),
        "max_mood": max(mood_scores),
        "count": len(filtered_history),
        "emotion_counts": emotion_counts
    }

# Prediction endpoint
@app.get("/api/predict/mood")
async def predict_user_mood(current_user: User = Depends(get_current_active_user)):
    # Get user's mood history
    mood_history = get_mood_history(current_user.id)
    
    if len(mood_history) < 3:
        return {"error": "Not enough mood data for prediction"}
    
    # Train predictor if not already trained
    if not mood_predictor.is_trained:
        mood_predictor.train_models(mood_history)
    
    # Generate prediction
    current_time = datetime.now()
    current_features = {
        'hour': current_time.hour,
        'day_of_week': current_time.weekday(),
        'is_weekend': 1 if current_time.weekday() >= 5 else 0,
        'mood_7d_avg': np.mean([entry['mood_score'] for entry in mood_history[:7]]),
        'mood_7d_std': np.std([entry['mood_score'] for entry in mood_history[:7]]),
        'mood_trend': mood_history[0]['mood_score'] - mood_history[1]['mood_score'] if len(mood_history) > 1 else 0,
        'late_night': 1 if current_time.hour >= 23 or current_time.hour <= 5 else 0,
        'emotion_joy': 1 if mood_history[0]['text_emotion'] == 'joy' else 0,
        'emotion_sadness': 1 if mood_history[0]['text_emotion'] == 'sadness' else 0,
        'emotion_anger': 1 if mood_history[0]['text_emotion'] == 'anger' else 0,
        'emotion_fear': 1 if mood_history[0]['text_emotion'] == 'fear' else 0,
    }
    
    prediction = mood_predictor.predict_mood(current_features)
    return prediction

# Intervention endpoints
@app.get("/api/interventions/recommended")
async def get_recommended_interventions(current_user: User = Depends(get_current_active_user)):
    mood_history = get_mood_history(current_user.id)
    recommendations = intervention_recommender.get_recommendation(current_user.id, mood_history)
    return {"recommendations": recommendations}

@app.get("/api/interventions/personalized")
async def get_personalized_intervention(
    current_user: User = Depends(get_current_active_user)
):
    mood_history = get_mood_history(current_user.id)
    current_mood = mood_history[0]["mood_score"] if mood_history else 5
    
    intervention = advanced_intervention_engine.get_personalized_intervention(
        user_id=current_user.id,
        current_mood=current_mood,
        history=mood_history
    )
    
    return intervention

# WebSocket for real-time monitoring
@app.websocket("/ws/monitor/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await connection_manager.connect(websocket, user_id)
    await realtime_monitor.start_monitoring(user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)
            
            # Process the update
            notifications = await realtime_monitor.process_mood_update(user_id, data_json)
            
            # Send back any notifications
            if notifications:
                await websocket.send_json({"notifications": notifications})
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, user_id)

# Database status endpoint
@app.get("/api/db/status")
async def db_status():
    if USE_MONGODB:
        try:
            stats = db_handler.get_collection_stats()
            return {
                "status": "connected",
                "type": "mongodb",
                "collections": stats
            }
        except Exception as e:
            return {
                "status": "error",
                "type": "mongodb",
                "error": str(e)
            }
    else:
        return {
            "status": "connected",
            "type": "sqlite",
            "collections": {
                "users": len(MOCK_USERS_DB),
                "mood_entries": sum(len(entries) for entries in MOCK_MOOD_HISTORY.values())
            }
        }

# Run the app directly if executed
if __name__ == "__main__":
    print("Starting Mental Health Companion API (Minimal Version)")
    print("Server running at http://localhost:8000")
    print("API documentation available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
