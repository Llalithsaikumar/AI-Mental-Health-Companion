"""
MongoDB-enabled minimal app for AI Mental Health Companion
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from jose import JWTError, jwt
from passlib.context import CryptContext

# Ensure data directory exists
data_dir = Path("../data")
data_dir.mkdir(exist_ok=True)

# Local imports
try:
    from mongodb_handler import db_handler
    USE_MONGODB = True
    print("MongoDB handler loaded successfully")
except ImportError:
    print("MongoDB handler not available. Using SQLite as fallback.")
    import sqlite3
    USE_MONGODB = False

# Security configurations
SECRET_KEY = os.environ.get("SECRET_KEY", "this_is_a_secret_key_please_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 password bearer token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")

# Pydantic models
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
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        orm_mode = True

class UserInDB(User):
    hashed_password: str

class MoodEntry(BaseModel):
    mood_score: float
    text: Optional[str] = None
    text_emotion: Optional[str] = None
    audio_emotion: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Auth functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(username: str):
    if USE_MONGODB:
        user_data = db_handler.get_user_by_username(username)
        if user_data:
            return UserInDB(**user_data)
    else:
        # Mock user data (for testing)
        fake_users_db = {
            "testuser": {
                "id": "user123",
                "username": "testuser",
                "email": "test@example.com",
                "full_name": "Test User",
                "hashed_password": get_password_hash("password123"),
                "is_active": True,
                "created_at": datetime.now()
            }
        }
        if username in fake_users_db:
            return UserInDB(**fake_users_db[username])
    return None

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

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
    except JWTError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Create FastAPI app instance
app = FastAPI(
    title="Mental Health Companion API", 
    version="1.0.0",
    description="AI-powered mental health companion API with MongoDB integration"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import local modules with fallbacks
try:
    from advanced_interventions import AdvancedInterventionEngine, AdaptiveLearningSystem
except ImportError:
    print("Warning: advanced_interventions module not found, using mock implementation")
    class AdvancedInterventionEngine:
        def get_personalized_intervention(self, **kwargs):
            return {"message": "Mock intervention", "suggestions": ["Take a break", "Breathe deeply"]}
    class AdaptiveLearningSystem:
        def record_intervention_feedback(self, **kwargs):
            pass

try:
    from predictive_models import MentalHealthPredictor, CrisisRiskAssessment
except ImportError:
    print("Warning: predictive_models module not found, using mock implementation")
    class MentalHealthPredictor:
        is_trained = False
        def train_models(self, data):
            self.is_trained = True
        def predict_next_mood(self, features):
            return {"predicted_mood": 6.0, "confidence": 0.7}
        def detect_anomaly(self, features):
            return {"is_anomaly": False}
    class CrisisRiskAssessment:
        def assess_crisis_risk(self, text, score, history):
            return {"risk_level": "LOW", "recommendations": ["Practice self-care"]}

try:
    from realtime_features import ConnectionManager, RealTimeMonitoringSystem, SmartNotificationSystem
except ImportError:
    print("Warning: realtime_features module not found, using mock implementation")
    class ConnectionManager:
        async def connect(self, websocket, user_id):
            await websocket.accept()
        def disconnect(self, websocket, user_id):
            pass
        async def send_personal_message(self, message, user_id):
            pass
    class RealTimeMonitoringSystem:
        async def start_monitoring(self, user_id):
            pass
        async def process_mood_update(self, user_id, data):
            return []
    class SmartNotificationSystem:
        pass

# Initialize systems
advanced_intervention_engine = AdvancedInterventionEngine()
adaptive_learning = AdaptiveLearningSystem()
mental_health_predictor = MentalHealthPredictor()
crisis_assessor = CrisisRiskAssessment()
connection_manager = ConnectionManager()
realtime_monitor = RealTimeMonitoringSystem()
smart_notifications = SmartNotificationSystem()

# API Routes
@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    db_info = "MongoDB" if USE_MONGODB else "SQLite"
    return {
        "message": f"Welcome to the Mental Health Companion API ({db_info})",
        "status": "operational",
        "version": "1.0.0",
        "documentation": "/docs"
    }

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
    
    return {"access_token": access_token, "token_type": "bearer", "user_id": user.id}

@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate):
    """Register a new user"""
    # Check if username already exists
    existing_user = get_user(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    user_id = str(uuid.uuid4())
    
    user_dict = {
        "id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "hashed_password": hashed_password,
        "is_active": True,
        "created_at": datetime.now()
    }
    
    if USE_MONGODB:
        db_handler.create_user(user_dict)
    
    # In a non-MongoDB implementation, you would save to SQLite here
    
    return {"message": "User created successfully", "user_id": user_id}

@app.get("/api/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user profile"""
    return current_user

@app.post("/api/mood/log")
async def log_mood(mood_data: MoodEntry, current_user: User = Depends(get_current_active_user)):
    """Log a mood entry"""
    try:
        # Prepare mood entry data
        entry = {
            "user_id": current_user.id,
            "mood_score": mood_data.mood_score,
            "text_content": mood_data.text,
            "text_emotion": mood_data.text_emotion,
            "audio_emotion": mood_data.audio_emotion,
            "timestamp": mood_data.timestamp
        }
        
        # Save to database
        if USE_MONGODB:
            entry_id = db_handler.create_mood_entry(entry)
        else:
            # SQLite fallback
            conn = sqlite3.connect('../data/mental_health.db')
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS mood_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                mood_score REAL NOT NULL,
                text_content TEXT,
                text_emotion TEXT,
                audio_emotion TEXT,
                timestamp TEXT NOT NULL
            )
            ''')
            
            cursor.execute(
                """INSERT INTO mood_entries 
                (user_id, mood_score, text_content, text_emotion, audio_emotion, timestamp) 
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    current_user.id, 
                    mood_data.mood_score, 
                    mood_data.text, 
                    mood_data.text_emotion,
                    mood_data.audio_emotion,
                    mood_data.timestamp.isoformat()
                )
            )
            
            conn.commit()
            entry_id = cursor.lastrowid
            conn.close()
            
        return {
            "status": "success",
            "mood_id": entry_id,
            "analysis": {
                "text_emotion": mood_data.text_emotion,
                "confidence": 0.85
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving mood entry: {str(e)}"
        )

@app.get("/api/mood/history")
async def get_mood_history(current_user: User = Depends(get_current_active_user)):
    """Get mood history for the current user"""
    try:
        if USE_MONGODB:
            entries = db_handler.get_mood_history(current_user.id)
            
            # Format data for response
            history = [
                {
                    "mood_score": entry.get("mood_score"),
                    "emotion": entry.get("text_emotion"),
                    "timestamp": entry.get("timestamp")
                }
                for entry in entries
            ]
        else:
            # SQLite fallback
            conn = sqlite3.connect('../data/mental_health.db')
            cursor = conn.cursor()
            
            try:
                cursor.execute(
                    "SELECT mood_score, text_emotion, timestamp FROM mood_entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT 30",
                    (current_user.id,)
                )
                entries = cursor.fetchall()
            except sqlite3.OperationalError:
                entries = []
                
            conn.close()
            
            # Format data
            history = [
                {
                    "mood_score": entry[0],
                    "emotion": entry[1],
                    "timestamp": entry[2]
                }
                for entry in entries
            ]
        
        return {
            "user_id": current_user.id,
            "history": history
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving mood history: {str(e)}"
        )

@app.post("/api/mood/predict")
async def predict_mood(current_user: User = Depends(get_current_active_user)):
    """Predict future mood based on patterns"""
    
    # Get mood history
    try:
        if USE_MONGODB:
            entries = db_handler.get_mood_history(current_user.id, limit=50)
            
            mood_history = [
                {
                    'mood_score': entry.get("mood_score"),
                    'text_emotion': entry.get("text_emotion"),
                    'timestamp': entry.get("timestamp")
                }
                for entry in entries
            ]
        else:
            # SQLite fallback
            try:
                conn = sqlite3.connect('../data/mental_health.db')
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT mood_score, text_emotion, timestamp FROM mood_entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
                    (current_user.id,)
                )
                entries = cursor.fetchall()
                conn.close()
                
                mood_history = [
                    {
                        'mood_score': entry[0],
                        'text_emotion': entry[1],
                        'timestamp': entry[2]
                    }
                    for entry in entries
                ]
            except:
                mood_history = []
    except Exception as e:
        return {"message": f"Error retrieving mood history: {str(e)}"}
    
    if len(mood_history) < 5:
        return {"message": "Insufficient data for prediction"}
    
    # Train predictor if not already trained
    if not mental_health_predictor.is_trained:
        mental_health_predictor.train_models(mood_history)
    
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
        'emotion_neutral': 1 if mood_history[0]['text_emotion'] == 'neutral' else 0,
    }
    
    prediction = mental_health_predictor.predict_next_mood(current_features)
    anomaly_detection = mental_health_predictor.detect_anomaly(current_features)
    
    return {
        "prediction": prediction,
        "anomaly_detection": anomaly_detection,
        "recommendations": "Consider maintaining current positive patterns" if prediction['predicted_mood'] > 6 else "Focus on self-care and consider reaching out for support"
    }

@app.post("/api/crisis/assess")
async def assess_crisis_risk(request: Request, current_user: User = Depends(get_current_active_user)):
    """Comprehensive crisis risk assessment"""
    data = await request.json()
    
    # Get mood history
    try:
        if USE_MONGODB:
            entries = db_handler.get_mood_history(current_user.id, limit=30)
            
            mood_history = [
                {
                    'mood_score': entry.get("mood_score"),
                    'text_emotion': entry.get("text_emotion"),
                    'timestamp': entry.get("timestamp")
                }
                for entry in entries
            ]
        else:
            try:
                conn = sqlite3.connect('../data/mental_health.db')
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT mood_score, text_emotion, timestamp FROM mood_entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT 30",
                    (current_user.id,)
                )
                entries = cursor.fetchall()
                conn.close()
                
                mood_history = [
                    {
                        'mood_score': entry[0],
                        'text_emotion': entry[1],
                        'timestamp': entry[2]
                    }
                    for entry in entries
                ]
            except:
                mood_history = []
    except Exception as e:
        mood_history = []
    
    # Assess crisis risk
    assessment = crisis_assessor.assess_crisis_risk(
        data.get('text', ''),
        data.get('mood_score', 5),
        mood_history
    )
    
    return assessment

@app.get("/api/interventions")
async def get_interventions(
    request: Request,
    emotion: Optional[str] = None,
    mood_score: Optional[float] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get intervention suggestions based on mood"""
    
    if mood_score is None:
        mood_score = float(request.query_params.get("mood_score", "5.0"))
    
    if emotion is None:
        emotion = request.query_params.get("emotion", "neutral")
    
    interventions = []
    
    # Generate interventions based on emotion and mood
    if mood_score <= 3:
        interventions = [
            "Practice deep breathing for 5 minutes",
            "Try a short guided meditation",
            "Call a friend or family member for support",
            "Write down three things you're grateful for"
        ]
    elif mood_score <= 5:
        interventions = [
            "Take a short walk outside",
            "Listen to uplifting music",
            "Write in a journal for 10 minutes",
            "Do a simple stretching routine"
        ]
    else:
        interventions = [
            "Celebrate your positive mood!",
            "Share your good feelings with someone",
            "Engage in a hobby you enjoy",
            "Set a goal for tomorrow"
        ]
    
    return {
        "mood_score": mood_score,
        "emotion": emotion,
        "interventions": interventions
    }

@app.post("/api/intervention/advanced")
async def get_advanced_intervention(request: Request, current_user: User = Depends(get_current_active_user)):
    """Get advanced personalized intervention"""
    data = await request.json()
    
    # Get mood history
    try:
        if USE_MONGODB:
            entries = db_handler.get_mood_history(current_user.id, limit=20)
            
            mood_history = [
                {
                    'mood_score': entry.get("mood_score"),
                    'text_emotion': entry.get("text_emotion"),
                    'timestamp': entry.get("timestamp")
                }
                for entry in entries
            ]
        else:
            try:
                conn = sqlite3.connect('../data/mental_health.db')
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT mood_score, text_emotion, timestamp FROM mood_entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20",
                    (current_user.id,)
                )
                entries = cursor.fetchall()
                conn.close()
                
                mood_history = [
                    {
                        'mood_score': entry[0],
                        'text_emotion': entry[1],
                        'timestamp': entry[2]
                    }
                    for entry in entries
                ]
            except:
                mood_history = []
    except Exception as e:
        mood_history = []
    
    # Get personalized intervention
    intervention = advanced_intervention_engine.get_personalized_intervention(
        current_emotion=data.get('emotion', 'neutral'),
        mood_score=data.get('mood_score', 5),
        mood_history=mood_history,
        user_preferences=data.get('user_preferences', {})
    )
    
    return intervention

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time monitoring"""
    await connection_manager.connect(websocket, user_id)
    await realtime_monitor.start_monitoring(user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            mood_data = json.loads(data)
            
            # Process real-time mood update
            alerts = await realtime_monitor.process_mood_update(user_id, mood_data)
            
            # Send alerts if any
            for alert in alerts:
                await connection_manager.send_personal_message(
                    json.dumps(alert), user_id
                )
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, user_id)

# MongoDB specific routes
@app.get("/api/db/status")
async def db_status():
    """Get database status"""
    if USE_MONGODB:
        try:
            stats = db_handler.get_collection_stats()
            return {
                "status": "connected",
                "type": "MongoDB",
                "collections": stats
            }
        except Exception as e:
            return {
                "status": "error",
                "type": "MongoDB",
                "error": str(e)
            }
    else:
        try:
            # Check SQLite
            conn = sqlite3.connect('../data/mental_health.db')
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            return {
                "status": "connected",
                "type": "SQLite",
                "tables": [table[0] for table in tables]
            }
        except Exception as e:
            return {
                "status": "error",
                "type": "SQLite",
                "error": str(e)
            }

if __name__ == "__main__":
    print("Starting Mental Health Companion API (MongoDB-enabled)")
    print("Server running at http://localhost:8000")
    print("API documentation available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
