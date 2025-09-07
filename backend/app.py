from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
import torch
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
import pandas as pd
from typing import Dict, List, Optional
import hashlib
from cryptography.fernet import Fernet
import jwt
from passlib.context import CryptContext
import uvicorn
import sys
sys.path.append('..')

# Initialize FastAPI app
app = FastAPI(title="AI Mental Health Companion API", version="1.0.0")

# Try to import models and preprocessing utilities, but handle the case if they're not available
try:
    from utils.preprocessing import TextPreprocessor, AudioPreprocessor, calculate_mood_score
    text_processor = TextPreprocessor()
    audio_processor = AudioPreprocessor()
except ImportError as e:
    print(f"Error importing preprocessing utilities: {e}")
    text_processor = None
    audio_processor = None
    
try:
    from models.emotion_models import TextEmotionModel, AudioEmotionModel
except ImportError as e:
    print(f"Error importing emotion models: {e}")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mental_health.db")
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security setup
SECRET_KEY = "your-secret-key-change-this"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Encryption setup
cipher_key = Fernet.generate_key()
cipher = Fernet(cipher_key)

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class MoodEntry(Base):
    __tablename__ = "mood_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    text_emotion = Column(String)
    audio_emotion = Column(String)
    facial_emotion = Column(String)
    mood_score = Column(Float)
    journal_text = Column(Text)
    encrypted_data = Column(Text)

class InterventionLog(Base):
    __tablename__ = "intervention_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    intervention_type = Column(String)
    content = Column(Text)
    effectiveness_rating = Column(Float, nullable=True)

Base.metadata.create_all(bind=engine)

# Pydantic Models
class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class MoodEntryCreate(BaseModel):
    text_input: Optional[str] = None
    mood_score: Optional[float] = None
    journal_text: Optional[str] = None

class InterventionRequest(BaseModel):
    user_id: int

class InterventionResponse(BaseModel):
    intervention_type: str
    content: str
    resources: List[str]

# Load AI Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize preprocessors if available
text_preprocessor = text_processor if text_processor else None
audio_preprocessor = audio_processor if audio_processor else None

# Load models
try:
    text_model = TextEmotionModel() if 'TextEmotionModel' in globals() else None
    model_paths = [
        'models/text_model.pth',
        '../models/text_model.pth',
        'c:/Users/lalit/OneDrive/Desktop/Project/AI-Mental-Health-Companion/models/text_model.pth'
    ]
    model_loaded = False
    for path in model_paths:
        if text_model and os.path.exists(path):
            try:
                text_model.load_state_dict(torch.load(path, map_location=device))
                text_model.to(device)
                text_model.eval()
                model_loaded = True
                print(f"Text model loaded from {path}")
                break
            except Exception as e:
                print(f"Failed to load from {path}: {e}")
                
    if not model_loaded:
        print("Text model not available or model file not found")
        text_model = None
except Exception as e:
    print(f"Error loading text model: {e}")
    text_model = None

try:
    audio_model = AudioEmotionModel() if 'AudioEmotionModel' in globals() else None
    model_paths = [
        'models/audio_model.pth',
        '../models/audio_model.pth',
        'c:/Users/lalit/OneDrive/Desktop/Project/AI-Mental-Health-Companion/models/audio_model.pth'
    ]
    model_loaded = False
    for path in model_paths:
        if audio_model and os.path.exists(path):
            try:
                audio_model.load_state_dict(torch.load(path, map_location=device))
                audio_model.to(device)
                audio_model.eval()
                model_loaded = True
                print(f"Audio model loaded from {path}")
                break
            except Exception as e:
                print(f"Failed to load from {path}: {e}")
                
    if not model_loaded:
        print("Audio model not available or model file not found")
        audio_model = None
except Exception as e:
    print(f"Error loading audio model: {e}")
    audio_model = None

# Dependency functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Intervention Engine
class InterventionEngine:
    def __init__(self):
        self.interventions = {
            'anger': {
                'breathing': "Try the 4-7-8 breathing technique: Inhale for 4, hold for 7, exhale for 8.",
                'activity': "Take a 5-minute walk or do some light stretching.",
                'cognitive': "Ask yourself: 'Will this matter in 5 years?' to gain perspective.",
                'resources': [
                    "Anger Management Workbook",
                    "Progressive Muscle Relaxation Guide",
                    "Mindfulness for Anger Control"
                ]
            },
            'sadness': {
                'journaling': "Write about three things you're grateful for today.",
                'social': "Reach out to a friend or family member you trust.",
                'activity': "Listen to uplifting music or watch a favorite movie.",
                'resources': [
                    "Depression Self-Help Guide",
                    "Mood Tracking Journal",
                    "Support Group Directory"
                ]
            },
            'fear': {
                'grounding': "Use the 5-4-3-2-1 technique: 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste.",
                'breathing': "Practice square breathing: 4 counts in, hold 4, out 4, hold 4.",
                'cognitive': "Challenge the thought: 'What evidence supports this fear?'",
                'resources': [
                    "Anxiety Management Toolkit",
                    "Fear Facing Workbook",
                    "Relaxation Techniques Guide"
                ]
            },
            'joy': {
                'sharing': "Share your positive experience with someone you care about.",
                'gratitude': "Write down what contributed to this good feeling.",
                'mindfulness': "Take a moment to fully savor this positive emotion.",
                'resources': [
                    "Positive Psychology Guide",
                    "Gratitude Journal Template",
                    "Happiness Habits Checklist"
                ]
            }
        }
        
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'not worth living', 
            'hopeless', 'can\'t go on', 'better off dead'
        ]
    
    def detect_crisis(self, text: str) -> bool:
        """Detect if text contains crisis indicators"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crisis_keywords)
    
    def get_intervention(self, emotion: str, mood_history: List[Dict]) -> Dict:
        """Get personalized intervention based on emotion and history"""
        if emotion not in self.interventions:
            emotion = 'sadness'  # Default fallback
        
        intervention_data = self.interventions[emotion]
        
        # Analyze mood history for personalization
        recent_moods = [entry.get('mood_score', 5) for entry in mood_history[-7:]]  # Last week
        avg_mood = np.mean(recent_moods) if recent_moods else 5
        
        # Select intervention based on mood trends
        if avg_mood < 3:  # Consistently low mood
            primary_intervention = intervention_data.get('cognitive', list(intervention_data.values())[0])
        elif avg_mood > 7:  # Generally positive
            primary_intervention = intervention_data.get('mindfulness', list(intervention_data.values())[0])
        else:
            primary_intervention = intervention_data.get('breathing', list(intervention_data.values())[0])
        
        return {
            'intervention_type': emotion,
            'content': primary_intervention,
            'resources': intervention_data['resources'],
            'additional_suggestions': list(intervention_data.values())[:3]
        }
    
    def get_crisis_intervention(self) -> Dict:
        """Emergency intervention for crisis situations"""
        return {
            'intervention_type': 'crisis',
            'content': "I'm concerned about you. Please reach out for immediate help.",
            'resources': [
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "Emergency Services: 911"
            ],
            'emergency': True
        }

intervention_engine = InterventionEngine()

# API Endpoints

@app.post("/api/auth/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return {"message": "User registered successfully"}

@app.post("/api/auth/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer", "user_id": db_user.id}

@app.post("/api/emotion/text")
def analyze_text_emotion(text: str, current_user: User = Depends(get_current_user)):
    if not text_model:
        raise HTTPException(status_code=503, detail="Text emotion model not available")
    
    try:
        # Preprocess text
        tokenized = text_preprocessor.tokenize_text(text)
        
        # Get emotion prediction
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        
        emotions = text_model.predict_emotion(input_ids, attention_mask)
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        
        return {
            "emotions": emotions,
            "dominant_emotion": dominant_emotion[0],
            "confidence": dominant_emotion[1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text emotion: {str(e)}")

@app.post("/api/emotion/audio")
async def analyze_audio_emotion(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    if not audio_model:
        raise HTTPException(status_code=503, detail="Audio emotion model not available")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process audio
        audio_data = audio_preprocessor.load_audio(tmp_file_path)
        features = audio_preprocessor.extract_features(audio_data)
        
        # Combine features for model input
        combined_features = np.concatenate([
            features['mfcc'].flatten(),
            features['spectral_centroid'].flatten(),
            features['spectral_rolloff'].flatten(),
            features['zcr'].flatten()
        ])
        
        # Ensure fixed size
        if len(combined_features) < 128:
            combined_features = np.pad(combined_features, (0, 128 - len(combined_features)))
        else:
            combined_features = combined_features[:128]
        
        # Convert to tensor
        feature_tensor = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Get emotion prediction
        emotions = audio_model.predict_emotion(feature_tensor)
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return {
            "emotions": emotions,
            "dominant_emotion": dominant_emotion[0],
            "confidence": dominant_emotion[1]
        }
    except Exception as e:
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)
        raise HTTPException(status_code=500, detail=f"Error analyzing audio emotion: {str(e)}")

@app.post("/api/mood/entry")
def create_mood_entry(entry: MoodEntryCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        text_emotion = None
        mood_score = entry.mood_score or 5.0
        
        # Analyze text if provided
        if entry.text_input and text_model:
            tokenized = text_preprocessor.tokenize_text(entry.text_input)
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            
            emotions = text_model.predict_emotion(input_ids, attention_mask)
            text_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Calculate mood score from emotions if not provided
            if not entry.mood_score:
                mood_score = calculate_mood_score(emotions)
        
        # Check for crisis indicators
        is_crisis = False
        if entry.text_input:
            is_crisis = intervention_engine.detect_crisis(entry.text_input)
        
        # Encrypt sensitive data
        sensitive_data = {
            'text_input': entry.text_input,
            'journal_text': entry.journal_text
        }
        encrypted_data = cipher.encrypt(str(sensitive_data).encode())
        
        # Create mood entry
        db_entry = MoodEntry(
            user_id=current_user.id,
            text_emotion=text_emotion,
            mood_score=mood_score,
            journal_text=entry.journal_text,
            encrypted_data=encrypted_data.decode()
        )
        
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        
        response = {
            "id": db_entry.id,
            "mood_score": mood_score,
            "text_emotion": text_emotion,
            "timestamp": db_entry.timestamp,
            "crisis_detected": is_crisis
        }
        
        if is_crisis:
            response["crisis_intervention"] = intervention_engine.get_crisis_intervention()
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating mood entry: {str(e)}")

@app.get("/api/mood/history")
def get_mood_history(days: int = 30, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        entries = db.query(MoodEntry).filter(
            MoodEntry.user_id == current_user.id,
            MoodEntry.timestamp >= start_date
        ).order_by(MoodEntry.timestamp.desc()).all()
        
        history = []
        for entry in entries:
            history.append({
                "id": entry.id,
                "timestamp": entry.timestamp,
                "mood_score": entry.mood_score,
                "text_emotion": entry.text_emotion,
                "audio_emotion": entry.audio_emotion,
                "facial_emotion": entry.facial_emotion
            })
        
        return {"history": history}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving mood history: {str(e)}")

@app.post("/api/intervention/recommend")
def get_intervention_recommendation(request: InterventionRequest, db: Session = Depends(get_db)):
    try:
        # Get recent mood history
        recent_entries = db.query(MoodEntry).filter(
            MoodEntry.user_id == request.user_id
        ).order_by(MoodEntry.timestamp.desc()).limit(10).all()
        
        if not recent_entries:
            return {"message": "No mood data available for recommendations"}
        
        # Get most recent emotion
        latest_entry = recent_entries[0]
        current_emotion = latest_entry.text_emotion or 'neutral'
        
        # Convert to history format
        mood_history = [
            {
                "mood_score": entry.mood_score,
                "emotion": entry.text_emotion,
                "timestamp": entry.timestamp
            }
            for entry in recent_entries
        ]
        
        # Get intervention recommendation
        intervention = intervention_engine.get_intervention(current_emotion, mood_history)
        
        # Log intervention
        intervention_log = InterventionLog(
            user_id=request.user_id,
            intervention_type=intervention['intervention_type'],
            content=intervention['content']
        )
        db.add(intervention_log)
        db.commit()
        
        return intervention
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating intervention: {str(e)}")

@app.get("/api/analytics/mood-trends")
def get_mood_trends(days: int = 30, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        entries = db.query(MoodEntry).filter(
            MoodEntry.user_id == current_user.id,
            MoodEntry.timestamp >= start_date
        ).all()
        
        if not entries:
            return {"message": "No data available for analysis"}
        
        # Convert to DataFrame for analysis
        data = []
        for entry in entries:
            data.append({
                'date': entry.timestamp.date(),
                'mood_score': entry.mood_score,
                'emotion': entry.text_emotion
            })
        
        df = pd.DataFrame(data)
        
        # Calculate trends
        daily_avg = df.groupby('date')['mood_score'].mean()
        emotion_counts = df['emotion'].value_counts()
        
        # Calculate mood statistics
        avg_mood = df['mood_score'].mean()
        mood_trend = "stable"
        
        if len(daily_avg) >= 7:
            recent_avg = daily_avg.tail(7).mean()
            older_avg = daily_avg.head(7).mean() if len(daily_avg) >= 14 else recent_avg
            
            if recent_avg > older_avg + 0.5:
                mood_trend = "improving"
            elif recent_avg < older_avg - 0.5:
                mood_trend = "declining"
        
        return {
            "average_mood": round(avg_mood, 2),
            "mood_trend": mood_trend,
            "daily_averages": {str(date): round(score, 2) for date, score in daily_avg.items()},
            "emotion_distribution": emotion_counts.to_dict(),
            "total_entries": len(entries)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing mood trends: {str(e)}")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "models_loaded": {
            "text_model": text_model is not None,
            "audio_model": audio_model is not None
        }
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)