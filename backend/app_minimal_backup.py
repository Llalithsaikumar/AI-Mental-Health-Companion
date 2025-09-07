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
from pydantic import BaseModel
try:
    from passlib.context import CryptContext
    from jose import JWTError, jwt
    SECURITY_ENABLED = True
except ImportError:
    print("Warning: security packages not found. Install with: pip install python-jose passlib[bcrypt]")
    SECURITY_ENABLED = False

# Security configurations
SECRET_KEY = os.environ.get("SECRET_KEY", "this_is_a_secret_key_please_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

if SECURITY_ENABLED:
    # Password context for hashing
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Local imports - check for MongoDB support
try:
    from mongodb_handler import db_handler
    USE_MONGODB = True
    print("MongoDB handler loaded successfully")
except ImportError:
    print("MongoDB handler not available. Using SQLite as fallback.")
    import sqlite3
    USE_MONGODB = False

# Ensure data directory exists
data_dir = Path("../data")
data_dir.mkdir(exist_ok=True)

# OAuth2 password bearer token
if SECURITY_ENABLED:
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

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    full_name: Optional[str] = None

# Password hashing functions
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
        return password  # Not secure, just for testing

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
else:
    # Simple mock implementation
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        return f"mock_token_{data.get('sub', 'unknown')}"

# User authentication functions
def get_user(username: str):
    if USE_MONGODB:
        user_data = db_handler.get_user_by_username(username)
        if user_data:
            return UserInDB(**user_data)
    else:
        # Mock user data (for SQLite or no database)
        # In a real application, you would retrieve this from a database
        fake_users_db = {
            "testuser": {
                "username": "testuser",
                "full_name": "Test User",
                "email": "test@example.com",
                "hashed_password": get_password_hash("password123"),
                "disabled": False,
            }
        }
        if username in fake_users_db:
            user_dict = fake_users_db[username]
            return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

if SECURITY_ENABLED:
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

    async def get_current_active_user(current_user: User = Depends(get_current_user)):
        if current_user.disabled:
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user
else:
    # Mock implementations for when security packages are not available
    async def get_current_user(token: str = None):
        if token and token.startswith("mock_token_"):
            username = token.split("_")[-1]
            return get_user(username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    async def get_current_active_user(current_user: User = Depends(get_current_user)):
        if current_user.disabled:
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user

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

# This section is now moved below

# Standard library imports
import datetime
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third-party imports
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel

# Local imports
try:
    from mongodb_handler import db_handler
    USE_MONGODB = True
except ImportError:
    print("MongoDB handler not available. Using SQLite as fallback.")
    import sqlite3
    USE_MONGODB = False

# Ensure data directory exists
data_dir = Path("../data")
data_dir.mkdir(exist_ok=True)

# Import local modules - with fallbacks if they don't exist
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

# Create FastAPI app instance
app = FastAPI(title="Mental Health Companion API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock active tokens for authentication
active_tokens = {}

# Initialize advanced systems
advanced_intervention_engine = AdvancedInterventionEngine()
adaptive_learning = AdaptiveLearningSystem()
mental_health_predictor = MentalHealthPredictor()
crisis_assessor = CrisisRiskAssessment()
connection_manager = ConnectionManager()
realtime_monitor = RealTimeMonitoringSystem()
smart_notifications = SmartNotificationSystem()

# Basic routes
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
    
    # For compatibility with the old API
    user_id = getattr(user, "id", hash(user.username) % 10000)
    
    return {"access_token": access_token, "token_type": "bearer", "user_id": user_id}

# Advanced endpoints

# Websocket endpoint for real-time monitoring
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
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

@app.post("/api/mood/predict")
async def predict_mood(request: Request):
    """Predict future mood based on patterns"""
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    token = auth_header.split(' ')[1]
    if token not in active_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = active_tokens[token]['user_id']
    
    # Get mood history
    try:
        conn = sqlite3.connect('../data/mental_health.db')
        cursor = conn.cursor()
        cursor.execute(
            "SELECT mood_score, text_emotion, timestamp FROM mood_entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
            (user_id,)
        )
        entries = cursor.fetchall()
        conn.close()
    except sqlite3.OperationalError:
        # If database doesn't exist yet, return a message
        return {"message": "Insufficient data for prediction"}
    
    if len(entries) < 10:
        return {"message": "Insufficient data for prediction"}
    
    # Prepare data for prediction
    mood_history = [
        {
            'mood_score': entry[0],
            'text_emotion': entry[1],
            'timestamp': entry[2]
        }
        for entry in entries
    ]
    
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
async def assess_crisis_risk(request: Request):
    """Comprehensive crisis risk assessment"""
    data = await request.json()
    auth_header = request.headers.get('Authorization')
    
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    token = auth_header.split(' ')[1]
    if token not in active_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = active_tokens[token]['user_id']
    
    # Get mood history
    try:
        conn = sqlite3.connect('../data/mental_health.db')
        cursor = conn.cursor()
        cursor.execute(
            "SELECT mood_score, text_emotion, timestamp FROM mood_entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT 30",
            (user_id,)
        )
        entries = cursor.fetchall()
        conn.close()
    except sqlite3.OperationalError:
        # If database doesn't exist yet, use empty list
        entries = []
    
    mood_history = [
        {
            'mood_score': entry[0],
            'text_emotion': entry[1],
            'timestamp': entry[2]
        }
        for entry in entries
    ]
    
    # Assess crisis risk
    assessment = crisis_assessor.assess_crisis_risk(
        data.get('text', ''),
        data.get('mood_score', 5),
        mood_history
    )
    
    return assessment

@app.post("/api/intervention/advanced")
async def get_advanced_intervention(request: Request):
    """Get advanced personalized intervention"""
    data = await request.json()
    auth_header = request.headers.get('Authorization')
    
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    token = auth_header.split(' ')[1]
    if token not in active_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = active_tokens[token]['user_id']
    
    # Get mood history and user preferences
    try:
        conn = sqlite3.connect('../data/mental_health.db')
        cursor = conn.cursor()
        cursor.execute(
            "SELECT mood_score, text_emotion, timestamp FROM mood_entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20",
            (user_id,)
        )
        entries = cursor.fetchall()
        conn.close()
    except sqlite3.OperationalError:
        # If database doesn't exist yet, use empty list
        entries = []
    
    mood_history = [
        {
            'mood_score': entry[0],
            'text_emotion': entry[1],
            'timestamp': entry[2]
        }
        for entry in entries
    ]
    
    # Get personalized intervention
    intervention = advanced_intervention_engine.get_personalized_intervention(
        current_emotion=data.get('emotion', 'neutral'),
        mood_score=data.get('mood_score', 5),
        mood_history=mood_history,
        user_preferences=data.get('user_preferences', {})
    )
    
    return intervention

# Run the app directly if executed
if __name__ == "__main__":
    import uvicorn
    print("Starting Mental Health Companion API (Minimal Version)")
    print("Server running at http://localhost:8000")
    print("API documentation available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
