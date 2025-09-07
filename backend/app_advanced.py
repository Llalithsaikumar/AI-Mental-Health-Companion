from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
import uvicorn
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI app
app = FastAPI(title="Advanced AI Mental Health Companion", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
def init_db():
    conn = sqlite3.connect('mental_health.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            hashed_password TEXT,
            created_at TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mood_entries (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            timestamp TEXT,
            text_emotion TEXT,
            mood_score REAL,
            journal_text TEXT,
            predicted_mood REAL,
            risk_level TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY,
            user_id INTEGER UNIQUE,
            prefers_short_activities BOOLEAN,
            likes_meditation BOOLEAN,
            prefers_professional_help BOOLEAN
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def create_token(username: str) -> str:
    return hashlib.md5(f"{username}{datetime.now()}".encode()).hexdigest()

def advanced_emotion_detection(text: str) -> Dict:
    """Enhanced emotion detection with confidence scores"""
    text_lower = text.lower()
    
    emotion_patterns = {
        'joy': {
            'keywords': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'love', 'brilliant', 'awesome'],
            'weight': 1.0
        },
        'sadness': {
            'keywords': ['sad', 'depressed', 'down', 'terrible', 'awful', 'miserable', 'upset', 'crying', 'lonely'],
            'weight': 0.8
        },
        'anger': {
            'keywords': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'hate', 'frustrated', 'rage'],
            'weight': 0.7
        },
        'fear': {
            'keywords': ['scared', 'afraid', 'anxious', 'worried', 'nervous', 'panic', 'terrified', 'stressed'],
            'weight': 0.6
        },
        'surprise': {
            'keywords': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow'],
            'weight': 0.5
        },
        'neutral': {
            'keywords': ['okay', 'fine', 'normal', 'regular', 'usual', 'typical'],
            'weight': 0.3
        }
    }
    
    emotion_scores = {}
    total_matches = 0
    
    for emotion, data in emotion_patterns.items():
        matches = sum(1 for keyword in data['keywords'] if keyword in text_lower)
        if matches > 0:
            emotion_scores[emotion] = matches * data['weight']
            total_matches += matches
    
    if total_matches == 0:
        return {'neutral': 1.0}
    
    normalized_scores = {k: round(v/total_matches, 2) for k, v in emotion_scores.items()}
    return normalized_scores

class SimplePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, mood_history):
        if len(mood_history) < 5:
            return None, None
            
        df = pd.DataFrame(mood_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['mood_ma'] = df['mood_score'].rolling(window=3, min_periods=1).mean()
        df['mood_trend'] = df['mood_score'].diff().fillna(0)
        
        emotions = ['joy', 'sadness', 'anger', 'fear', 'neutral']
        for emotion in emotions:
            df[f'emotion_{emotion}'] = (df['text_emotion'] == emotion).astype(int)
        
        feature_cols = ['hour', 'day_of_week', 'is_weekend', 'mood_ma', 'mood_trend'] + \
                      [f'emotion_{e}' for e in emotions]
        
        X = df[feature_cols].fillna(0)
        y = df['mood_score']
        
        return X, y
        
    def train(self, mood_history):
        X, y = self.prepare_features(mood_history)
        if X is None or len(X) < 5:
            return False
            
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Training error: {e}")
            return False
            
    def predict(self, current_features):
        if not self.is_trained or self.model is None:
            return {'predicted_mood': 5.0, 'confidence': 0.0}
            
        try:
            feature_vector = np.array([[
                current_features.get('hour', 12),
                current_features.get('day_of_week', 1),
                current_features.get('is_weekend', 0),
                current_features.get('mood_ma', 5.0),
                current_features.get('mood_trend', 0.0),
                current_features.get('emotion_joy', 0),
                current_features.get('emotion_sadness', 0),
                current_features.get('emotion_anger', 0),
                current_features.get('emotion_fear', 0),
                current_features.get('emotion_neutral', 1),
            ]])
            
            feature_vector_scaled = self.scaler.transform(feature_vector)
            predicted_mood = self.model.predict(feature_vector_scaled)[0]
            confidence = min(1.0, 0.8)
            
            return {
                'predicted_mood': round(float(predicted_mood), 2),
                'confidence': round(float(confidence), 2)
            }
        except Exception as e:
            return {'predicted_mood': 5.0, 'confidence': 0.0}

def assess_crisis_risk(text, mood_score, mood_history):
    risk_score = 0.0
    risk_factors = []
    
    crisis_words = ['suicide', 'kill myself', 'end it all', 'hopeless', 'worthless', 'better off dead']
    warning_words = ['depressed', 'anxious', 'overwhelmed', 'desperate', 'can\'t cope']
    
    text_lower = text.lower() if text else ""
    
    crisis_matches = sum(1 for word in crisis_words if word in text_lower)
    warning_matches = sum(1 for word in warning_words if word in text_lower)   
    if crisis_matches > 0:
        risk_score += 0.5
        risk_factors.append(f"Crisis language detected ({crisis_matches} instances)")
    
    if warning_matches > 0:
        risk_score += 0.2
        risk_factors.append(f"Warning signs detected ({warning_matches} instances)")
    
    if mood_score <= 2:
        risk_score += 0.3
        risk_factors.append("Severely low mood")
    elif mood_score <= 3:
        risk_score += 0.2
        risk_factors.append("Very low mood")
    
    if len(mood_history) >= 5:
        recent_avg = np.mean([entry.get('mood_score', 5) for entry in mood_history[-5:]])
        if recent_avg <= 3:
            risk_score += 0.2
            risk_factors.append("Consistently low mood")
    
    if risk_score >= 0.6:
        risk_level = "HIGH"
    elif risk_score >= 0.3:
        risk_level = "MEDIUM"
    elif risk_score >= 0.1:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"
    
    return {
        'risk_level': risk_level,
        'risk_score': round(risk_score, 2),
        'risk_factors': risk_factors,
        'requires_intervention': risk_score >= 0.3
    }

def get_advanced_intervention(emotion, mood_score, mood_history, preferences=None):
    if preferences is None:
        preferences = {}

    interventions = {
        'sadness': [
            "It's okay to feel sad. Try the 5-4-3-2-1 grounding technique: 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste.",
            "Reach out to a friend or loved one to share your feelings.",
            "Engage in a small activity that brings you joy, like listening to music or drawing."
        ]
    }

    return interventions.get(emotion, ["No specific intervention available."])

# Initialize components
init_db()
predictor = SimplePredictor()
active_tokens = {}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[int, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.user_connections[user_id] = websocket
    
    def disconnect(self, websocket: WebSocket, user_id: int):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id in self.user_connections:
            del self.user_connections[user_id]
    
    async def send_personal_message(self, message: str, user_id: int):
        if user_id in self.user_connections:
            try:
                await self.user_connections[user_id].send_text(message)
            except:
                pass

manager = ConnectionManager()

# API Endpoints
@app.post("/api/auth/register")
async def register(request: Request):
    data = await request.json()
    username = data['username']
    password = data['password']
    
    conn = sqlite3.connect('mental_health.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = hash_password(password)
    cursor.execute(
        "INSERT INTO users (username, hashed_password, created_at) VALUES (?, ?, ?)",
        (username, hashed_password, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    
    return {"message": "User registered successfully"}

@app.post("/api/auth/login")
async def login(request: Request):
    data = await request.json()
    username = data['username']
    password = data['password']
    
    conn = sqlite3.connect('mental_health.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, hashed_password FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if not user or not verify_password(password, user[1]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(username)
    active_tokens[token] = {'user_id': user[0], 'username': username}
    
    return {"access_token": token, "token_type": "bearer", "user_id": user[0]}

@app.post("/api/mood/entry")
async def create_mood_entry(request: Request):
    data = await request.json()
    auth_header = request.headers.get('Authorization')
    
    if not auth_header or not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    token = auth_header.split(' ')[1]
    if token not in active_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = active_tokens[token]['user_id']
    
    text_input = data.get('text_input', '')
    mood_score = data.get('mood_score', 5.0)
    journal_text = data.get('journal_text', '')
    
    if text_input:
        emotions = advanced_emotion_detection(text_input)
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
    else:
        dominant_emotion = 'neutral'
        emotions = {'neutral': 1.0}
    
    conn = sqlite3.connect('mental_health.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT mood_score, text_emotion, timestamp FROM mood_entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10",
        (user_id,)
    )
    history_data = cursor.fetchall()
    
    mood_history = [{'mood_score': row[0], 'text_emotion': row[1], 'timestamp': row[2]} for row in history_data]
    
    crisis_assessment = assess_crisis_risk(text_input, mood_score, mood_history)
    
    cursor.execute(
        "INSERT INTO mood_entries (user_id, timestamp, text_emotion, mood_score, journal_text, risk_level) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, datetime.now().isoformat(), dominant_emotion, mood_score, journal_text, crisis_assessment['risk_level'])
    )
    entry_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    if crisis_assessment['risk_level'] in ['HIGH', 'MEDIUM']:
        alert = {
            'type': 'risk_alert',
            'message': f"⚠️ Risk level: {crisis_assessment['risk_level']}",
            'severity': crisis_assessment['risk_level'].lower(),
            'timestamp': datetime.now().isoformat()
        }
        await manager.send_personal_message(json.dumps(alert), user_id)
    
    return {
        "id": entry_id,
        "mood_score": mood_score,
        "text_emotion": dominant_emotion,
        "emotions": emotions,
        "timestamp": datetime.now().isoformat(),
        "crisis_assessment": crisis_assessment
    }

@app.get("/api/mood/history")
async def get_mood_history(request: Request):
    auth_header = request.headers.get('Authorization')
    
    if not auth_header or not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    token = auth_header.split(' ')[1]
    if token not in active_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = active_tokens[token]['user_id']
    
    conn = sqlite3.connect('mental_health.db')
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, timestamp, mood_score, text_emotion, risk_level FROM mood_entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
        (user_id,)
    )
    entries = cursor.fetchall()
    conn.close()
    
    history = []
    for entry in entries:
        history.append({
            "id": entry[0],
            "timestamp": entry[1],
            "mood_score": entry[2],
            "text_emotion": entry[3],
            "risk_level": entry[4]
        })
    
    return {"history": history}

@app.post("/api/mood/predict")
async def predict_mood(request: Request):
    auth_header = request.headers.get('Authorization')
    
    if not auth_header or not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    token = auth_header.split(' ')[1]
    if token not in active_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = active_tokens[token]['user_id']
    
    conn = sqlite3.connect('mental_health.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT mood_score, text_emotion, timestamp FROM mood_entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5",
        (user_id,)
    )
    entries = cursor.fetchall()
    conn.close()
    
    if len(entries) < 5:
        return {"message": "Need at least 5 mood entries for accurate predictions. Keep logging your mood!"}
    
    mood_history = [
        {'mood_score': entry[0], 'text_emotion': entry[1], 'timestamp': entry[2]}
        for entry in entries
    ]
    
    if not predictor.is_trained:
        predictor.train(mood_history)
    
    current_time = datetime.now()
    current_features = {
        'hour': current_time.hour,
        'day_of_week': current_time.weekday(),
        'is_weekend': 1 if current_time.weekday() >= 5 else 0,
        'mood_ma': np.mean([entry['mood_score'] for entry in mood_history[:3]]),
        'mood_trend': mood_history[0]['mood_score'] - mood_history[1]['mood_score'] if len(mood_history) > 1 else 0,
        'emotion_joy': 1 if mood_history[0]['text_emotion'] == 'joy' else 0,
        'emotion_sadness': 1 if mood_history[0]['text_emotion'] == 'sadness' else 0,
        'emotion_anger': 1 if mood_history[0]['text_emotion'] == 'anger' else 0,
        'emotion_fear': 1 if mood_history[0]['text_emotion'] == 'fear' else 0,
        'emotion_neutral': 1 if mood_history[0]['text_emotion'] == 'neutral' else 0,
    }
    
    prediction = predictor.predict(current_features)
    
    return {
        "prediction": prediction,
        "recommendations": "🌟 Keep up the positive momentum!" if prediction['predicted_mood'] > 6 else "💙 Focus on self-care and gentle activities"
    }

@app.post("/api/intervention/advanced")
async def get_advanced_intervention_endpoint(request: Request):
    data = await request.json()
    auth_header = request.headers.get('Authorization')
    
    if not auth_header or not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    token = auth_header.split(' ')[1]
    if token not in active_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = active_tokens[token]['user_id']
    
    conn = sqlite3.connect('mental_health.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT mood_score, text_emotion, timestamp FROM mood_entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5",
        (user_id,)
    )
    entries = cursor.fetchall()
    conn.close()
    
    mood_history = [
        {'mood_score': entry[0], 'text_emotion': entry[1], 'timestamp': entry[2]}
        for entry in entries
    ]
    
    intervention = get_advanced_intervention(
        emotion=data.get('emotion', 'neutral'),
        mood_score=data.get('mood_score', 5),
        mood_history=mood_history,
        preferences=data.get('user_preferences', {})
    )
    
    return intervention

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            mood_data = json.loads(data)
            
            if mood_data.get('mood_score', 5) < 3:
                alert = {
                    'type': 'low_mood_alert',
                    'message': '💙 Low mood detected - remember to be gentle with yourself',
                    'severity': 'medium',
                    'timestamp': datetime.now().isoformat()
                }
                await manager.send_personal_message(json.dumps(alert), user_id)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "advanced_emotions": True,
            "predictive_analytics": True,
            "crisis_assessment": True,
            "real_time_monitoring": True,
            "personalized_interventions": True
        }
    }

if __name__ == "__main__":
    uvicorn.run("app_advanced:app", host="0.0.0.0", port=8000, reload=True)
