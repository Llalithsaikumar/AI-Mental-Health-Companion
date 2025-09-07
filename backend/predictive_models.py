import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MentalHealthPredictor:
    def __init__(self):
        self.mood_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, mood_history: List[Dict]) -> pd.DataFrame:
        """Prepare features from mood history for ML models"""
        df = pd.DataFrame(mood_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Rolling statistics
        df['mood_7d_avg'] = df['mood_score'].rolling(window=7, min_periods=1).mean()
        df['mood_7d_std'] = df['mood_score'].rolling(window=7, min_periods=1).std()
        df['mood_trend'] = df['mood_score'].diff()
        
        # Emotion encoding
        emotions = ['joy', 'sadness', 'anger', 'fear', 'neutral']
        for emotion in emotions:
            df[f'emotion_{emotion}'] = (df['text_emotion'] == emotion).astype(int)
        
        # Sleep pattern proxy (entries late at night or early morning)
        df['late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
        
        return df
        
    def train_models(self, mood_history: List[Dict]):
        """Train predictive models"""
        if len(mood_history) < 10:
            return False
            
        df = self.prepare_features(mood_history)
        
        # Features for prediction
        feature_cols = ['hour', 'day_of_week', 'is_weekend', 'mood_7d_avg', 
                       'mood_7d_std', 'mood_trend', 'late_night'] + \
                      [f'emotion_{e}' for e in ['joy', 'sadness', 'anger', 'fear', 'neutral']]
        
        # Remove rows with NaN
        df_clean = df.dropna(subset=feature_cols + ['mood_score'])
        
        if len(df_clean) < 5:
            return False
            
        X = df_clean[feature_cols]
        y = df_clean['mood_score']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train mood predictor
        self.mood_predictor.fit(X_scaled, y)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_scaled)
        
        self.is_trained = True
        return True
        
    def predict_next_mood(self, current_features: Dict) -> Dict:
        """Predict next mood score"""
        if not self.is_trained:
            return {'predicted_mood': 5.0, 'confidence': 0.0}
            
        # Convert to feature vector
        feature_vector = np.array([[
            current_features.get('hour', 12),
            current_features.get('day_of_week', 1),
            current_features.get('is_weekend', 0),
            current_features.get('mood_7d_avg', 5.0),
            current_features.get('mood_7d_std', 1.0),
            current_features.get('mood_trend', 0.0),
            current_features.get('late_night', 0),
            current_features.get('emotion_joy', 0),
            current_features.get('emotion_sadness', 0),
            current_features.get('emotion_anger', 0),
            current_features.get('emotion_fear', 0),
            current_features.get('emotion_neutral', 1),
        ]])
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        predicted_mood = self.mood_predictor.predict(feature_vector_scaled)[0]
        
        # Calculate confidence (inverse of prediction variance)
        confidence = min(1.0, 1.0 / (np.var(self.mood_predictor.estimators_[0].predict([feature_vector_scaled[0]])) + 0.1))
        
        return {
            'predicted_mood': round(float(predicted_mood), 2),
            'confidence': round(float(confidence), 2)
        }
        
    def detect_anomaly(self, current_features: Dict) -> Dict:
        """Detect anomalous mental health patterns"""
        if not self.is_trained:
            return {'is_anomaly': False, 'anomaly_score': 0.0}
            
        feature_vector = np.array([[
            current_features.get('hour', 12),
            current_features.get('day_of_week', 1),
            current_features.get('is_weekend', 0),
            current_features.get('mood_7d_avg', 5.0),
            current_features.get('mood_7d_std', 1.0),
            current_features.get('mood_trend', 0.0),
            current_features.get('late_night', 0),
            current_features.get('emotion_joy', 0),
            current_features.get('emotion_sadness', 0),
            current_features.get('emotion_anger', 0),
            current_features.get('emotion_fear', 0),
            current_features.get('emotion_neutral', 1),
        ]])
        
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Detect anomaly
        is_anomaly = self.anomaly_detector.predict(feature_vector_scaled)[0] == -1
        anomaly_score = self.anomaly_detector.decision_function(feature_vector_scaled)[0]
        
        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': round(float(anomaly_score), 3)
        }

class CrisisRiskAssessment:
    def __init__(self):
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'not worth living',
            'hopeless', 'helpless', 'worthless', 'better off dead',
            'no point', 'give up', 'can\'t go on', 'end the pain'
        ]
        
        self.warning_keywords = [
            'depressed', 'anxiety', 'panic', 'overwhelmed',
            'isolated', 'alone', 'scared', 'desperate'
        ]
        
    def assess_crisis_risk(self, text: str, mood_score: float, mood_history: List[Dict]) -> Dict:
        """Comprehensive crisis risk assessment"""
        risk_score = 0.0
        risk_factors = []
        
        text_lower = text.lower()
        
        # Text-based risk factors
        crisis_matches = sum(1 for keyword in self.crisis_keywords if keyword in text_lower)
        warning_matches = sum(1 for keyword in self.warning_keywords if keyword in text_lower)
        
        if crisis_matches > 0:
            risk_score += crisis_matches * 0.4
            risk_factors.append(f"Crisis language detected ({crisis_matches} instances)")
            
        if warning_matches > 0:
            risk_score += warning_matches * 0.2
            risk_factors.append(f"Warning signs in language ({warning_matches} instances)")
            
        # Mood-based risk factors
        if mood_score <= 2:
            risk_score += 0.3
            risk_factors.append("Severely low mood score")
        elif mood_score <= 3:
            risk_score += 0.2
            risk_factors.append("Very low mood score")
            
        # Historical pattern analysis
        if len(mood_history) >= 7:
            recent_moods = [entry['mood_score'] for entry in mood_history[-7:]]
            avg_recent_mood = np.mean(recent_moods)
            mood_decline = np.mean(mood_history[-14:-7]) - avg_recent_mood if len(mood_history) >= 14 else 0
            
            if avg_recent_mood <= 3:
                risk_score += 0.2
                risk_factors.append("Consistently low mood over past week")
                
            if mood_decline > 2:
                risk_score += 0.2
                risk_factors.append("Significant mood decline trend")
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        elif risk_score >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
            
        return {
            'risk_level': risk_level,
            'risk_score': round(risk_score, 2),
            'risk_factors': risk_factors,
            'requires_intervention': risk_score >= 0.4
        }
