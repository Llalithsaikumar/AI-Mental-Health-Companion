import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List
import pickle
import cv2
import librosa
from datetime import datetime

class AdvancedTextEmotionModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 8)  # 8 emotions
        self.emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return torch.softmax(logits, dim=-1)

class SpeechEmotionAnalyzer:
    def __init__(self):
        self.sample_rate = 16000
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
    def extract_advanced_features(self, audio_path: str) -> np.ndarray:
        """Extract comprehensive audio features"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # MFCC features (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        
        # Combine all features
        features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1),
            np.mean(zcr),
            np.std(zcr)
        ]).reshape(1, -1)
        
        return features

class FacialEmotionAnalyzer:
    def __init__(self):
        # Load pre-trained face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
    def detect_facial_emotion(self, image_path: str) -> Dict[str, float]:
        """Advanced facial emotion detection"""
        try:
            from deepface import DeepFace
            
            # Analyze emotions
            result = DeepFace.analyze(
                img_path=image_path, 
                actions=['emotion'],
                enforce_detection=False
            )
            
            emotions = result['emotion'] if isinstance(result, dict) else result[0]['emotion']
            
            # Normalize probabilities
            total = sum(emotions.values())
            normalized_emotions = {k: v/total for k, v in emotions.items()}
            
            return normalized_emotions
            
        except Exception as e:
            # Fallback to simple detection
            return {'neutral': 1.0}

class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        
    def analyze_sentiment_intensity(self, text: str) -> Dict[str, float]:
        """Advanced sentiment analysis with intensity scores"""
        from transformers import pipeline
        
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True
        )
        
        results = sentiment_pipeline(text)
        sentiment_scores = {item['label'].lower(): item['score'] for item in results[0]}
        
        return sentiment_scores
