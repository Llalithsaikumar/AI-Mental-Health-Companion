import re
import numpy as np
import librosa
from transformers import AutoTokenizer
import cv2
from typing import List, Dict, Any

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text input"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove special characters but keep punctuation for emotion
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def tokenize_text(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """Tokenize text for model input"""
        cleaned = self.clean_text(text)
        return self.tokenizer(
            cleaned,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

class AudioPreprocessor:
    def __init__(self, sample_rate: int = 16000, duration: float = 5.0):
        self.sample_rate = sample_rate
        self.duration = duration
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            return audio
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return np.zeros(int(self.sample_rate * self.duration))
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract audio features for emotion recognition"""
        features = {}
        
        # MFCC features
        features['mfcc'] = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=13
        ).T
        
        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate
        ).T
        
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate
        ).T
        
        # Zero crossing rate
        features['zcr'] = librosa.feature.zero_crossing_rate(audio).T
        
        return features

class ImagePreprocessor:
    def __init__(self, target_size: tuple = (224, 224)):
        self.target_size = target_size
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for facial emotion recognition"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def detect_faces(self, image_path: str) -> List[np.ndarray]:
        """Detect and extract faces from image"""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_images = []
        
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, self.target_size)
            face_images.append(face)
        
        return face_images

def calculate_mood_score(emotions: Dict[str, float]) -> float:
    """Calculate overall mood score from emotion probabilities"""
    emotion_weights = {
        'joy': 1.0,
        'happiness': 1.0,
        'love': 0.8,
        'surprise': 0.3,
        'neutral': 0.0,
        'fear': -0.6,
        'anger': -0.8,
        'sadness': -0.9,
        'disgust': -0.7
    }
    
    weighted_score = sum(
        emotions.get(emotion, 0) * weight 
        for emotion, weight in emotion_weights.items()
    )
    
    # Normalize to 0-10 scale
    normalized_score = (weighted_score + 1) * 5
    return max(0, min(10, normalized_score))
