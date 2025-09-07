import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import numpy as np
from typing import Dict, List, Tuple

class TextEmotionModel(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", num_emotions: int = 6):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.config.hidden_size, num_emotions)
        
        # Emotion labels
        self.emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return torch.softmax(logits, dim=-1)
    
    def predict_emotion(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            probabilities = self.forward(input_ids, attention_mask)
            probs_dict = {
                emotion: float(prob) 
                for emotion, prob in zip(self.emotions, probabilities[0])
            }
        return probs_dict

class AudioEmotionModel(nn.Module):
    def __init__(self, input_features: int = 128, hidden_size: int = 256, num_emotions: int = 7):
        super().__init__()
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.lstm = nn.LSTM(
            hidden_size // 2, 
            hidden_size // 4, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_emotions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, features)
        batch_size, seq_len, _ = x.shape
        
        # Extract features for each timestep
        x_reshaped = x.view(-1, x.size(-1))
        features = self.feature_extractor(x_reshaped)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Use last output for classification
        last_output = lstm_out[:, -1, :]
        logits = self.classifier(last_output)
        
        return torch.softmax(logits, dim=-1)
    
    def predict_emotion(self, features: torch.Tensor) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            probabilities = self.forward(features)
            probs_dict = {
                emotion: float(prob) 
                for emotion, prob in zip(self.emotions, probabilities[0])
            }
        return probs_dict

class MultimodalFusionModel(nn.Module):
    def __init__(self, text_dim: int = 6, audio_dim: int = 7, image_dim: int = 7):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(text_dim + audio_dim + image_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # Final emotion categories
        )
        
        self.emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    
    def forward(self, text_features: torch.Tensor, audio_features: torch.Tensor, 
                image_features: torch.Tensor) -> torch.Tensor:
        # Concatenate all modality features
        combined_features = torch.cat([text_features, audio_features, image_features], dim=1)
        logits = self.fusion_layer(combined_features)
        return torch.softmax(logits, dim=-1)
    
    def predict_multimodal_emotion(self, text_features: torch.Tensor, 
                                 audio_features: torch.Tensor, 
                                 image_features: torch.Tensor) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            probabilities = self.forward(text_features, audio_features, image_features)
            probs_dict = {
                emotion: float(prob) 
                for emotion, prob in zip(self.emotions, probabilities[0])
            }
        return probs_dict
