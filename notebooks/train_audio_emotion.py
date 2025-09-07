import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
import os
from typing import List
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils.preprocessing import AudioPreprocessor
from models.emotion_models import AudioEmotionModel

class AudioEmotionDataset(Dataset):
    def __init__(self, audio_paths: List[str], labels: List[str], preprocessor: AudioPreprocessor):
        self.audio_paths = audio_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.encoded_labels[idx]
        
        # Load and preprocess audio
        audio = self.preprocessor.load_audio(audio_path)
        features = self.preprocessor.extract_features(audio)
        
        # Combine all features
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
        
        return {
            'features': torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_audio_emotion_model():
    # Load dataset
    df = pd.read_csv('../data/audio_emotion_dataset.csv')
    
    # Initialize preprocessor and model
    preprocessor = AudioPreprocessor()
    model = AudioEmotionModel(input_features=128)
    
    # Prepare data
    audio_paths = df['audio_path'].tolist()
    emotions = df['emotion'].tolist()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        audio_paths, emotions, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = AudioEmotionDataset(X_train, y_train, preprocessor)
    val_dataset = AudioEmotionDataset(X_val, y_val, preprocessor)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), '../models/audio_model.pth')
            print('Model saved!')

if __name__ == "__main__":
    train_audio_emotion_model()
