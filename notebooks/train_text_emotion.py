import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List
import os
import sys
sys.path.append('..')

from utils.preprocessing import TextPreprocessor
from models.emotion_models import TextEmotionModel

class EmotionDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], preprocessor: TextPreprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.encoded_labels[idx]
        
        tokenized = self.preprocessor.tokenize_text(text)
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_text_emotion_model():
    # Load dataset (replace with your dataset path)
    # Expected format: CSV with 'text' and 'emotion' columns
    df = pd.read_csv('../data/emotion_dataset.csv')
    
    # Initialize preprocessor and model
    preprocessor = TextPreprocessor()
    model = TextEmotionModel()
    
    # Prepare data
    texts = df['text'].tolist()
    emotions = df['emotion'].tolist()
    
    X_train, X_val, y_train, y_val = train_test_split(
        texts, emotions, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = EmotionDataset(X_train, y_train, preprocessor)
    val_dataset = EmotionDataset(X_val, y_val, preprocessor)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 5
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
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
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
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
            torch.save(model.state_dict(), '../models/text_model.pth')
            print('Model saved!')
        
        print('-' * 50)

if __name__ == "__main__":
    train_text_emotion_model()
