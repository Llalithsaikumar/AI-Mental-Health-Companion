# AI-Mental-Health-Companion
AI-powered companion that detects emotions from text and voice, tracks mood trends, and provides personalized wellness support

# 🧠 AI Mental Health Companion

An AI-powered mental health companion that detects emotions from **text and voice**, tracks user mood trends, and provides personalized wellness guidance using NLP and Speech Emotion Recognition.

---

## ✨ Features
- 📝 **Text Emotion Detection** → Understands emotions from user journaling.
- 🎙️ **Voice Emotion Recognition** → Detects stress/anxiety from voice.
- 📊 **Mood Tracker** → Visualizes mood trends over time.
- 💡 **Personalized Guidance** → Suggests exercises (CBT, breathing, mindfulness).

---

## 🏗️ Project Structure

AI-Mental-Health-Companion/
│── data/ # datasets (links provided)
│── notebooks/ # model training & experiments
│── models/ # trained models
│── backend/ # FastAPI backend
│── frontend/ # React/Flutter app
│── utils/ # helper scripts
│── requirements.txt # dependencies
│── README.md # documentation


---

## 📊 Datasets
- **Text Emotions:** [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)  
- **Speech Emotions:** [RAVDESS](https://zenodo.org/record/1188976), [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad)  

---

## ⚙️ Tech Stack
- **ML/DL:** PyTorch, HuggingFace Transformers
- **Audio:** Librosa, Torchaudio
- **Backend:** FastAPI
- **Frontend:** React/Flutter
- **Database:** MongoDB/Firebase

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/AI-Mental-Health-Companion.git
cd AI-Mental-Health-Companion

pip install -r requirements.txt

jupyter notebook notebooks/Text_Emotion_Demo.ipynb


---

## ✅ Step 4: Requirements File
Create `requirements.txt` so anyone can replicate:  

```txt
torch
torchaudio
transformers
datasets
scikit-learn
matplotlib
seaborn
librosa
fastapi
uvicorn
