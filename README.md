# 🧠 AI Mental Health Companion

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![MIT License](https://img.shields.io/badge/License-MIT-green.svg)

An AI-powered mental health companion that detects emotions from **text and voice**, tracks user mood trends, and provides personalized wellness guidance using NLP and Speech Emotion Recognition.

---

## ✨ Features

- 📝 **Text Emotion Detection**: Understands emotions from user journaling.
- 🎙️ **Voice Emotion Recognition**: Detects stress, anxiety, and other emotions from speech.
- 📊 **Mood Tracker**: Visualizes mood trends over time.
- 💡 **Personalized Guidance**: Suggests exercises (CBT, breathing, mindfulness).

---

## 🏗️ Project Structure

```bash
AI-Mental-Health-Companion/
│── data/             # datasets (links below)
│── notebooks/        # model training & experiments
│── models/           # trained models
│── backend/          # FastAPI backend
│── frontend/         # React/Flutter app
│── utils/            # helper scripts
│── requirements.txt  # dependencies
│── README.md         # documentation
```

---

## 📊 Datasets

- **Text Emotions:** [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- **Speech Emotions:** [RAVDESS](https://zenodo.org/record/1188976), [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad)

---

## ⚙️ Tech Stack

- **ML/DL:** PyTorch, HuggingFace Transformers
- **Audio Processing:** Librosa, Torchaudio
- **Backend:** FastAPI
- **Frontend:** React / Flutter
- **Database:** MongoDB / Firebase

---

## 🚀 Getting Started

1. **Clone the repository**
    ```bash
    git clone https://github.com/Llalithsaikumar/AI-Mental-Health-Companion.git
    cd AI-Mental-Health-Companion
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment**
    ```bash
    # Create and activate virtual environment
    python -m venv mental_health_env
    # Activate environment - Windows
    mental_health_env\Scripts\activate
    # Activate environment - macOS/Linux
    source mental_health_env/bin/activate
    ```

4. **Run the application**
    ```bash
    # For full version (requires models)
    python start_backend.py
    
    # For minimal demo version (no models required)
    python start_minimal.py
    ```

5. **Run frontend** (in another terminal)
    ```bash
    cd frontend
    npm install
    npm start
    ```

---

## 💻 Usage Options

### Full Version
- Requires trained models
- Complete functionality
- Start with `python start_backend.py`

### Minimal Demo Version
- No models required
- Mock data for demonstration
- Start with `python start_minimal.py`

---

## 📌 Future Improvements

- 🧩 Multi-lingual emotion detection
- 📱 Mobile-first Flutter application
- 🔒 Enhanced privacy & encryption for user data
- 🤝 Integration with mental health APIs (journaling, therapy chatbots)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License – see the `LICENSE` file for details.
