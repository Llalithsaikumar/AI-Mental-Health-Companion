
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
	git clone https://github.com/<your-username>/AI-Mental-Health-Companion.git
	cd AI-Mental-Health-Companion
	```

2. **Install dependencies**
	```bash
	pip install -r requirements.txt
	```

3. **Run notebook demo**
	```bash
	jupyter notebook notebooks/Text_Emotion_Demo.ipynb
	```

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