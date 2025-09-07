#!/usr/bin/env python
import os
import sys
import subprocess

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create directories if they don't exist
for dir_path in ["models", "data", "data/audio"]:
    os.makedirs(dir_path, exist_ok=True)

# Check if models exist
models_exist = (
    os.path.exists("models/text_model.pth") and 
    os.path.exists("models/audio_model.pth")
)

if not models_exist:
    print("Warning: Model files not found. The application will run with limited functionality.")
    print("You might need to train the models first using the notebooks.")

# Start the backend server
try:
    print("Starting backend server...")
    subprocess.run([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
except KeyboardInterrupt:
    print("\nShutting down server...")
except Exception as e:
    print(f"Error starting server: {e}")
