"""
Setup script for the Mental Health Companion API
"""
import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("=" * 80)
    print("Installing required dependencies...")
    print("=" * 80)
    
    # List of dependencies
    dependencies = [
        "fastapi",
        "uvicorn",
        "numpy",
        "websockets"
    ]
    
    # Install dependencies
    for dep in dependencies:
        print(f"Installing {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    
    print("\nAll dependencies installed successfully!")

def setup_data_directory():
    """Set up data directory for the database"""
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    
    # Create data directory if it doesn't exist
    if not data_dir.exists():
        print(f"Creating data directory at {data_dir}...")
        data_dir.mkdir(exist_ok=True)

def main():
    """Main setup function"""
    print("=" * 80)
    print("Mental Health Companion API Setup")
    print("=" * 80)
    
    # Install dependencies
    install_dependencies()
    
    # Set up data directory
    setup_data_directory()
    
    print("\nSetup completed successfully!")
    print("\nYou can now run the application using:")
    print("  python run_minimal.py")
    
    # Ask if user wants to run the application now
    run_now = input("\nDo you want to run the application now? (y/n): ").lower()
    if run_now == 'y':
        print("\nStarting application...")
        # Get the script directory
        script_dir = Path(__file__).resolve().parent
        
        # Run the minimal app
        os.chdir(script_dir)
        subprocess.call([sys.executable, "run_minimal.py"])

if __name__ == "__main__":
    main()
