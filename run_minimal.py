"""
Simple startup script for the minimal version of the Mental Health Companion API
"""
import os
import sys
from pathlib import Path

def main():
    """Run the minimal API server"""
    # Print welcome message
    print("=" * 80)
    print("Starting Mental Health Companion API (Minimal Version)")
    print("=" * 80)
    
    # Ensure working directory is correct
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    
    # Ensure data directory exists
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Add the backend directory to the Python path if it's not already there
    backend_dir = script_dir / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.append(str(backend_dir))
    
    # Import and run the minimal API
    try:
        # Try to import the module
        import backend.app_minimal as minimal_app
        
        print("\nStarting server at http://localhost:8000")
        print("API documentation will be available at http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the server")
        
        # Run the app
        import uvicorn
        uvicorn.run("backend.app_minimal:app", host="0.0.0.0", port=8000)
    
    except ImportError as e:
        print(f"Error importing app_minimal: {e}")
        print("\nMake sure you have all required dependencies installed:")
        print("pip install fastapi uvicorn numpy")
        sys.exit(1)

if __name__ == "__main__":
    main()
