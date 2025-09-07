"""
Run script for Mental Health Companion API with MongoDB support
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Check MongoDB connection
try:
    from backend.mongodb_handler import db_handler
    
    if db_handler.client is None:
        print("⚠️ Warning: MongoDB connection not available.")
        print("Make sure MongoDB is running and connection string is correct.")
        print("You can set the connection string via MONGODB_CONNECTION_STRING environment variable.")
        print("Falling back to SQLite database.")
    else:
        print(f"✅ Connected to MongoDB database: {db_handler.db_name}")
        stats = db_handler.get_collection_stats()
        print("Collection stats:", stats)
except ImportError:
    print("⚠️ Warning: MongoDB handler not found.")
    print("Make sure pymongo is installed: pip install pymongo")
    print("Falling back to SQLite database.")

if __name__ == "__main__":
    print("Starting Mental Health Companion API")
    print("Documentation available at: http://localhost:8000/docs")
    
    # Determine which app to run
    if len(sys.argv) > 1 and sys.argv[1] == "--minimal":
        # Run minimal app
        print("Starting minimal app...")
        from backend.app_minimal import app
    elif len(sys.argv) > 1 and sys.argv[1] == "--mongodb":
        # Run MongoDB-enabled app
        print("Starting MongoDB-enabled app...")
        from backend.app_mongodb import app
    else:
        # Run full app by default
        print("Starting full app...")
        try:
            from backend.app import app
        except ImportError:
            print("Full app not available. Falling back to MongoDB-enabled app.")
            try:
                from backend.app_mongodb import app
            except ImportError:
                print("MongoDB app not available. Falling back to minimal app.")
                from backend.app_minimal import app
    
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
