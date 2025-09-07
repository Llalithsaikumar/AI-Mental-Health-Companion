"""
MongoDB Configuration Handler for AI Mental Health Companion
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

class MongoDBConfig:
    """MongoDB configuration handler"""
    
    def __init__(self):
        """Initialize MongoDB configuration"""
        # Default configuration
        self.config = {
            "connection_string": os.environ.get("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/mental_health_companion"),
            "database_name": os.environ.get("MONGODB_DATABASE", "mental_health_companion"),
            "use_mongodb": True if os.environ.get("MONGODB_CONNECTION_STRING") else False
        }
        
        # Ensure config directory exists
        self.config_dir = Path("../config")
        self.config_dir.mkdir(exist_ok=True)
        
        self.config_file = self.config_dir / "mongodb_config.json"
        
        # Load configuration from file if it exists
        self._load_config()
        
    def _load_config(self):
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
            except Exception as e:
                print(f"Error loading MongoDB configuration: {e}")
    
    def save_config(self, connection_string=None, database_name=None):
        """Save configuration to file"""
        if connection_string:
            self.config["connection_string"] = connection_string
        
        if database_name:
            self.config["database_name"] = database_name
        
        # Set use_mongodb flag
        self.config["use_mongodb"] = bool(self.config["connection_string"])
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving MongoDB configuration: {e}")
            return False
    
    def get_connection_string(self):
        """Get MongoDB connection string"""
        return self.config["connection_string"]
    
    def get_database_name(self):
        """Get MongoDB database name"""
        return self.config["database_name"]
    
    def is_mongodb_enabled(self):
        """Check if MongoDB is enabled"""
        return self.config["use_mongodb"]
    
    def test_connection(self):
        """Test MongoDB connection"""
        if not self.config["use_mongodb"]:
            return False
            
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
            
            client = MongoClient(
                self.config["connection_string"], 
                serverSelectionTimeoutMS=5000
            )
            
            # Test connection
            client.admin.command('ping')
            
            # Get database name from connection string if not specified
            db_name = self.config["database_name"]
            if not db_name:
                db_name = self.config["connection_string"].split("/")[-1]
                if "?" in db_name:
                    db_name = db_name.split("?")[0]
            
            # Test database access
            db = client[db_name]
            db.list_collection_names()
            
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"MongoDB connection test failed: {e}")
            return False
        except ImportError:
            print("pymongo not installed")
            return False

# Create a singleton instance
mongodb_config = MongoDBConfig()
