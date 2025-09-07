"""
MongoDB database handler for the Mental Health Companion API
"""
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv
from datetime import datetime
import logging

# Load environment variables from .env file if it exists
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mongodb_handler")

class MongoDBHandler:
    """MongoDB database handler for the application"""
    
    def __init__(self):
        """Initialize the MongoDB connection"""
        self.client = None
        self.db = None
        self.connection_string = os.environ.get('MONGODB_CONNECTION_STRING')
        self.db_name = os.environ.get('MONGODB_DATABASE', 'mental_health_companion')
        
        if not self.connection_string:
            logger.warning("MongoDB connection string not found in environment variables.")
            self.connection_string = "mongodb://localhost:27017/"
            logger.info(f"Using default connection string: {self.connection_string}")
            
        self.connect()
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Verify connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            logger.info(f"Connected to MongoDB: {self.db_name}")
            
            # Create indexes for common queries
            self._create_indexes()
            return True
        
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def _create_indexes(self):
        """Create indexes for collections"""
        try:
            # User collection indexes
            self.db.users.create_index("username", unique=True)
            self.db.users.create_index("email", unique=True)
            
            # Mood entries indexes
            self.db.mood_entries.create_index([("user_id", 1), ("timestamp", -1)])
            
            # Interventions indexes
            self.db.interventions.create_index("user_id")
            self.db.interventions.create_index("type")
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")
    
    def close(self):
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    # User operations
    def create_user(self, user_data):
        """Create a new user"""
        try:
            user_data["created_at"] = datetime.now()
            result = self.db.users.insert_one(user_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    def get_user_by_username(self, username):
        """Get user by username"""
        return self.db.users.find_one({"username": username})
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        return self.db.users.find_one({"_id": user_id})
    
    def update_user(self, user_id, update_data):
        """Update user data"""
        update_data["updated_at"] = datetime.now()
        result = self.db.users.update_one(
            {"_id": user_id}, 
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    # Mood entry operations
    def create_mood_entry(self, mood_data):
        """Create a new mood entry"""
        mood_data["created_at"] = datetime.now()
        result = self.db.mood_entries.insert_one(mood_data)
        return str(result.inserted_id)
    
    def get_mood_history(self, user_id, limit=30):
        """Get mood history for a user"""
        cursor = self.db.mood_entries.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit)
        
        return list(cursor)
    
    def get_mood_stats(self, user_id, days=30):
        """Get mood statistics for a user"""
        # Get data for the specified time period
        from_date = datetime.now() - datetime.timedelta(days=days)
        
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "timestamp": {"$gte": from_date}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_mood": {"$avg": "$mood_score"},
                    "min_mood": {"$min": "$mood_score"},
                    "max_mood": {"$max": "$mood_score"},
                    "count": {"$sum": 1},
                    "emotions": {"$push": "$text_emotion"}
                }
            }
        ]
        
        result = list(self.db.mood_entries.aggregate(pipeline))
        
        if result:
            # Count emotion frequency
            emotions = result[0].get("emotions", [])
            emotion_counts = {}
            for emotion in emotions:
                if emotion:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
            result[0]["emotion_counts"] = emotion_counts
            return result[0]
        
        return {
            "avg_mood": 0,
            "min_mood": 0,
            "max_mood": 0,
            "count": 0,
            "emotion_counts": {}
        }
    
    # Intervention operations
    def create_intervention(self, intervention_data):
        """Create an intervention record"""
        intervention_data["created_at"] = datetime.now()
        result = self.db.interventions.insert_one(intervention_data)
        return str(result.inserted_id)
    
    def get_user_interventions(self, user_id, limit=10):
        """Get interventions for a user"""
        cursor = self.db.interventions.find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(limit)
        
        return list(cursor)
    
    def update_intervention_feedback(self, intervention_id, feedback_data):
        """Update intervention with feedback"""
        feedback_data["feedback_at"] = datetime.now()
        result = self.db.interventions.update_one(
            {"_id": intervention_id},
            {"$set": {"feedback": feedback_data}}
        )
        return result.modified_count > 0
    
    # Utility methods
    def get_collection_stats(self):
        """Get stats for collections"""
        stats = {}
        for collection_name in ["users", "mood_entries", "interventions"]:
            stats[collection_name] = self.db[collection_name].count_documents({})
        return stats

# Singleton instance
db_handler = MongoDBHandler()
