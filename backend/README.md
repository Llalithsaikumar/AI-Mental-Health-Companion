# AI Mental Health Companion API

This is the backend API for the AI Mental Health Companion application.

## Features

- User authentication with JWT
- MongoDB integration with SQLite fallback
- Mood logging and tracking
- Personalized intervention suggestions
- Predictive mood analysis
- Real-time monitoring via WebSockets
- Crisis risk assessment

## Getting Started

### 1. Install Dependencies

Install all required dependencies:

```bash
pip install -r ../requirements.txt
```

For MongoDB support:
```bash
pip install pymongo
```

### 2. Configure MongoDB (Optional)

Create a `.env` file in the backend directory:

```
MONGODB_CONNECTION_STRING=mongodb://username:password@hostname:port/database
MONGODB_DATABASE=mental_health_companion
SECRET_KEY=your_secret_key_for_jwt_tokens
```

If MongoDB is not configured, the system will automatically use SQLite as a fallback.

### 3. Initialize SQLite Database (for SQLite fallback)

```bash
python create_sqlite_db.py
```

To add sample data:
```bash
python create_sqlite_db.py --sample-data
```

### 4. Run the Application

You can run the application using:

```bash
python run.py
```

Available options:
- `python run.py --minimal`: Run the minimal version with fewer dependencies
- `python run.py --mongodb`: Run the MongoDB-enabled version

The server will start at `http://localhost:8000`

### 5. API Documentation

Once the server is running, you can access the API documentation at:

```
http://localhost:8000/docs
```

## API Endpoints

### Authentication

- `POST /api/auth/login` - Login with username and password

### Mood Tracking

- `POST /api/mood/log` - Log a new mood entry
- `GET /api/mood/history` - Get mood history
- `POST /api/mood/predict` - Predict future mood based on patterns

### Interventions

- `GET /api/interventions` - Get intervention suggestions
- `POST /api/intervention/advanced` - Get advanced personalized interventions

### Crisis Assessment

- `POST /api/crisis/assess` - Assess crisis risk

### WebSockets

- `WebSocket /ws/{user_id}` - Real-time mood monitoring

## Database

The application uses SQLite for data storage. The database file is created at:

```
data/mental_health.db
```

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed
2. Check that the data directory exists
3. Ensure you have appropriate permissions to create and write to files
4. Check the console output for error messages

## Notes

This is a minimal version with many features using mock implementations. For the full version with actual machine learning models, use the main application.
