# Deployment Guide

## Prerequisites
- Python 3.8+
- Node.js 16+
- Docker (optional)

## Local Development Setup

### Backend
cd backend
pip install -r ../requirements.txt
python app.py


### Frontend
cd frontend
npm install
npm start


## Production Deployment

### Docker Deployment
Build backend image
docker build -t mental-health-backend ./backend

Build frontend image
docker build -t mental-health-frontend ./frontend

Run with docker-compose
docker-compose up -d


### Cloud Deployment (AWS)
1. Deploy backend on AWS ECS or Lambda
2. Host frontend on AWS S3 + CloudFront
3. Use AWS RDS for database
4. Set up AWS Cognito for authentication

### Environment Variables
SECRET_KEY=your-secret-key
DATABASE_URL=your-database-url
CORS_ORIGINS=your-frontend-domains


## Security Considerations
- Use HTTPS in production
- Set up proper CORS policies
- Configure rate limiting
- Enable request logging
- Set up monitoring and alerting

## Scaling
- Use load balancers for high availability
- Implement caching for frequently accessed data
- Consider microservices architecture for large-scale deployment
