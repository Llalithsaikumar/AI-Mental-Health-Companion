import api from './api';

// MongoDB specific API endpoints
const mongodbAPI = {
  // Database Status
  checkStatus: () => api.get('/db/status'),
  
  // User operations
  createUser: (userData) => api.post('/auth/register', userData),
  getUserProfile: () => api.get('/users/me'),
  updateUserProfile: (userData) => api.put('/users/me', userData),
  
  // Mood tracking
  logMood: (moodData) => api.post('/mood/log', moodData),
  getMoodHistory: () => api.get('/mood/history'),
  predictMood: () => api.post('/mood/predict'),
  
  // Crisis assessment
  assessCrisisRisk: (data) => api.post('/crisis/assess', data),
  
  // Interventions
  getInterventions: (params) => api.get('/interventions', { params }),
  getAdvancedIntervention: (data) => api.post('/intervention/advanced', data),
  recordInterventionFeedback: (interventionId, feedbackData) => 
    api.post(`/intervention/${interventionId}/feedback`, feedbackData),
};

export default mongodbAPI;
