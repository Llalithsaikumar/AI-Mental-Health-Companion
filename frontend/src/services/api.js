import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Request interceptor for adding the auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for handling common errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle 401 Unauthorized errors
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user_id');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  login: (credentials) => api.post('/auth/login', credentials),
  register: (userData) => api.post('/auth/register', userData),
  validate: () => api.get('/auth/validate'),
};

// Mood API
export const moodAPI = {
  logMood: (data) => api.post('/mood', data),
  getMoodHistory: () => api.get('/mood/history'),
  predictMood: () => api.get('/predict/mood'),
  getMoodStats: (days = 30) => api.get(`/mood/stats?days=${days}`),
};

// Helper functions for working with the API
export const logMood = async (moodData) => {
  try {
    const response = await moodAPI.logMood(moodData);
    return response.data;
  } catch (error) {
    console.error('Error logging mood:', error);
    throw error;
  }
};

export const getModdHistory = async () => {
  try {
    const response = await moodAPI.getMoodHistory();
    return response.data;
  } catch (error) {
    console.error('Error fetching mood history:', error);
    throw error;
  }
};

export const getMoodStats = async (days = 30) => {
  try {
    const response = await moodAPI.getMoodStats(days);
    return response.data;
  } catch (error) {
    console.error('Error fetching mood stats:', error);
    throw error;
  }
};

// Interventions API
export const interventionsAPI = {
  getInterventions: (params) => api.get('/interventions', { params }),
  getAdvancedInterventions: (data) => api.post('/intervention/advanced', data),
  provideFeedback: (data) => api.post('/intervention/feedback', data),
};

// Crisis Assessment API
export const crisisAPI = {
  assessRisk: (data) => api.post('/crisis/assess', data),
};

// User API
export const userAPI = {
  getProfile: () => api.get('/user/profile'),
  updateProfile: (data) => api.put('/user/profile', data),
  updatePreferences: (data) => api.put('/user/preferences', data),
};

export default api;
