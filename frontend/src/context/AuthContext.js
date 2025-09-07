import React, { createContext, useState, useContext, useEffect } from 'react';
import api from '../services/api';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    // Check if user is already logged in
    const token = localStorage.getItem('token');
    const userId = localStorage.getItem('user_id');
    
    if (token && userId) {
      // Validate token
      api.get('/auth/validate')
        .then(() => {
          setCurrentUser({ id: userId });
        })
        .catch(() => {
          // Token invalid, clean up
          localStorage.removeItem('token');
          localStorage.removeItem('user_id');
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      setLoading(false);
    }
  }, []);

  const login = async (username, password) => {
    try {
      setError('');
      // Create form data as expected by OAuth2PasswordRequestForm
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);
      
      // Use the correct endpoint matching the backend with form data
      const response = await api.post('/auth/token', formData);
      const { access_token, user_id } = response.data;
      
      // Store token and user info
      localStorage.setItem('token', access_token);
      localStorage.setItem('user_id', user_id);
      setCurrentUser({ id: user_id });
      return true;
    } catch (err) {
      console.error('Login error:', err);
      setError(err.response?.data?.detail || 'Login failed. Please check your credentials.');
      return false;
    }
  };

  const register = async (username, password, email, name) => {
    try {
      setError('');
      const response = await api.post('/auth/register', { 
        username, 
        password, 
        email, 
        full_name: name // Changed to match the backend's expected parameter name
      });
      return { success: true, message: response.data.message };
    } catch (err) {
      console.error('Registration error:', err);
      setError(err.response?.data?.detail || 'Registration failed. Please try again.');
      return { success: false, message: err.response?.data?.detail || 'Registration failed' };
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user_id');
    setCurrentUser(null);
  };

  const value = {
    currentUser,
    login,
    register,
    logout,
    error,
    loading
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
