import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  Box,
  Button,
  TextField,
  Typography,
  Container,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Grid,
  Link as MuiLink
} from '@mui/material';
import { motion } from 'framer-motion';
import LoginIcon from '@mui/icons-material/Login';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  
  const { login, error } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMessage('');

    try {
      const success = await login(username, password);
      if (success) {
        navigate('/dashboard');
      } else {
        setErrorMessage(error || 'Login failed. Please check your credentials.');
      }
    } catch (err) {
      setErrorMessage('An unexpected error occurred. Please try again.');
      console.error('Login error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxWidth="sm">
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          py: 4
        }}
      >
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Typography 
            component="h1" 
            variant="h3" 
            color="primary" 
            gutterBottom
            textAlign="center"
            sx={{ fontWeight: 700 }}
          >
            🧠 AI Mental Health Companion
          </Typography>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          style={{ width: '100%' }}
        >
          <Card 
            elevation={8}
            sx={{
              borderRadius: 4,
              overflow: 'hidden',
            }}
          >
            <CardContent sx={{ p: 4 }}>
              <Typography
                variant="h4" 
                component="h2" 
                gutterBottom
                textAlign="center"
                sx={{ mb: 3 }}
              >
                Welcome Back
              </Typography>

              {errorMessage && (
                <Alert severity="error" sx={{ mb: 3 }}>
                  {errorMessage}
                </Alert>
              )}

              <form onSubmit={handleSubmit}>
                <TextField
                  label="Username"
                  variant="outlined"
                  fullWidth
                  margin="normal"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  disabled={isLoading}
                />

                <TextField
                  label="Password"
                  variant="outlined"
                  fullWidth
                  margin="normal"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  disabled={isLoading}
                />

                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  color="primary"
                  size="large"
                  disabled={isLoading}
                  sx={{ mt: 3, mb: 2, py: 1.5 }}
                  startIcon={isLoading ? <CircularProgress size={24} color="inherit" /> : <LoginIcon />}
                >
                  {isLoading ? 'Signing In...' : 'Sign In'}
                </Button>

                <Grid container justifyContent="center">
                  <Grid item>
                    <MuiLink 
                      component={Link} 
                      to="/register" 
                      variant="body2"
                      underline="hover"
                      color="primary.main"
                    >
                      Don't have an account? Sign up
                    </MuiLink>
                  </Grid>
                </Grid>
              </form>
            </CardContent>
          </Card>

          <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 4 }}>
            Your mental wellbeing companion powered by AI
          </Typography>
        </motion.div>
      </Box>
    </Container>
  );
};

export default Login;
