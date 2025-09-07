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
  Link as MuiLink,
  Stepper,
  Step,
  StepLabel
} from '@mui/material';
import { motion } from 'framer-motion';
import PersonAddIcon from '@mui/icons-material/PersonAdd';

const Register = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  
  const { register } = useAuth();
  const navigate = useNavigate();

  const steps = ['Account Information', 'Personal Details'];

  const validateFirstStep = () => {
    if (password !== confirmPassword) {
      setErrorMessage('Passwords do not match');
      return false;
    }
    if (password.length < 8) {
      setErrorMessage('Password must be at least 8 characters long');
      return false;
    }
    return true;
  };

  const handleNext = () => {
    if (activeStep === 0 && !validateFirstStep()) {
      return;
    }
    setErrorMessage('');
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setErrorMessage('');
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMessage('');

    try {
      const result = await register(username, password, email, name);
      if (result.success) {
        setSuccessMessage('Registration successful! You will be redirected to login...');
        setTimeout(() => {
          navigate('/login');
        }, 2000);
      } else {
        setErrorMessage(result.message || 'Registration failed. Please try again.');
      }
    } catch (err) {
      setErrorMessage('An unexpected error occurred. Please try again.');
      console.error('Registration error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const renderStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <>
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
              helperText="Password must be at least 8 characters long"
            />
            <TextField
              label="Confirm Password"
              variant="outlined"
              fullWidth
              margin="normal"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              disabled={isLoading}
              error={password !== confirmPassword}
              helperText={password !== confirmPassword ? "Passwords don't match" : ""}
            />
          </>
        );
      case 1:
        return (
          <>
            <TextField
              label="Full Name"
              variant="outlined"
              fullWidth
              margin="normal"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              disabled={isLoading}
            />
            <TextField
              label="Email"
              variant="outlined"
              fullWidth
              margin="normal"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              disabled={isLoading}
            />
          </>
        );
      default:
        return 'Unknown step';
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
                Create Your Account
              </Typography>

              <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
                {steps.map((label) => (
                  <Step key={label}>
                    <StepLabel>{label}</StepLabel>
                  </Step>
                ))}
              </Stepper>

              {successMessage && (
                <Alert severity="success" sx={{ mb: 3 }}>
                  {successMessage}
                </Alert>
              )}

              {errorMessage && (
                <Alert severity="error" sx={{ mb: 3 }}>
                  {errorMessage}
                </Alert>
              )}

              <form onSubmit={activeStep === steps.length - 1 ? handleSubmit : handleNext}>
                {renderStepContent(activeStep)}

                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
                  <Button
                    disabled={activeStep === 0 || isLoading}
                    onClick={handleBack}
                    variant="outlined"
                  >
                    Back
                  </Button>
                  
                  <Button
                    variant="contained"
                    color="primary"
                    type={activeStep === steps.length - 1 ? "submit" : "button"}
                    onClick={activeStep === steps.length - 1 ? undefined : handleNext}
                    disabled={isLoading}
                    startIcon={activeStep === steps.length - 1 ? (isLoading ? <CircularProgress size={24} color="inherit" /> : <PersonAddIcon />) : null}
                  >
                    {activeStep === steps.length - 1 ? (isLoading ? 'Creating Account...' : 'Create Account') : 'Next'}
                  </Button>
                </Box>
              </form>

              <Grid container justifyContent="center" sx={{ mt: 3 }}>
                <Grid item>
                  <MuiLink 
                    component={Link} 
                    to="/login" 
                    variant="body2"
                    underline="hover"
                    color="primary.main"
                  >
                    Already have an account? Sign in
                  </MuiLink>
                </Grid>
              </Grid>
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

export default Register;
