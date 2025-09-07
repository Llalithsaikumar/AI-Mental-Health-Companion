import React, { useState, useEffect } from 'react';
import { Box, Container, Typography, Paper, TextField, Button, Avatar, Grid, Divider } from '@mui/material';
import { styled } from '@mui/material/styles';
import DashboardHeader from '../components/DashboardHeader';
import { userAPI } from '../services/api';

const ProfileContainer = styled(Container)(({ theme }) => ({
  marginTop: theme.spacing(4),
  marginBottom: theme.spacing(4),
}));

const ProfilePaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  borderRadius: '2px',
  border: '1px solid #e0e0e0',
  boxShadow: '0 2px 6px rgba(0, 0, 0, 0.03)',
}));

const Profile = () => {
  const [profile, setProfile] = useState({
    fullName: '',
    email: '',
    username: '',
    bio: ''
  });
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        setLoading(true);
        // In a real app, you would fetch from API
        // const response = await userAPI.getProfile();
        // setProfile(response.data);
        
        // Mock data for now
        setTimeout(() => {
          setProfile({
            fullName: 'John Doe',
            email: 'john.doe@example.com',
            username: 'johndoe',
            bio: 'Mental health enthusiast and wellbeing advocate.'
          });
          setLoading(false);
        }, 500);
      } catch (error) {
        console.error('Error fetching profile:', error);
        setError('Failed to load profile information');
        setLoading(false);
      }
    };
    
    fetchProfile();
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setProfile(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess(false);
    
    try {
      // In a real app, you would update via API
      // await userAPI.updateProfile(profile);
      
      // Mock update
      setTimeout(() => {
        setSuccess(true);
        setLoading(false);
      }, 500);
    } catch (error) {
      console.error('Error updating profile:', error);
      setError('Failed to update profile information');
      setLoading(false);
    }
  };

  return (
    <>
      <DashboardHeader title="Profile" />
      <ProfileContainer maxWidth="md">
        <Typography variant="h4" component="h1" gutterBottom className="title-minimalist">
          Your Profile
        </Typography>
        <Typography variant="body1" paragraph className="text-minimalist">
          Manage your personal information and account details
        </Typography>
        
        <ProfilePaper elevation={0} className="card-minimalist">
          <Grid container spacing={4}>
            <Grid item xs={12} md={3} sx={{ textAlign: 'center' }}>
              <Avatar 
                sx={{ 
                  width: 120, 
                  height: 120, 
                  margin: '0 auto 16px',
                  backgroundColor: '#1a1a1a' 
                }}
              >
                {profile.fullName?.charAt(0) || 'U'}
              </Avatar>
              <Typography variant="body2" color="textSecondary">
                Profile Picture
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={9}>
              <Box component="form" onSubmit={handleSubmit} noValidate>
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom className="title-minimalist">
                      Personal Information
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Full Name"
                      name="fullName"
                      value={profile.fullName}
                      onChange={handleChange}
                      variant="outlined"
                      className="input-minimalist"
                    />
                  </Grid>
                  
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Username"
                      name="username"
                      value={profile.username}
                      onChange={handleChange}
                      variant="outlined"
                      disabled
                    />
                  </Grid>
                  
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Email"
                      name="email"
                      type="email"
                      value={profile.email}
                      onChange={handleChange}
                      variant="outlined"
                    />
                  </Grid>
                  
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Bio"
                      name="bio"
                      value={profile.bio}
                      onChange={handleChange}
                      multiline
                      rows={4}
                      variant="outlined"
                    />
                  </Grid>
                  
                  {error && (
                    <Grid item xs={12}>
                      <Typography color="error">{error}</Typography>
                    </Grid>
                  )}
                  
                  {success && (
                    <Grid item xs={12}>
                      <Typography color="success.main">
                        Profile updated successfully!
                      </Typography>
                    </Grid>
                  )}
                  
                  <Grid item xs={12}>
                    <Button
                      type="submit"
                      variant="contained"
                      color="primary"
                      disabled={loading}
                      className="btn-minimalist"
                      sx={{ mt: 2 }}
                    >
                      {loading ? 'Updating...' : 'Save Changes'}
                    </Button>
                  </Grid>
                </Grid>
              </Box>
            </Grid>
          </Grid>
        </ProfilePaper>
      </ProfileContainer>
    </>
  );
};

export default Profile;
