import React, { useState, useEffect } from 'react';
import { 
  Box, Container, Typography, Paper, Switch, 
  FormControlLabel, Divider, Button, Grid, 
  RadioGroup, Radio, FormControl, FormLabel 
} from '@mui/material';
import { styled } from '@mui/material/styles';
import DashboardHeader from '../components/DashboardHeader';

const SettingsContainer = styled(Container)(({ theme }) => ({
  marginTop: theme.spacing(4),
  marginBottom: theme.spacing(4),
}));

const SettingsPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  marginBottom: theme.spacing(3),
  borderRadius: '2px',
  border: '1px solid #e0e0e0',
  boxShadow: '0 2px 6px rgba(0, 0, 0, 0.03)',
}));

const Settings = () => {
  const [settings, setSettings] = useState({
    notifications: {
      email: true,
      app: true,
      reminders: true,
      progressSummaries: false
    },
    privacy: {
      shareData: 'anonymized',
      allowResearch: true,
      dataDeletion: 'archive'
    },
    preferences: {
      theme: 'light',
      language: 'english',
      fontSize: 'medium',
      autoSave: true
    }
  });
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleSwitchChange = (category, setting) => (event) => {
    setSettings({
      ...settings,
      [category]: {
        ...settings[category],
        [setting]: event.target.checked
      }
    });
  };

  const handleRadioChange = (category, setting) => (event) => {
    setSettings({
      ...settings,
      [category]: {
        ...settings[category],
        [setting]: event.target.value
      }
    });
  };

  const handleSaveSettings = async () => {
    setLoading(true);
    // In a real app, you would save to API
    // await settingsAPI.updateSettings(settings);
    
    // Mock saving
    setTimeout(() => {
      setSuccess(true);
      setLoading(false);
      
      // Reset success message after 3 seconds
      setTimeout(() => setSuccess(false), 3000);
    }, 800);
  };

  return (
    <>
      <DashboardHeader title="Settings" />
      <SettingsContainer maxWidth="md">
        <Typography variant="h4" component="h1" gutterBottom className="title-minimalist">
          Settings
        </Typography>
        <Typography variant="body1" paragraph className="text-minimalist">
          Customize your application preferences and notification settings
        </Typography>
        
        {/* Notification Settings */}
        <SettingsPaper elevation={0} className="card-minimalist">
          <Typography variant="h5" gutterBottom className="title-minimalist">
            Notifications
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            Control when and how you receive notifications
          </Typography>
          
          <Box sx={{ mt: 3 }}>
            <FormControlLabel
              control={
                <Switch 
                  checked={settings.notifications.email}
                  onChange={handleSwitchChange('notifications', 'email')}
                />
              }
              label="Email Notifications"
            />
            <Typography variant="body2" color="textSecondary" sx={{ ml: 4, mb: 2 }}>
              Receive updates and reminders via email
            </Typography>
            
            <FormControlLabel
              control={
                <Switch 
                  checked={settings.notifications.app}
                  onChange={handleSwitchChange('notifications', 'app')}
                />
              }
              label="App Notifications"
            />
            <Typography variant="body2" color="textSecondary" sx={{ ml: 4, mb: 2 }}>
              Receive in-app notifications and alerts
            </Typography>
            
            <FormControlLabel
              control={
                <Switch 
                  checked={settings.notifications.reminders}
                  onChange={handleSwitchChange('notifications', 'reminders')}
                />
              }
              label="Daily Reminders"
            />
            <Typography variant="body2" color="textSecondary" sx={{ ml: 4, mb: 2 }}>
              Get daily reminders to log your mood
            </Typography>
            
            <FormControlLabel
              control={
                <Switch 
                  checked={settings.notifications.progressSummaries}
                  onChange={handleSwitchChange('notifications', 'progressSummaries')}
                />
              }
              label="Weekly Progress Summaries"
            />
            <Typography variant="body2" color="textSecondary" sx={{ ml: 4, mb: 2 }}>
              Receive weekly reports on your mental health journey
            </Typography>
          </Box>
        </SettingsPaper>
        
        {/* Privacy Settings */}
        <SettingsPaper elevation={0} className="card-minimalist">
          <Typography variant="h5" gutterBottom className="title-minimalist">
            Privacy & Data
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            Manage how your data is stored and used
          </Typography>
          
          <Box sx={{ mt: 3 }}>
            <FormControl component="fieldset" sx={{ mb: 3 }}>
              <FormLabel component="legend">Data Sharing</FormLabel>
              <RadioGroup
                value={settings.privacy.shareData}
                onChange={handleRadioChange('privacy', 'shareData')}
              >
                <FormControlLabel value="none" control={<Radio />} label="Don't share my data" />
                <FormControlLabel value="anonymized" control={<Radio />} label="Share anonymized data only" />
                <FormControlLabel value="all" control={<Radio />} label="Share all data for better recommendations" />
              </RadioGroup>
            </FormControl>
            
            <FormControlLabel
              control={
                <Switch 
                  checked={settings.privacy.allowResearch}
                  onChange={handleSwitchChange('privacy', 'allowResearch')}
                />
              }
              label="Allow anonymized data to be used for research"
            />
            <Typography variant="body2" color="textSecondary" sx={{ ml: 4, mb: 3 }}>
              Your data can help improve mental health understanding and treatments
            </Typography>
            
            <FormControl component="fieldset">
              <FormLabel component="legend">Data Deletion Policy</FormLabel>
              <RadioGroup
                value={settings.privacy.dataDeletion}
                onChange={handleRadioChange('privacy', 'dataDeletion')}
              >
                <FormControlLabel value="immediate" control={<Radio />} label="Delete data immediately on request" />
                <FormControlLabel value="archive" control={<Radio />} label="Archive data for 30 days before deletion" />
              </RadioGroup>
            </FormControl>
          </Box>
        </SettingsPaper>
        
        {/* App Preferences */}
        <SettingsPaper elevation={0} className="card-minimalist">
          <Typography variant="h5" gutterBottom className="title-minimalist">
            App Preferences
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            Customize your experience
          </Typography>
          
          <Box sx={{ mt: 3 }}>
            <FormControl component="fieldset" sx={{ mb: 3 }}>
              <FormLabel component="legend">Theme</FormLabel>
              <RadioGroup
                value={settings.preferences.theme}
                onChange={handleRadioChange('preferences', 'theme')}
              >
                <FormControlLabel value="light" control={<Radio />} label="Light" />
                <FormControlLabel value="dark" control={<Radio />} label="Dark" />
                <FormControlLabel value="system" control={<Radio />} label="Use system setting" />
              </RadioGroup>
            </FormControl>
            
            <FormControl component="fieldset" sx={{ mb: 3 }}>
              <FormLabel component="legend">Language</FormLabel>
              <RadioGroup
                value={settings.preferences.language}
                onChange={handleRadioChange('preferences', 'language')}
              >
                <FormControlLabel value="english" control={<Radio />} label="English" />
                <FormControlLabel value="spanish" control={<Radio />} label="Spanish" />
                <FormControlLabel value="french" control={<Radio />} label="French" />
              </RadioGroup>
            </FormControl>
            
            <FormControlLabel
              control={
                <Switch 
                  checked={settings.preferences.autoSave}
                  onChange={handleSwitchChange('preferences', 'autoSave')}
                />
              }
              label="Auto-save journal entries"
            />
          </Box>
        </SettingsPaper>
        
        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
          {success && (
            <Typography color="success.main" sx={{ mr: 2, alignSelf: 'center' }}>
              Settings saved successfully!
            </Typography>
          )}
          <Button
            variant="contained"
            color="primary"
            onClick={handleSaveSettings}
            disabled={loading}
            className="btn-minimalist"
          >
            {loading ? 'Saving...' : 'Save Settings'}
          </Button>
        </Box>
      </SettingsContainer>
    </>
  );
};

export default Settings;
