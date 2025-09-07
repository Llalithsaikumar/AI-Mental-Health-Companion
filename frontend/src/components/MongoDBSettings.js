import React, { useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  Alert,
  Divider,
  InputAdornment,
  IconButton,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import StorageIcon from '@mui/icons-material/Storage';
import api from '../services/api';

const MongoDBSettings = () => {
  const [connectionString, setConnectionString] = useState('');
  const [databaseName, setDatabaseName] = useState('mental_health_companion');
  const [showConnectionString, setShowConnectionString] = useState(false);
  const [loading, setLoading] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [openDialog, setOpenDialog] = useState(false);
  
  const handleTestConnection = async () => {
    setLoading(true);
    setTestResult(null);
    
    try {
      const response = await api.post('/db/test-connection', {
        connection_string: connectionString,
        database_name: databaseName
      });
      
      setTestResult({
        success: true,
        message: 'Successfully connected to MongoDB!'
      });
    } catch (error) {
      setTestResult({
        success: false,
        message: error.response?.data?.detail || 'Failed to connect to MongoDB. Check your connection string and try again.'
      });
    } finally {
      setLoading(false);
    }
  };
  
  const handleSaveSettings = async () => {
    setLoading(true);
    try {
      await api.post('/db/configure', {
        connection_string: connectionString,
        database_name: databaseName
      });
      
      setTestResult({
        success: true,
        message: 'MongoDB settings saved successfully. The application will use these settings on next restart.'
      });
      setOpenDialog(false);
    } catch (error) {
      setTestResult({
        success: false,
        message: error.response?.data?.detail || 'Failed to save MongoDB settings.'
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Paper elevation={2} sx={{ p: 3, borderRadius: 2 }}>
        <Box display="flex" alignItems="center" mb={2}>
          <StorageIcon fontSize="large" sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" fontWeight="600">
            MongoDB Configuration
          </Typography>
        </Box>
        
        <Typography variant="body2" color="text.secondary" mb={3}>
          Configure your MongoDB connection to enable advanced features like user authentication, 
          mood tracking history, and personalized interventions.
        </Typography>
        
        <Button 
          variant="outlined" 
          startIcon={<SettingsIcon />}
          onClick={() => setOpenDialog(true)}
          sx={{ mb: 2 }}
        >
          Configure MongoDB Connection
        </Button>
        
        {testResult && (
          <Alert 
            severity={testResult.success ? "success" : "error"}
            sx={{ mt: 2 }}
          >
            {testResult.message}
          </Alert>
        )}
      </Paper>
      
      <Dialog 
        open={openDialog} 
        onClose={() => setOpenDialog(false)}
        fullWidth
        maxWidth="sm"
      >
        <DialogTitle>
          Configure MongoDB Connection
        </DialogTitle>
        
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <Typography variant="body2" color="text.secondary" mb={2}>
              Enter your MongoDB connection string and database name to connect to your database.
            </Typography>
            
            <TextField
              label="MongoDB Connection String"
              fullWidth
              margin="normal"
              value={connectionString}
              onChange={(e) => setConnectionString(e.target.value)}
              placeholder="mongodb://username:password@hostname:port/database"
              type={showConnectionString ? "text" : "password"}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowConnectionString(!showConnectionString)}
                      edge="end"
                    >
                      {showConnectionString ? <VisibilityOffIcon /> : <VisibilityIcon />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
            
            <TextField
              label="Database Name"
              fullWidth
              margin="normal"
              value={databaseName}
              onChange={(e) => setDatabaseName(e.target.value)}
              placeholder="mental_health_companion"
              helperText="Leave as default if you're using the connection string with database name"
            />
            
            <Box mt={2}>
              <Button 
                variant="outlined" 
                onClick={handleTestConnection}
                disabled={loading || !connectionString}
                sx={{ mr: 2 }}
              >
                {loading ? <CircularProgress size={20} sx={{ mr: 1 }} /> : "Test Connection"}
              </Button>
            </Box>
            
            {testResult && (
              <Alert 
                severity={testResult.success ? "success" : "error"}
                sx={{ mt: 2 }}
              >
                {testResult.message}
              </Alert>
            )}
          </Box>
        </DialogContent>
        
        <DialogActions sx={{ px: 3, pb: 3 }}>
          <Button onClick={() => setOpenDialog(false)}>
            Cancel
          </Button>
          <Button 
            variant="contained" 
            onClick={handleSaveSettings}
            disabled={loading || !connectionString}
          >
            Save Settings
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default MongoDBSettings;
