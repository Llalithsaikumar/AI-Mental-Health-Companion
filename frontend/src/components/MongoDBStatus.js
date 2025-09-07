import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Divider
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import StorageIcon from '@mui/icons-material/Storage';
import api from '../services/api';

const MongoDBStatus = () => {
  const [status, setStatus] = useState({
    loading: true,
    connected: false,
    type: '',
    collections: {},
    error: null
  });

  useEffect(() => {
    const checkDatabaseStatus = async () => {
      try {
        const response = await api.get('/db/status');
        setStatus({
          loading: false,
          connected: response.data.status === 'connected',
          type: response.data.type,
          collections: response.data.collections || response.data.tables || {},
          error: response.data.error || null
        });
      } catch (error) {
        setStatus({
          loading: false,
          connected: false,
          type: 'Unknown',
          collections: {},
          error: 'Could not reach the database API'
        });
        console.error('Error checking database status:', error);
      }
    };

    checkDatabaseStatus();
  }, []);

  if (status.loading) {
    return (
      <Paper elevation={2} sx={{ p: 3, borderRadius: 2, height: '100%' }}>
        <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" height="100%">
          <CircularProgress size={40} sx={{ mb: 2 }} />
          <Typography variant="body1">Checking database connection...</Typography>
        </Box>
      </Paper>
    );
  }

  return (
    <Paper elevation={2} sx={{ p: 3, borderRadius: 2, height: '100%' }}>
      <Box display="flex" alignItems="center" mb={2}>
        <StorageIcon fontSize="large" sx={{ mr: 1, color: 'primary.main' }} />
        <Typography variant="h6" fontWeight="600">
          Database Status
        </Typography>
      </Box>

      <Box display="flex" alignItems="center" mb={2}>
        <Typography variant="body1" mr={2}>
          Connection:
        </Typography>
        <Chip
          icon={status.connected ? <CheckCircleIcon /> : <ErrorIcon />}
          label={status.connected ? 'Connected' : 'Disconnected'}
          color={status.connected ? 'success' : 'error'}
          variant="outlined"
          size="small"
        />
      </Box>

      <Box mb={2}>
        <Typography variant="body1" mb={1}>
          Database Type: <strong>{status.type}</strong>
        </Typography>
      </Box>

      {status.connected ? (
        <Box>
          <Typography variant="body1" mb={1} fontWeight={500}>
            {status.type === 'MongoDB' ? 'Collections' : 'Tables'}:
          </Typography>
          <List dense sx={{ bgcolor: 'background.default', borderRadius: 1, maxHeight: 200, overflow: 'auto' }}>
            {Object.entries(status.collections).map(([name, count], index) => (
              <React.Fragment key={name}>
                {index > 0 && <Divider />}
                <ListItem>
                  <ListItemText 
                    primary={name} 
                    secondary={`${count} ${count === 1 ? 'document' : 'documents'}`} 
                  />
                </ListItem>
              </React.Fragment>
            ))}
            {Object.keys(status.collections).length === 0 && (
              <ListItem>
                <ListItemText primary="No collections found" />
              </ListItem>
            )}
          </List>
        </Box>
      ) : (
        <Box sx={{ mt: 2, p: 2, bgcolor: 'error.light', borderRadius: 1 }}>
          <Typography variant="body2" color="error.contrastText">
            {status.error || 'Database connection failed. Please check server logs.'}
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default MongoDBStatus;
