import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Rating,
  TextField
} from '@mui/material';
import { styled } from '@mui/material/styles';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import BookmarkBorderIcon from '@mui/icons-material/BookmarkBorder';
import BookmarkIcon from '@mui/icons-material/Bookmark';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import { interventionsAPI } from '../services/api';

const StyledCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  transition: 'box-shadow 0.2s ease',
  borderRadius: '2px',
  border: '1px solid #e0e0e0',
  boxShadow: '0 2px 6px rgba(0, 0, 0, 0.03)',
  '&:hover': {
    boxShadow: '0 4px 10px rgba(0, 0, 0, 0.06)',
  },
}));

const typeColors = {
  mindfulness: '#4CAF50',
  cognitive: '#2196F3',
  physical: '#FF9800',
  social: '#9C27B0',
  creative: '#F44336'
};

const emotionColors = {
  anxiety: '#FFC107',
  sadness: '#9C27B0',
  stress: '#FF5722',
  anger: '#F44336',
  general: '#607D8B'
};

const InterventionCard = ({ intervention }) => {
  const [saved, setSaved] = useState(false);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [rating, setRating] = useState(0);
  const [feedback, setFeedback] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleSave = () => {
    setSaved(!saved);
  };

  const handleFeedbackOpen = () => {
    setFeedbackOpen(true);
  };

  const handleFeedbackClose = () => {
    setFeedbackOpen(false);
  };

  const handleSubmitFeedback = async () => {
    if (rating === 0) return;
    
    setSubmitting(true);
    try {
      await interventionsAPI.provideFeedback({
        intervention_id: intervention.id || 'mock-id',
        rating: rating,
        feedback: feedback,
        timestamp: new Date().toISOString()
      });
      
      // Success handling
      setTimeout(() => {
        setFeedbackOpen(false);
        setRating(0);
        setFeedback('');
      }, 500);
      
    } catch (error) {
      console.error('Error submitting feedback:', error);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <StyledCard>
        <CardContent sx={{ flexGrow: 1 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <Typography variant="h6" component="h2" gutterBottom>
              {intervention.title}
            </Typography>
            <Button 
              onClick={handleSave}
              sx={{ minWidth: 'auto', p: 0 }}
            >
              {saved ? 
                <BookmarkIcon color="primary" /> : 
                <BookmarkBorderIcon color="action" />
              }
            </Button>
          </Box>
          
          <Box sx={{ mb: 2, display: 'flex', gap: 1 }}>
            <Chip 
              label={intervention.type} 
              size="small"
              sx={{ 
                bgcolor: `${typeColors[intervention.type]}20`, 
                color: typeColors[intervention.type],
                fontWeight: 'medium'
              }} 
            />
            <Chip 
              label={intervention.emotion} 
              size="small" 
              sx={{ 
                bgcolor: `${emotionColors[intervention.emotion]}20`, 
                color: emotionColors[intervention.emotion],
                fontWeight: 'medium'
              }} 
            />
            <Chip 
              icon={<AccessTimeIcon sx={{ fontSize: '1rem !important' }} />} 
              label={intervention.duration} 
              size="small" 
              variant="outlined"
            />
          </Box>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {intervention.description}
          </Typography>
        </CardContent>
        
        <Box sx={{ p: 2, pt: 0, display: 'flex', justifyContent: 'space-between' }}>
          <Button 
            variant="contained" 
            startIcon={<PlayCircleOutlineIcon />}
            size="small"
          >
            Start
          </Button>
          
          <Button 
            variant="outlined"
            size="small"
            onClick={handleFeedbackOpen}
          >
            Give Feedback
          </Button>
        </Box>
      </StyledCard>
      
      {/* Feedback Dialog */}
      <Dialog open={feedbackOpen} onClose={handleFeedbackClose} maxWidth="sm" fullWidth>
        <DialogTitle>How helpful was this intervention?</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', my: 2 }}>
            <Rating
              name="feedback-rating"
              value={rating}
              onChange={(event, newValue) => {
                setRating(newValue);
              }}
              size="large"
              precision={1}
              sx={{ fontSize: '2.5rem', mb: 2 }}
            />
            
            <TextField
              label="Additional feedback (optional)"
              multiline
              rows={4}
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              fullWidth
              variant="outlined"
              placeholder="How did this intervention help you? Any suggestions for improvement?"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleFeedbackClose}>
            Cancel
          </Button>
          <Button 
            onClick={handleSubmitFeedback} 
            variant="contained"
            disabled={rating === 0 || submitting}
          >
            Submit
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default InterventionCard;
