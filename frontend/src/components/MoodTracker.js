import React, { useState } from 'react';
import { Box, Slider, Typography, TextField, Button, Paper, Chip } from '@mui/material';
import { styled } from '@mui/material/styles';
import SentimentVeryDissatisfiedIcon from '@mui/icons-material/SentimentVeryDissatisfied';
import SentimentDissatisfiedIcon from '@mui/icons-material/SentimentDissatisfied';
import SentimentNeutralIcon from '@mui/icons-material/SentimentNeutral';
import SentimentSatisfiedIcon from '@mui/icons-material/SentimentSatisfied';
import SentimentSatisfiedAltIcon from '@mui/icons-material/SentimentSatisfiedAltOutlined';
import SentimentVerySatisfiedIcon from '@mui/icons-material/SentimentVerySatisfied';
import { moodAPI } from '../services/api';

const StyledSlider = styled(Slider)(({ theme }) => ({
  height: 8,
  '& .MuiSlider-track': {
    border: 'none',
  },
  '& .MuiSlider-thumb': {
    height: 24,
    width: 24,
    backgroundColor: '#fff',
    border: '2px solid currentColor',
    '&:focus, &:hover, &.Mui-active, &.Mui-focusVisible': {
      boxShadow: 'inherit',
    },
    '&:before': {
      display: 'none',
    },
  },
  '& .MuiSlider-valueLabel': {
    lineHeight: 1.2,
    fontSize: 12,
    background: 'unset',
    padding: 0,
    width: 32,
    height: 32,
    borderRadius: '50% 50% 50% 0',
    backgroundColor: theme.palette.primary.main,
    transformOrigin: 'bottom left',
    transform: 'translate(50%, -100%) rotate(-45deg) scale(0)',
    '&:before': { display: 'none' },
    '&.MuiSlider-valueLabelOpen': {
      transform: 'translate(50%, -100%) rotate(-45deg) scale(1)',
    },
    '& > *': {
      transform: 'rotate(45deg)',
    },
  },
}));

const EmotionChip = styled(Chip)(({ theme, selected }) => ({
  margin: theme.spacing(0.5),
  fontWeight: selected ? 'bold' : 'normal',
  backgroundColor: selected ? theme.palette.primary.main : theme.palette.background.default,
  color: selected ? theme.palette.primary.contrastText : theme.palette.text.primary,
  '&:hover': {
    backgroundColor: selected ? theme.palette.primary.dark : theme.palette.action.hover,
  },
}));

const emotions = [
  'Joy', 'Contentment', 'Gratitude', 'Pride', 'Excitement',
  'Neutral', 'Calm', 'Relaxed',
  'Anxiety', 'Stress', 'Worry', 'Fear',
  'Sadness', 'Disappointment', 'Loneliness', 'Grief',
  'Anger', 'Frustration', 'Irritation'
];

const getMoodIcon = (value) => {
  if (value <= 2) {
    return <SentimentVeryDissatisfiedIcon sx={{ fontSize: 40, color: '#d32f2f' }} />;
  } else if (value <= 4) {
    return <SentimentDissatisfiedIcon sx={{ fontSize: 40, color: '#f57c00' }} />;
  } else if (value <= 6) {
    return <SentimentNeutralIcon sx={{ fontSize: 40, color: '#fbc02d' }} />;
  } else if (value <= 8) {
    return <SentimentSatisfiedIcon sx={{ fontSize: 40, color: '#7cb342' }} />;
  } else {
    return <SentimentVerySatisfiedIcon sx={{ fontSize: 40, color: '#388e3c' }} />;
  }
};

const getMoodLabel = (value) => {
  if (value <= 2) {
    return 'Very Bad';
  } else if (value <= 4) {
    return 'Bad';
  } else if (value <= 6) {
    return 'Okay';
  } else if (value <= 8) {
    return 'Good';
  } else {
    return 'Very Good';
  }
};

const MoodTracker = ({ onMoodLogged = () => {} }) => {
  const [moodScore, setMoodScore] = useState(5);
  const [text, setText] = useState('');
  const [selectedEmotion, setSelectedEmotion] = useState('');
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleMoodChange = (event, newValue) => {
    setMoodScore(newValue);
  };

  const handleEmotionSelect = (emotion) => {
    setSelectedEmotion(emotion);
  };

  const handleSubmit = async () => {
    setLoading(true);
    
    try {
      // Convert the emotion to lowercase to match backend expectations
      const textEmotion = selectedEmotion ? selectedEmotion.toLowerCase() : '';
      
      const response = await moodAPI.logMood({
        mood_score: moodScore,
        text_content: text,
        text_emotion: textEmotion,
        audio_emotion: null
      });
      
      console.log("Mood logged successfully:", response.data);
      setSubmitted(true);
      
      // Call the callback to update parent components
      onMoodLogged();
      
      // Reset the form after submission
      setTimeout(() => {
        setMoodScore(5);
        setText('');
        setSelectedEmotion('');
        setSubmitted(false);
      }, 3000);
      
    } catch (error) {
      console.error('Error logging mood:', error);
      alert('Failed to log mood. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ py: 2 }}>
      {submitted ? (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <SentimentSatisfiedAltIcon sx={{ fontSize: 60, color: 'success.main', mb: 2 }} />
          <Typography variant="h5" gutterBottom>
            Thanks for checking in!
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Your mood has been logged successfully.
          </Typography>
        </Box>
      ) : (
        <>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            {getMoodIcon(moodScore)}
            <Box sx={{ ml: 2, flexGrow: 1 }}>
              <Typography variant="h6" gutterBottom>
                {getMoodLabel(moodScore)}
              </Typography>
              <StyledSlider
                value={moodScore}
                onChange={handleMoodChange}
                aria-label="Mood Score"
                valueLabelDisplay="auto"
                step={1}
                marks
                min={1}
                max={10}
              />
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                <Typography variant="body2" color="text.secondary">Not well</Typography>
                <Typography variant="body2" color="text.secondary">Very well</Typography>
              </Box>
            </Box>
          </Box>

          <Typography variant="subtitle1" gutterBottom sx={{ mt: 3 }}>
            How would you describe your emotions?
          </Typography>
          <Paper variant="outlined" sx={{ p: 1, maxHeight: 150, overflowY: 'auto' }}>
            <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
              {emotions.map((emotion) => (
                <EmotionChip
                  key={emotion}
                  label={emotion}
                  onClick={() => handleEmotionSelect(emotion)}
                  selected={selectedEmotion === emotion}
                  clickable
                />
              ))}
            </Box>
          </Paper>

          <TextField
            label="Journal your thoughts (optional)"
            multiline
            rows={4}
            value={text}
            onChange={(e) => setText(e.target.value)}
            fullWidth
            margin="normal"
            variant="outlined"
            placeholder="How are you feeling today? What's on your mind?"
          />

          <Button
            variant="contained"
            color="primary"
            fullWidth
            size="large"
            onClick={handleSubmit}
            disabled={loading || !selectedEmotion}
            sx={{ mt: 2 }}
          >
            {loading ? 'Logging...' : 'Log My Mood'}
          </Button>
        </>
      )}
    </Box>
  );
};

export default MoodTracker;
