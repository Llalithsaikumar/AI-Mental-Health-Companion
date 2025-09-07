import React, { useState } from 'react';
import api from '../services/api'; // Make sure this path is correct

const MoodEntry = ({ onMoodEntry }) => {
  const [textInput, setTextInput] = useState('');
  const [moodScore, setMoodScore] = useState(5);
  const [journalText, setJournalText] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [emotionAnalysis, setEmotionAnalysis] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      const response = await api.post('/mood/entry', {
        text_input: textInput,
        mood_score: moodScore,
        journal_text: journalText
      });
      
      onMoodEntry(response.data);
      setEmotionAnalysis(response.data.emotions);
      setTextInput('');
      setJournalText('');
      setMoodScore(5);
      
      if (response.data.crisis_assessment?.risk_level === 'HIGH') {
        alert('🆘 High risk detected. Please consider reaching out for immediate support.');
      } else if (response.data.crisis_assessment?.risk_level === 'MEDIUM') {
        alert('⚠️ Moderate risk detected. Consider connecting with support resources');
      }
      
    } catch (error) {
      alert('Error submitting mood entry: ' + (error.response?.data?.detail || error.message));
    }
    
    setIsSubmitting(false);
  };

  const getMoodEmoji = (score) => {
    if (score >= 8) return '😊';
    if (score >= 6) return '🙂';
    if (score >= 4) return '😐';
    if (score >= 2) return '😔';
    return '😢';
  };

  return (
    <div className="mood-entry-container">
      <div className="section-header">
        <h2>📝 How are you feeling today?</h2>
        <p>Share your thoughts and emotions with your AI companion</p>
      </div>
      
      <form onSubmit={handleSubmit} className="mood-form">
        <div className="form-group">
          <label className="form-label">💭 Tell me about your day:</label>
          <textarea
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="Share your thoughts, feelings, experiences, or anything on your mind..."
            rows={4}
            className="form-textarea"
          />
        </div>
        
        <div className="form-group">
          <label className="form-label">
            🎭 Overall Mood: {getMoodEmoji(moodScore)} {moodScore}/10
          </label>
          <input
            type="range"
            min="1"
            max="10"
            value={moodScore}
            onChange={(e) => setMoodScore(Number(e.target.value))}
            className="mood-slider"
          />
          <div className="mood-labels">
            <span>😢 Very Low</span>
            <span>😐 Neutral</span>
            <span>😊 Excellent</span>
          </div>
        </div>
        
        <div className="form-group">
          <label className="form-label">📖 Journal Entry (Optional):</label>
          <textarea
            value={journalText}
            onChange={(e) => setJournalText(e.target.value)}
            placeholder="Reflect on your experiences, goals, challenges, gratitude, or insights..."
            rows={3}
            className="form-textarea"
          />
        </div>
        
        <button type="submit" disabled={isSubmitting} className="submit-btn">
          {isSubmitting ? '🤖 AI Analyzing...' : '💾 Submit Entry'}
        </button>
      </form>

      {emotionAnalysis && (
        <div className="emotion-analysis">
          <h3>🎯 AI Emotion Analysis</h3>
          <div className="emotion-grid">
            {Object.entries(emotionAnalysis).map(([emotion, score]) => (
              <div key={emotion} className="emotion-card">
                <div className="emotion-name">{emotion}</div>
                <div className="emotion-bar">
                  <div 
                    className="emotion-fill" 
                    style={{ width: `${score * 100}%` }}
                  />
                </div>
                <div className="emotion-score">{(score * 100).toFixed(0)}%</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default MoodEntry;
