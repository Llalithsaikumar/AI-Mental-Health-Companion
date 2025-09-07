import React, { useState, useCallback, useEffect } from 'react';
import api from '../services/api'; // Make sure this path is correct

const MoodPredictor = () => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const getPrediction = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.post('/mood/predict');
      setPrediction(response.data);
    } catch (error) {
      console.error('Error getting prediction:', error);
      if (error.response?.data?.message) {
        setPrediction({ message: error.response.data.message });
      }
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    getPrediction();
  }, [getPrediction]);

  return (
    <div className="predictor-container">
      <div className="section-header">
        <h2>🔮 AI Mood Predictions</h2>
        <p>Advanced analytics powered by machine learning</p>
      </div>
      
      {loading && <div className="loading-spinner">🤖 AI analyzing patterns...</div>}
      
      {prediction?.message && (
        <div className="info-card">
          <h3>📚 Getting Started</h3>
          <p>{prediction.message}</p>
        </div>
      )}
      
      {prediction?.prediction && (
        <div className="prediction-results">
          <div className="prediction-inner">
            <div className="prediction-score">
              <h3>Next Predicted Mood</h3>
              <div className="score-display">
                <span className="score-number">{prediction.prediction.predicted_mood}</span>
                <span className="score-max">/10</span>
              </div>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill" 
                  style={{ width: `${prediction.prediction.confidence * 100}%` }}
                />
                <span className="confidence-text">
                  {(prediction.prediction.confidence * 100).toFixed(0)}% Confidence
                </span>
              </div>
            </div>
          </div>
          
          <div className="recommendations-card">
            <h4>🎯 AI Recommendations</h4>
            <p>{prediction.recommendations}</p>
          </div>
          
          <button onClick={getPrediction} className="update-btn">
            🔄 Update Prediction
          </button>
        </div>
      )}
    </div>
  );
};

export default MoodPredictor;
