import React, { useState, useEffect } from 'react';
import axios from 'axios';

// Real-time mood monitoring component
export const RealTimeMoodMonitor = ({ userId }) => {
  const [ws, setWs] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [isMonitoring, setIsMonitoring] = useState(false);

  useEffect(() => {
    if (userId && isMonitoring) {
      const websocket = new WebSocket(`ws://localhost:8000/ws/${userId}`);
      
      websocket.onopen = () => {
        console.log('Real-time monitoring connected');
        setWs(websocket);
      };

      websocket.onmessage = (event) => {
        const alert = JSON.parse(event.data);
        setAlerts(prev => [...prev, { ...alert, timestamp: new Date() }]);
      };

      websocket.onclose = () => {
        console.log('Real-time monitoring disconnected');
        setWs(null);
      };

      return () => {
        websocket.close();
      };
    }
  }, [userId, isMonitoring]);

  const startMonitoring = () => {
    setIsMonitoring(true);
  };

  const stopMonitoring = () => {
    setIsMonitoring(false);
    if (ws) {
      ws.close();
    }
  };

  return (
    <div className="realtime-monitor">
      <h3>Real-time Monitoring</h3>
      
      <div className="monitor-controls">
        {!isMonitoring ? (
          <button onClick={startMonitoring} className="start-monitoring">
            Start Monitoring
          </button>
        ) : (
          <button onClick={stopMonitoring} className="stop-monitoring">
            Stop Monitoring
          </button>
        )}
      </div>

      {alerts.length > 0 && (
        <div className="alerts-section">
          <h4>Recent Alerts</h4>
          {alerts.slice(-5).map((alert, index) => (
            <div key={index} className={`alert alert-${alert.severity}`}>
              <span className="alert-time">
                {alert.timestamp.toLocaleTimeString()}
              </span>
              <span className="alert-message">{alert.message}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Mood prediction component
export const MoodPredictor = ({ userId }) => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const getPrediction = async () => {
    setLoading(true);
    try {
      const response = await api.post('/mood/predict');
      setPrediction(response.data);
    } catch (error) {
      console.error('Error getting mood prediction:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    if (userId) {
      getPrediction();
    }
  }, [userId]);

  if (loading) return <div>Generating prediction...</div>;
  if (!prediction) return <div>No prediction available</div>;

  return (
    <div className="mood-predictor">
      <h3>Mood Prediction</h3>
      
      <div className="prediction-card">
        <div className="predicted-score">
          <h4>Predicted Next Mood</h4>
          <div className="score-display">
            {prediction.prediction.predicted_mood}/10
          </div>
          <div className="confidence">
            Confidence: {(prediction.prediction.confidence * 100).toFixed(1)}%
          </div>
        </div>

        {prediction.anomaly_detection.is_anomaly && (
          <div className="anomaly-alert">
            <strong>⚠️ Unusual pattern detected</strong>
            <p>Your mood patterns seem different from usual. Consider reaching out for support.</p>
          </div>
        )}

        <div className="recommendations">
          <h5>Recommendations</h5>
          <p>{prediction.recommendations}</p>
        </div>
      </div>

      <button onClick={getPrediction} className="refresh-prediction">
        Update Prediction
      </button>
    </div>
  );
};

// Advanced intervention component
export const AdvancedInterventions = ({ userId, currentMood, currentEmotion }) => {
  const [intervention, setIntervention] = useState(null);
  const [userPreferences, setUserPreferences] = useState({
    prefers_short_activities: true,
    likes_meditation: true,
    prefers_professional_help: false
  });
  const [feedback, setFeedback] = useState({});

  const getAdvancedIntervention = async () => {
    try {
      const response = await api.post('/intervention/advanced', {
        emotion: currentEmotion,
        mood_score: currentMood,
        user_preferences: userPreferences
      });
      setIntervention(response.data);
    } catch (error) {
      console.error('Error getting advanced intervention:', error);
    }
  };

  const submitFeedback = async (interventionId, rating, comments) => {
    try {
      await api.post('/intervention/feedback', {
        intervention_id: interventionId,
        effectiveness_rating: rating,
        feedback: comments
      });
      alert('Thank you for your feedback!');
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  useEffect(() => {
    if (currentEmotion && currentMood) {
      getAdvancedIntervention();
    }
  }, [currentEmotion, currentMood]);

  return (
    <div className="advanced-interventions">
      <h3>Personalized Interventions</h3>

      {/* User Preferences */}
      <div className="preferences-section">
        <h4>Preferences</h4>
        <label>
          <input
            type="checkbox"
            checked={userPreferences.prefers_short_activities}
            onChange={(e) => setUserPreferences({
              ...userPreferences,
              prefers_short_activities: e.target.checked
            })}
          />
          Prefer short activities (under 10 minutes)
        </label>
        
        <label>
          <input
            type="checkbox"
            checked={userPreferences.likes_meditation}
            onChange={(e) => setUserPreferences({
              ...userPreferences,
              likes_meditation: e.target.checked
            })}
          />
          Enjoy meditation and mindfulness
        </label>
        
        <label>
          <input
            type="checkbox"
            checked={userPreferences.prefers_professional_help}
            onChange={(e) => setUserPreferences({
              ...userPreferences,
              prefers_professional_help: e.target.checked
            })}
          />
          Prefer professional therapy resources
        </label>
        
        <button onClick={getAdvancedIntervention} className="update-preferences">
          Update Recommendations
        </button>
      </div>

      {intervention && (
        <div className="intervention-content">
          {intervention.urgency === "IMMEDIATE" && (
            <div className="emergency-alert">
              <strong>🚨 IMMEDIATE ATTENTION NEEDED</strong>
            </div>
          )}

          <div className="primary-intervention">
            <h4>Recommended Action</h4>
            <p>{intervention.primary_suggestion}</p>
            <div className="duration">
              Estimated time: {intervention.estimated_duration}
            </div>
          </div>

          {intervention.alternative_suggestions.length > 0 && (
            <div className="alternative-interventions">
              <h5>Alternative Options</h5>
              {intervention.alternative_suggestions.map((suggestion, index) => (
                <div key={index} className="alternative-item">
                  {suggestion}
                </div>
              ))}
            </div>
          )}

          <div className="resources-section">
            <h5>Additional Resources</h5>
            {intervention.resources.map((resource, index) => (
              <div key={index} className="resource-item">
                {resource}
              </div>
            ))}
          </div>

          {intervention.follow_up_recommended && (
            <div className="follow-up-notice">
              <strong>📅 Follow-up recommended</strong>
              <p>Please check in again within 24 hours to track your progress.</p>
            </div>
          )}

          {/* Feedback Section */}
          <div className="feedback-section">
            <h5>How helpful was this suggestion?</h5>
            <div className="rating-buttons">
              {[1, 2, 3, 4, 5].map(rating => (
                <button
                  key={rating}
                  onClick={() => setFeedback({ ...feedback, rating })}
                  className={`rating-btn ${feedback.rating === rating ? 'selected' : ''}`}
                >
                  {rating}
                </button>
              ))}
            </div>
            
            <textarea
              placeholder="Any additional feedback? (optional)"
              value={feedback.comments || ''}
              onChange={(e) => setFeedback({ ...feedback, comments: e.target.value })}
            />
            
            <button
              onClick={() => submitFeedback(intervention.intervention_type, feedback.rating, feedback.comments)}
              disabled={!feedback.rating}
              className="submit-feedback"
            >
              Submit Feedback
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// Crisis assessment component
export const CrisisAssessment = ({ currentText, currentMood }) => {
  const [assessment, setAssessment] = useState(null);
  const [loading, setLoading] = useState(false);

  const performAssessment = async () => {
    if (!currentText && !currentMood) return;

    setLoading(true);
    try {
      const response = await api.post('/crisis/assess', {
        text: currentText,
        mood_score: currentMood
      });
      setAssessment(response.data);
    } catch (error) {
      console.error('Error performing crisis assessment:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    performAssessment();
  }, [currentText, currentMood]);

  if (loading) return <div>Analyzing...</div>;
  if (!assessment) return null;

  return (
    <div className="crisis-assessment">
      {assessment.risk_level !== 'MINIMAL' && (
        <div className={`risk-alert risk-${assessment.risk_level.toLowerCase()}`}>
          <h4>Risk Level: {assessment.risk_level}</h4>
          <div className="risk-score">
            Risk Score: {assessment.risk_score}
          </div>
          
          {assessment.risk_factors.length > 0 && (
            <div className="risk-factors">
              <h5>Identified Factors:</h5>
              <ul>
                {assessment.risk_factors.map((factor, index) => (
                  <li key={index}>{factor}</li>
                ))}
              </ul>
            </div>
          )}
          
          {assessment.requires_intervention && (
            <div className="intervention-required">
              <strong>⚠️ Professional intervention recommended</strong>
              <p>Please consider reaching out to a mental health professional or crisis support service.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
