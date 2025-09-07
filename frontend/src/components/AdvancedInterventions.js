import React, { useState, useCallback, useEffect } from 'react';
import api from '../services/api'; // Make sure this path is correct

const AdvancedInterventions = ({ userId, latestMood }) => {
  const [intervention, setIntervention] = useState(null);
  const [preferences, setPreferences] = useState({
    prefers_short_activities: true,
    likes_meditation: true,
    prefers_professional_help: false
  });
  const [loading, setLoading] = useState(false);

  const getIntervention = useCallback(async () => {
    if (!latestMood) return;
    
    setLoading(true);
    try {
      const response = await api.post('/intervention/advanced', {
        emotion: latestMood.text_emotion,
        mood_score: latestMood.mood_score,
        user_preferences: preferences
      });
      setIntervention(response.data);
    } catch (error) {
      console.error('Error getting intervention:', error);
    }
    setLoading(false);
  }, [latestMood, preferences]);

  useEffect(() => {
    getIntervention();
  }, [getIntervention]);

  return (
    <div className="interventions-container">
      <div className="section-header">
        <h2>💡 Personalized Support</h2>
        <p>AI-curated wellness activities just for you</p>
      </div>
      
      <div className="preferences-card">
        <h3>🎛️ Your Preferences</h3>
        <div className="preferences-grid">
          <label className="preference-item">
            <input
              type="checkbox"
              checked={preferences.prefers_short_activities}
              onChange={(e) => setPreferences({
                ...preferences,
                prefers_short_activities: e.target.checked
              })}
            />
            <span>⏱️ Quick activities (under 10 minutes)</span>
          </label>
          
          <label className="preference-item">
            <input
              type="checkbox"
              checked={preferences.likes_meditation}
              onChange={(e) => setPreferences({
                ...preferences,
                likes_meditation: e.target.checked
              })}
            />
            <span>🧘‍♀️ Meditation & mindfulness</span>
          </label>
          
          <label className="preference-item">
            <input
              type="checkbox"
              checked={preferences.prefers_professional_help}
              onChange={(e) => setPreferences({
                ...preferences,
                prefers_professional_help: e.target.checked
              })}
            />
            <span>👨‍⚕️ Professional therapy resources</span>
          </label>
        </div>
        
        <button onClick={getIntervention} className="update-preferences-btn">
          🔄 Update Suggestions
        </button>
      </div>

      {loading && <div className="loading-spinner">🤖 Personalizing suggestions...</div>}

      {intervention && (
        <div className="intervention-content">
          {intervention.urgency === "IMMEDIATE" && (
            <div className="emergency-alert">
              <h3>🆘 IMMEDIATE SUPPORT NEEDED</h3>
              <p>Your safety is our top priority. Please reach out for help immediately.</p>
            </div>
          )}

          <div className="primary-suggestion">
            <h3>🎯 Recommended for You Right Now</h3>
            <div className="suggestion-content">
              {intervention.primary_suggestion}
            </div>
            <div className="duration-badge">
              ⏱️ {intervention.estimated_duration}
            </div>
          </div>

          {intervention.alternative_suggestions && intervention.alternative_suggestions.length > 0 && (
            <div className="alternatives-section">
              <h4>🔄 Alternative Options</h4>
              <div className="alternatives-grid">
                {intervention.alternative_suggestions.map((suggestion, index) => (
                  <div key={index} className="alternative-card">
                    {suggestion}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="resources-section">
            <h4>📚 Additional Resources</h4>
            <div className="resources-grid">
              {intervention.resources?.map((resource, index) => (
                <div key={index} className="resource-card">
                  {resource}
                </div>
              ))}
            </div>
          </div>

          {intervention.follow_up_recommended && (
            <div className="follow-up-notice">
              <strong>📅 Follow-up recommended within 24 hours</strong>
              <p>Please check in again to track your progress and wellbeing.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AdvancedInterventions;
