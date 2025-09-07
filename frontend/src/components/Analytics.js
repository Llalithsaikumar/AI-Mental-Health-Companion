import React, { useState, useEffect } from 'react';
import api from '../services/api'; // Make sure this path is correct

const Analytics = ({ userId }) => {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await api.get('/analytics/mood-trends?days=30');
        setAnalytics(response.data);
      } catch (error) {
        console.error('Error fetching analytics:', error);
      }
      setLoading(false);
    };

    fetchAnalytics();
  }, [userId]);

  if (loading) return <div className="loading-spinner">📊 Loading analytics...</div>;
  if (!analytics) return <div>No analytics data available</div>;

  return (
    <div className="analytics-container">
      <div className="section-header">
        <h2>📈 Your Mental Health Analytics</h2>
        <p>Insights and patterns from your wellness journey</p>
      </div>
      
      <div className="analytics-grid">
        <div className="stat-card">
          <div className="stat-icon">📊</div>
          <h4>Average Mood</h4>
          <div className="stat-value">{analytics.average_mood || 'N/A'}<span>/10</span></div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon">📈</div>
          <h4>Trend</h4>
          <div className={`stat-value trend-${analytics.mood_trend}`}>
            {analytics.mood_trend || 'stable'}
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon">📝</div>
          <h4>Total Entries</h4>
          <div className="stat-value">{analytics.total_entries || 0}</div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
