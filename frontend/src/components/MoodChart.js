import React from 'react';

const MoodChart = ({ history }) => {
  if (!history.length) {
    return (
      <div className="no-data">
        <div className="section-header">
          <h2>📊 Mood Trends</h2>
          <p>Start logging your mood to see beautiful charts and insights!</p>
        </div>
      </div>
    );
  }

  const chartData = history.slice(-14).reverse();

  const getBarColor = (score, riskLevel) => {
    if (riskLevel === 'HIGH') return '#F44336';
    if (riskLevel === 'MEDIUM') return '#FF9800';
    if (score >= 7) return '#4CAF50';
    if (score >= 4) return '#2196F3';
    return '#FF5722';
  };

  return (
    <div className="mood-chart-container">
      <div className="section-header">
        <h2>📊 Your Mood Journey</h2>
        <p>Track your emotional wellness over the past {chartData.length} days</p>
      </div>
      
      <div className="chart-wrapper">
        <div className="chart-container">
          {chartData.map((entry, index) => (
            <div key={index} className="chart-bar-wrapper">
              <div 
                className="chart-bar" 
                style={{ 
                  height: `${(entry.mood_score / 10) * 100}%`,
                  backgroundColor: getBarColor(entry.mood_score, entry.risk_level)
                }}
                title={`${entry.mood_score}/10 - ${entry.text_emotion} - ${entry.risk_level} risk`}
              />
              <div className="bar-label">
                {new Date(entry.timestamp).toLocaleDateString('en-US', { 
                  month: 'short', 
                  day: 'numeric' 
                })}
              </div>
            </div>
          ))}
        </div>
        
        <div className="chart-legend">
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#4CAF50' }}></div>
            <span>Great (7-10)</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#2196F3' }}></div>
            <span>Good (4-6)</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#FF9800' }}></div>
            <span>Medium Risk</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#F44336' }}></div>
            <span>High Risk</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MoodChart;
