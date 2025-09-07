import React, { useState, useEffect } from 'react';

const RealTimeAlerts = ({ userId }) => {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    if (userId) {
      const websocket = new WebSocket(`ws://localhost:8000/ws/${userId}`);
      
      websocket.onopen = () => {
        console.log('🔔 Real-time monitoring connected');
      };

      websocket.onmessage = (event) => {
        const alert = JSON.parse(event.data);
        setAlerts(prev => [...prev.slice(-2), { ...alert, timestamp: new Date() }]);
      };

      websocket.onclose = () => {
        console.log('Real-time monitoring disconnected');
      };

      return () => {
        websocket.close();
      };
    }
  }, [userId]);

  if (alerts.length === 0) return null;

  return (
    <div className="alerts-container">
      <h4>🔔 Live Alerts</h4>
      {alerts.map((alert, index) => (
        <div key={index} className={`alert alert-${alert.severity}`}>
          <div className="alert-time">
            {alert.timestamp.toLocaleTimeString()}
          </div>
          <div className="alert-message">
            {alert.message}
          </div>
        </div>
      ))}
    </div>
  );
};

export default RealTimeAlerts;
