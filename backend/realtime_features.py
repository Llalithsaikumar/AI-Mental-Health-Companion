from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict
import asyncio
import json
from datetime import datetime
import uuid

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[int, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.user_connections[user_id] = websocket
        
    def disconnect(self, websocket: WebSocket, user_id: int):
        self.active_connections.remove(websocket)
        if user_id in self.user_connections:
            del self.user_connections[user_id]
            
    async def send_personal_message(self, message: str, user_id: int):
        if user_id in self.user_connections:
            await self.user_connections[user_id].send_text(message)
            
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

class RealTimeMonitoringSystem:
    def __init__(self):
        self.monitoring_active = {}
        self.alert_thresholds = {
            'low_mood_threshold': 2.5,
            'mood_drop_threshold': 2.0,
            'inactivity_hours': 24
        }
        
    async def start_monitoring(self, user_id: int):
        """Start real-time monitoring for a user"""
        self.monitoring_active[user_id] = {
            'start_time': datetime.now(),
            'last_activity': datetime.now(),
            'mood_alerts': 0,
            'intervention_suggestions': []
        }
        
    async def process_mood_update(self, user_id: int, mood_data: Dict):
        """Process real-time mood updates"""
        if user_id not in self.monitoring_active:
            await self.start_monitoring(user_id)
            
        # Update last activity
        self.monitoring_active[user_id]['last_activity'] = datetime.now()
        
        # Check for alerts
        alerts = []
        
        if mood_data['mood_score'] <= self.alert_thresholds['low_mood_threshold']:
            alerts.append({
                'type': 'low_mood',
                'message': 'Low mood detected - intervention recommended',
                'severity': 'medium'
            })
            
        # Check for rapid mood drop
        if 'previous_mood' in mood_data:
            mood_drop = mood_data['previous_mood'] - mood_data['mood_score']
            if mood_drop >= self.alert_thresholds['mood_drop_threshold']:
                alerts.append({
                    'type': 'mood_drop',
                    'message': 'Significant mood drop detected',
                    'severity': 'high'
                })
                
        return alerts
        
    async def check_inactivity(self, user_id: int) -> Dict:
        """Check for user inactivity"""
        if user_id not in self.monitoring_active:
            return {'inactive': False}
            
        last_activity = self.monitoring_active[user_id]['last_activity']
        hours_inactive = (datetime.now() - last_activity).total_seconds() / 3600
        
        if hours_inactive >= self.alert_thresholds['inactivity_hours']:
            return {
                'inactive': True,
                'hours_inactive': round(hours_inactive, 1),
                'message': 'User has been inactive - consider check-in'
            }
            
        return {'inactive': False}

# Smart notification system
class SmartNotificationSystem:
    def __init__(self):
        self.notification_preferences = {}
        self.notification_history = {}
        
    def set_user_preferences(self, user_id: int, preferences: Dict):
        """Set notification preferences for user"""
        self.notification_preferences[user_id] = preferences
        
    async def should_send_notification(self, user_id: int, notification_type: str) -> bool:
        """Determine if notification should be sent based on user preferences and history"""
        
        prefs = self.notification_preferences.get(user_id, {})
        
        # Check if user wants this type of notification
        if not prefs.get(f'allow_{notification_type}', True):
            return False
            
        # Check timing preferences
        current_hour = datetime.now().hour
        quiet_hours = prefs.get('quiet_hours', [])
        if current_hour in quiet_hours:
            return False
            
        # Check frequency limits
        if user_id in self.notification_history:
            recent_notifications = [
                n for n in self.notification_history[user_id] 
                if (datetime.now() - n['timestamp']).total_seconds() < 3600  # Last hour
            ]
            
            max_per_hour = prefs.get('max_notifications_per_hour', 3)
            if len(recent_notifications) >= max_per_hour:
                return False
                
        return True
        
    async def send_smart_notification(self, user_id: int, notification: Dict):
        """Send notification if conditions are met"""
        
        if await self.should_send_notification(user_id, notification['type']):
            # Record notification
            if user_id not in self.notification_history:
                self.notification_history[user_id] = []
                
            notification['timestamp'] = datetime.now()
            notification['id'] = str(uuid.uuid4())
            
            self.notification_history[user_id].append(notification)
            
            return {
                'sent': True,
                'notification_id': notification['id'],
                'message': notification['message']
            }
            
        return {'sent': False, 'reason': 'Filtered by preferences'}
