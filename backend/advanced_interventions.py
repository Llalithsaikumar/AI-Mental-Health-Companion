import random
from typing import Dict, List
import numpy as np
from datetime import datetime, timedelta

class AdvancedInterventionEngine:
    def __init__(self):
        self.interventions = {
            'cognitive_behavioral': {
                'anger': [
                    "Let's challenge that angry thought. Ask yourself: 'Is this thought helpful or accurate?'",
                    "Try the STOP technique: Stop, Take a breath, Observe your feelings, Proceed mindfully.",
                    "Practice cognitive reframing: What would you tell a friend in this situation?"
                ],
                'sadness': [
                    "Challenge negative thoughts with evidence. What contradicts this sad thought?",
                    "Try behavioral activation: What small activity brought you joy in the past?",
                    "Use the 3-3-3 rule: Name 3 things you see, hear, and feel to ground yourself."
                ],
                'anxiety': [
                    "Challenge catastrophic thinking: What's the most realistic outcome?",
                    "Use the 5-4-3-2-1 grounding technique for anxiety relief.",
                    "Practice thought defusion: 'I'm having the thought that...' instead of believing it directly."
                ]
            },
            'mindfulness': {
                'general': [
                    "Try a 10-minute guided meditation focusing on breath awareness.",
                    "Practice loving-kindness meditation: send compassion to yourself and others.",
                    "Do a body scan meditation to release tension and increase awareness."
                ],
                'stress': [
                    "Practice the RAIN technique: Recognize, Allow, Investigate, Non-attachment.",
                    "Try mindful breathing: 4 counts in, 6 counts out, for 5 minutes.",
                    "Use progressive muscle relaxation to release physical tension."
                ]
            },
            'behavioral_activation': {
                'low_mood': [
                    "Schedule one small pleasant activity for today - even 5 minutes counts.",
                    "Connect with one person today, even if it's just a text message.",
                    "Do one small task that gives you a sense of accomplishment."
                ],
                'isolation': [
                    "Reach out to one person from your support network today.",
                    "Join an online community or virtual event that interests you.",
                    "Plan a small social activity for this week."
                ]
            },
            'crisis_intervention': [
                "I'm concerned about you. Please reach out to a crisis hotline immediately.",
                "Your safety is the priority. Contact emergency services if you're in immediate danger.",
                "You're not alone. Please connect with a mental health professional today."
            ]
        }
        
        self.resources = {
            'crisis': [
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/",
                "Emergency Services: 911"
            ],
            'therapy': [
                "Psychology Today Therapist Directory",
                "BetterHelp Online Therapy Platform",
                "Talkspace Digital Therapy",
                "Local Community Mental Health Centers"
            ],
            'self_help': [
                "Headspace - Meditation and Mindfulness App",
                "Calm - Sleep and Relaxation App",
                "MindShift - Anxiety Management App",
                "Sanvello - Mood and Anxiety Tracker"
            ]
        }
        
    def get_personalized_intervention(self, 
                                    current_emotion: str, 
                                    mood_score: float,
                                    mood_history: List[Dict],
                                    user_preferences: Dict = None) -> Dict:
        """Generate personalized intervention based on comprehensive analysis"""
        
        intervention_type = self._determine_intervention_type(current_emotion, mood_score, mood_history)
        
        # Select appropriate intervention category
        if mood_score <= 2 or current_emotion in ['crisis', 'severe_depression']:
            interventions = self.interventions['crisis_intervention']
            resources = self.resources['crisis']
            urgency = "IMMEDIATE"
        else:
            interventions = self._select_intervention_strategy(current_emotion, mood_history, user_preferences)
            resources = self._select_resources(intervention_type, user_preferences)
            urgency = "NORMAL"
            
        # Personalize based on user preferences
        if user_preferences:
            interventions = self._filter_by_preferences(interventions, user_preferences)
            
        # Select specific interventions
        selected_interventions = random.sample(
            interventions, 
            min(3, len(interventions))
        )
        
        return {
            'intervention_type': intervention_type,
            'urgency': urgency,
            'primary_suggestion': selected_interventions[0] if selected_interventions else "Take a moment for self-care.",
            'alternative_suggestions': selected_interventions[1:],
            'resources': resources[:3],  # Limit to 3 resources
            'estimated_duration': self._estimate_duration(intervention_type),
            'follow_up_recommended': self._should_follow_up(mood_score, mood_history)
        }
        
    def _determine_intervention_type(self, emotion: str, mood_score: float, mood_history: List[Dict]) -> str:
        """Determine the most appropriate intervention type"""
        
        if mood_score <= 2:
            return "crisis_intervention"
            
        # Analyze mood patterns
        if len(mood_history) >= 7:
            recent_avg = np.mean([entry['mood_score'] for entry in mood_history[-7:]])
            if recent_avg <= 3:
                return "intensive_support"
                
        # Emotion-based selection
        if emotion in ['anger', 'frustration']:
            return "cognitive_behavioral"
        elif emotion in ['anxiety', 'panic', 'worry']:
            return "mindfulness"
        elif emotion in ['sadness', 'depression', 'isolation']:
            return "behavioral_activation"
        else:
            return "general_wellness"
            
    def _select_intervention_strategy(self, emotion: str, mood_history: List[Dict], user_preferences: Dict) -> List[str]:
        """Select intervention strategy based on emotion and history"""
        
        strategies = []
        
        # Add CBT interventions
        if emotion in self.interventions['cognitive_behavioral']:
            strategies.extend(self.interventions['cognitive_behavioral'][emotion])
        else:
            strategies.extend(self.interventions['cognitive_behavioral'].get('general', []))
            
        # Add mindfulness interventions
        if len([entry for entry in mood_history[-7:] if entry.get('mood_score', 5) < 4]) >= 3:
            strategies.extend(self.interventions['mindfulness']['stress'])
        else:
            strategies.extend(self.interventions['mindfulness']['general'])
            
        # Add behavioral activation if low mood pattern
        recent_moods = [entry['mood_score'] for entry in mood_history[-5:]]
        if len(recent_moods) > 0 and np.mean(recent_moods) < 4:
            strategies.extend(self.interventions['behavioral_activation']['low_mood'])
            
        return strategies
        
    def _select_resources(self, intervention_type: str, user_preferences: Dict) -> List[str]:
        """Select appropriate resources based on intervention type"""
        
        if intervention_type == "crisis_intervention":
            return self.resources['crisis']
        elif user_preferences and user_preferences.get('prefers_professional_help'):
            return self.resources['therapy']
        else:
            return self.resources['self_help']
            
    def _filter_by_preferences(self, interventions: List[str], user_preferences: Dict) -> List[str]:
        """Filter interventions based on user preferences"""
        
        filtered = interventions.copy()
        
        # Filter based on preferences
        if user_preferences.get('prefers_short_activities'):
            # Prefer shorter interventions
            filtered = [i for i in filtered if 'minute' in i and int(i.split('minute')[0].split()[-1]) <= 10]
            
        if not user_preferences.get('likes_meditation'):
            filtered = [i for i in filtered if 'meditation' not in i.lower()]
            
        return filtered if filtered else interventions
        
    def _estimate_duration(self, intervention_type: str) -> str:
        """Estimate time needed for intervention"""
        
        duration_map = {
            "crisis_intervention": "Immediate - ongoing",
            "mindfulness": "10-20 minutes",
            "cognitive_behavioral": "15-30 minutes",
            "behavioral_activation": "30+ minutes",
            "general_wellness": "5-15 minutes"
        }
        
        return duration_map.get(intervention_type, "10-15 minutes")
        
    def _should_follow_up(self, mood_score: float, mood_history: List[Dict]) -> bool:
        """Determine if follow-up is recommended"""
        
        if mood_score <= 3:
            return True
            
        # Check for declining trend
        if len(mood_history) >= 5:
            recent_scores = [entry['mood_score'] for entry in mood_history[-5:]]
            if len(recent_scores) >= 3 and recent_scores[-1] < recent_scores[0] - 1:
                return True
                
        return False

class AdaptiveLearningSystem:
    def __init__(self):
        self.intervention_effectiveness = {}
        self.user_preferences = {}
        
    def record_intervention_feedback(self, user_id: int, intervention_id: str, 
                                   effectiveness_rating: float, user_feedback: str):
        """Record user feedback on intervention effectiveness"""
        
        if user_id not in self.intervention_effectiveness:
            self.intervention_effectiveness[user_id] = {}
            
        # Record the effectiveness rating for this intervention
        if intervention_id not in self.intervention_effectiveness[user_id]:
            self.intervention_effectiveness[user_id][intervention_id] = []
            
        # Store the rating and feedback
        self.intervention_effectiveness[user_id][intervention_id].append({
            'rating': effectiveness_rating,
            'feedback': user_feedback,
            'timestamp': datetime.now()
        })
        
    def update_user_preferences(self, user_id: int, preferences: Dict):
        """Update user preferences based on feedback and behavior"""
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
            
        # Update preferences
        for key, value in preferences.items():
            self.user_preferences[user_id][key] = value
            
    def get_intervention_recommendations(self, user_id: int) -> List[str]:
        """Get personalized intervention recommendations based on past effectiveness"""
        
        if user_id not in self.intervention_effectiveness:
            return []  # No data for this user yet
            
        # Calculate average effectiveness for each intervention
        intervention_ratings = {}
        for intervention_id, ratings in self.intervention_effectiveness[user_id].items():
            if ratings:
                avg_rating = sum(entry['rating'] for entry in ratings) / len(ratings)
                intervention_ratings[intervention_id] = avg_rating
                
        # Sort interventions by effectiveness (highest first)
        sorted_interventions = sorted(
            intervention_ratings.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return the top intervention IDs
        return [intervention_id for intervention_id, _ in sorted_interventions[:5]]
        
    def analyze_user_patterns(self, user_id: int, mood_history: List[Dict]) -> Dict:
        """Analyze user patterns to improve intervention recommendations"""
        
        if not mood_history or len(mood_history) < 5:
            return {'sufficient_data': False}
            
        # Extract recent mood data
        recent_moods = [entry['mood_score'] for entry in mood_history[-30:]]
        
        # Calculate statistics
        avg_mood = np.mean(recent_moods)
        std_dev = np.std(recent_moods)
        trend = 0
        
        if len(recent_moods) >= 7:
            # Calculate trend (positive or negative)
            week_avg = np.mean(recent_moods[-7:])
            prev_week_avg = np.mean(recent_moods[-14:-7]) if len(recent_moods) >= 14 else avg_mood
            trend = week_avg - prev_week_avg
            
        return {
            'sufficient_data': True,
            'average_mood': avg_mood,
            'mood_stability': std_dev,
            'recent_trend': trend,
            'weekday_patterns': self._analyze_weekday_patterns(mood_history),
            'suggested_frequency': self._suggest_check_in_frequency(recent_moods)
        }
        
    def _analyze_weekday_patterns(self, mood_history: List[Dict]) -> Dict:
        """Analyze mood patterns by day of the week"""
        
        weekday_moods = {i: [] for i in range(7)}  # 0 = Monday, 6 = Sunday
        
        for entry in mood_history:
            if 'timestamp' in entry:
                try:
                    # Convert timestamp string to datetime if needed
                    if isinstance(entry['timestamp'], str):
                        dt = datetime.fromisoformat(entry['timestamp'])
                    else:
                        dt = entry['timestamp']
                        
                    weekday = dt.weekday()
                    weekday_moods[weekday].append(entry['mood_score'])
                except (ValueError, KeyError):
                    continue
        
        # Calculate average mood for each weekday
        weekday_averages = {}
        for weekday, moods in weekday_moods.items():
            if moods:
                weekday_averages[weekday] = np.mean(moods)
                
        return weekday_averages
        
    def _suggest_check_in_frequency(self, mood_scores: List[float]) -> str:
        """Suggest how often the user should check in based on mood stability"""
        
        if not mood_scores or len(mood_scores) < 3:
            return "daily"  # Default when not enough data
            
        std_dev = np.std(mood_scores)
        
        if std_dev > 2.0:  # High volatility
            return "multiple-daily"
        elif std_dev > 1.0:  # Moderate volatility
            return "daily"
        else:  # Stable mood
            return "every-other-day"
