"""
Conversational Memory System for Persistent Farm Context
Maintains farm profiles, conversation history, and contextual information
"""

import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pymongo import MongoClient
import os

@dataclass
class FarmProfile:
    """Represents a farmer's profile and farm details"""
    farmer_id: str
    name: Optional[str]
    location: str
    farm_size: float  # in acres
    crops: List[str]
    soil_type: str
    irrigation_type: str
    farming_system: str  # organic, traditional, modern
    language_preference: str
    literacy_level: str
    phone_number: Optional[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class ConversationContext:
    """Represents context for a conversation"""
    session_id: str
    farmer_id: str
    current_topic: str
    active_workflow: Optional[str]
    mentioned_crops: List[str]
    mentioned_issues: List[str]
    seasonal_context: str
    conversation_stage: str  # greeting, problem_identification, solution_providing, follow_up

class ConversationalMemory:
    """
    Manages persistent conversational memory and farm profiles
    """
    
    def __init__(self):
        # MongoDB connection
        mongo_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
        self.client = MongoClient(mongo_uri)
        self.db = self.client.farmchat
        self.profiles_collection = self.db.farm_profiles
        self.conversations_collection = self.db.conversations
        self.context_collection = self.db.conversation_contexts
        
        # In-memory cache for active sessions
        self.active_contexts = {}
        self.session_timeout = timedelta(hours=2)
    
    def get_or_create_farmer_profile(self, identifier: str, initial_data: Dict[str, Any] = None) -> FarmProfile:
        """
        Get existing farmer profile or create new one
        
        Args:
            identifier: Phone number, user ID, or other unique identifier
            initial_data: Initial profile data if creating new profile
            
        Returns:
            FarmProfile object
        """
        # Generate farmer ID from identifier
        farmer_id = self._generate_farmer_id(identifier)
        
        # Try to find existing profile
        existing_profile = self.profiles_collection.find_one({'farmer_id': farmer_id})
        
        if existing_profile:
            return self._dict_to_farm_profile(existing_profile)
        
        # Create new profile
        new_profile = FarmProfile(
            farmer_id=farmer_id,
            name=initial_data.get('name') if initial_data else None,
            location=initial_data.get('location', 'Unknown') if initial_data else 'Unknown',
            farm_size=initial_data.get('farm_size', 0.0) if initial_data else 0.0,
            crops=initial_data.get('crops', []) if initial_data else [],
            soil_type=initial_data.get('soil_type', 'Unknown') if initial_data else 'Unknown',
            irrigation_type=initial_data.get('irrigation_type', 'Unknown') if initial_data else 'Unknown',
            farming_system=initial_data.get('farming_system', 'mixed') if initial_data else 'mixed',
            language_preference=initial_data.get('language_preference', 'en') if initial_data else 'en',
            literacy_level=initial_data.get('literacy_level', 'medium') if initial_data else 'medium',
            phone_number=initial_data.get('phone_number') if initial_data else None,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save to database
        self.profiles_collection.insert_one(asdict(new_profile))
        
        return new_profile
    
    def update_farmer_profile(self, farmer_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update farmer profile with new information
        
        Args:
            farmer_id: Farmer's unique ID
            updates: Dictionary of fields to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            updates['updated_at'] = datetime.now()
            result = self.profiles_collection.update_one(
                {'farmer_id': farmer_id},
                {'$set': updates}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating farmer profile: {e}")
            return False
    
    def get_conversation_context(self, session_id: str, farmer_id: str) -> ConversationContext:
        """
        Get or create conversation context for a session
        
        Args:
            session_id: Unique session identifier
            farmer_id: Farmer's unique ID
            
        Returns:
            ConversationContext object
        """
        # Check in-memory cache first
        if session_id in self.active_contexts:
            context = self.active_contexts[session_id]
            # Check if context is still valid (not expired)
            if datetime.now() - context.get('last_activity', datetime.now()) < self.session_timeout:
                return self._dict_to_conversation_context(context)
        
        # Try to find existing context in database
        existing_context = self.context_collection.find_one({'session_id': session_id})
        
        if existing_context and self._is_context_valid(existing_context):
            context = self._dict_to_conversation_context(existing_context)
            self.active_contexts[session_id] = asdict(context)
            return context
        
        # Create new context
        new_context = ConversationContext(
            session_id=session_id,
            farmer_id=farmer_id,
            current_topic='general',
            active_workflow=None,
            mentioned_crops=[],
            mentioned_issues=[],
            seasonal_context=self._get_current_season(),
            conversation_stage='greeting'
        )
        
        # Save to database and cache
        context_dict = asdict(new_context)
        context_dict['created_at'] = datetime.now()
        context_dict['last_activity'] = datetime.now()
        
        self.context_collection.insert_one(context_dict)
        self.active_contexts[session_id] = context_dict
        
        return new_context
    
    def update_conversation_context(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update conversation context
        
        Args:
            session_id: Session identifier
            updates: Dictionary of fields to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            updates['last_activity'] = datetime.now()
            
            # Update database
            result = self.context_collection.update_one(
                {'session_id': session_id},
                {'$set': updates}
            )
            
            # Update cache
            if session_id in self.active_contexts:
                self.active_contexts[session_id].update(updates)
            
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating conversation context: {e}")
            return False
    
    def add_conversation_turn(self, session_id: str, farmer_id: str, user_message: str, 
                           bot_response: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a conversation turn to history
        
        Args:
            session_id: Session identifier
            farmer_id: Farmer's unique ID
            user_message: User's message
            bot_response: Bot's response
            metadata: Additional metadata (tools used, reasoning chain, etc.)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            conversation_turn = {
                'session_id': session_id,
                'farmer_id': farmer_id,
                'timestamp': datetime.now(),
                'user_message': user_message,
                'bot_response': bot_response,
                'metadata': metadata or {}
            }
            
            self.conversations_collection.insert_one(conversation_turn)
            return True
        except Exception as e:
            print(f"Error saving conversation turn: {e}")
            return False
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history for a session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of turns to retrieve
            
        Returns:
            List of conversation turns
        """
        try:
            history = list(self.conversations_collection.find(
                {'session_id': session_id}
            ).sort('timestamp', -1).limit(limit))
            
            # Reverse to get chronological order
            return list(reversed(history))
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []
    
    def extract_farm_info_from_conversation(self, user_message: str, current_profile: FarmProfile) -> Dict[str, Any]:
        """
        Extract farm information from user message to update profile
        
        Args:
            user_message: User's message
            current_profile: Current farm profile
            
        Returns:
            Dictionary of extracted information
        """
        extracted_info = {}
        message_lower = user_message.lower()
        
        # Extract location
        location_keywords = ['from', 'in', 'at', 'village', 'district', 'state']
        for keyword in location_keywords:
            if keyword in message_lower:
                # Simple extraction - in real implementation, use NER
                words = user_message.split()
                try:
                    keyword_index = [w.lower() for w in words].index(keyword)
                    if keyword_index + 1 < len(words):
                        extracted_info['location'] = words[keyword_index + 1]
                except ValueError:
                    pass
        
        # Extract farm size
        size_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:acre|acres|एकड़)',
            r'(\d+(?:\.\d+)?)\s*(?:hectare|hectares|हेक्टेयर)'
        ]
        
        for pattern in size_patterns:
            import re
            match = re.search(pattern, message_lower)
            if match:
                extracted_info['farm_size'] = float(match.group(1))
                break
        
        # Extract crops
        crop_keywords = [
            'wheat', 'rice', 'cotton', 'sugarcane', 'maize', 'barley',
            'गेहूं', 'चावल', 'कपास', 'गन्ना', 'मक्का'
        ]
        
        mentioned_crops = []
        for crop in crop_keywords:
            if crop in message_lower:
                mentioned_crops.append(crop)
        
        if mentioned_crops:
            # Merge with existing crops
            existing_crops = current_profile.crops or []
            all_crops = list(set(existing_crops + mentioned_crops))
            extracted_info['crops'] = all_crops
        
        # Extract farming system
        if any(word in message_lower for word in ['organic', 'natural', 'जैविक']):
            extracted_info['farming_system'] = 'organic'
        elif any(word in message_lower for word in ['traditional', 'पारंपरिक']):
            extracted_info['farming_system'] = 'traditional'
        elif any(word in message_lower for word in ['modern', 'scientific', 'आधुनिक']):
            extracted_info['farming_system'] = 'modern'
        
        return extracted_info
    
    def build_contextual_prompt(self, user_message: str, session_id: str, farmer_id: str) -> str:
        """
        Build contextual prompt including farm profile and conversation history
        
        Args:
            user_message: Current user message
            session_id: Session identifier
            farmer_id: Farmer's unique ID
            
        Returns:
            Enhanced prompt with context
        """
        # Get farm profile
        profile = self.get_or_create_farmer_profile(farmer_id)
        
        # Get conversation context
        context = self.get_conversation_context(session_id, farmer_id)
        
        # Get recent conversation history
        history = self.get_conversation_history(session_id, limit=5)
        
        # Build contextual prompt
        prompt_parts = []
        
        # Farm profile context
        if profile.location != 'Unknown':
            prompt_parts.append(f"Farmer Location: {profile.location}")
        
        if profile.farm_size > 0:
            prompt_parts.append(f"Farm Size: {profile.farm_size} acres")
        
        if profile.crops:
            prompt_parts.append(f"Crops: {', '.join(profile.crops)}")
        
        if profile.farming_system != 'mixed':
            prompt_parts.append(f"Farming System: {profile.farming_system}")
        
        # Conversation context
        if context.current_topic != 'general':
            prompt_parts.append(f"Current Topic: {context.current_topic}")
        
        if context.active_workflow:
            prompt_parts.append(f"Active Workflow: {context.active_workflow}")
        
        if context.mentioned_issues:
            prompt_parts.append(f"Recent Issues: {', '.join(context.mentioned_issues)}")
        
        # Recent conversation history
        if history:
            prompt_parts.append("Recent Conversation:")
            for turn in history[-3:]:  # Last 3 turns
                prompt_parts.append(f"User: {turn['user_message']}")
                prompt_parts.append(f"Bot: {turn['bot_response']}")
        
        # Current message
        prompt_parts.append(f"Current Question: {user_message}")
        
        return "\n".join(prompt_parts)
    
    def _generate_farmer_id(self, identifier: str) -> str:
        """Generate unique farmer ID from identifier"""
        return hashlib.md5(identifier.encode()).hexdigest()[:12]
    
    def _dict_to_farm_profile(self, data: Dict[str, Any]) -> FarmProfile:
        """Convert dictionary to FarmProfile object"""
        return FarmProfile(**data)
    
    def _dict_to_conversation_context(self, data: Dict[str, Any]) -> ConversationContext:
        """Convert dictionary to ConversationContext object"""
        # Remove extra fields that aren't in ConversationContext
        context_fields = {
            'session_id', 'farmer_id', 'current_topic', 'active_workflow',
            'mentioned_crops', 'mentioned_issues', 'seasonal_context', 'conversation_stage'
        }
        filtered_data = {k: v for k, v in data.items() if k in context_fields}
        return ConversationContext(**filtered_data)
    
    def _is_context_valid(self, context: Dict[str, Any]) -> bool:
        """Check if conversation context is still valid"""
        last_activity = context.get('last_activity', datetime.now())
        if isinstance(last_activity, str):
            last_activity = datetime.fromisoformat(last_activity)
        
        return datetime.now() - last_activity < self.session_timeout
    
    def _get_current_season(self) -> str:
        """Get current agricultural season"""
        current_month = datetime.now().month
        
        if current_month in [11, 12, 1, 2, 3]:
            return 'rabi_season'
        elif current_month in [6, 7, 8, 9, 10]:
            return 'kharif_season'
        else:
            return 'zaid_season'
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        try:
            profile_count = self.profiles_collection.count_documents({})
            conversation_count = self.conversations_collection.count_documents({})
            active_sessions = len(self.active_contexts)
            
            return {
                'total_farmers': profile_count,
                'total_conversations': conversation_count,
                'active_sessions': active_sessions,
                'memory_efficiency': f"{active_sessions}/{profile_count}" if profile_count > 0 else "0/0"
            }
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions from memory and database"""
        try:
            # Clean up in-memory cache
            expired_sessions = []
            for session_id, context in self.active_contexts.items():
                last_activity = context.get('last_activity', datetime.now())
                if isinstance(last_activity, str):
                    last_activity = datetime.fromisoformat(last_activity)
                
                if datetime.now() - last_activity > self.session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.active_contexts[session_id]
            
            # Clean up database contexts older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.context_collection.delete_many({
                'last_activity': {'$lt': cutoff_time}
            })
            
            return len(expired_sessions)
        except Exception as e:
            print(f"Error cleaning up expired sessions: {e}")
            return 0