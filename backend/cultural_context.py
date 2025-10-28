"""
Cultural Context Manager for Multilingual Agricultural Advisory
Handles cultural adaptation, language detection, and context-aware responses
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class CulturalContext:
    """Represents cultural context for a user interaction"""
    language: str
    region: str
    literacy_level: str
    farming_system: str
    seasonal_context: str
    cultural_terms: Dict[str, str]

class CulturalContextManager:
    """
    Manages cultural context and language adaptation for agricultural advisory
    """
    
    def __init__(self):
        self.supported_languages = {
            'hi': 'Hindi',
            'pa': 'Punjabi', 
            'en': 'English',
            'bn': 'Bengali',
            'te': 'Telugu',
            'ta': 'Tamil',
            'gu': 'Gujarati',
            'mr': 'Marathi',
            'kn': 'Kannada',
            'ml': 'Malayalam'
        }
        
        self.agricultural_terms = self._load_agricultural_terms()
        self.seasonal_calendar = self._load_seasonal_calendar()
        self.regional_practices = self._load_regional_practices()
    
    def detect_language_and_context(self, text: str, user_profile: Dict[str, Any] = None) -> CulturalContext:
        """
        Detect language and infer cultural context from user input
        
        Args:
            text: User input text
            user_profile: Optional user profile with location, preferences
            
        Returns:
            CulturalContext object with detected/inferred context
        """
        # Language detection
        language = self._detect_language(text)
        
        # Region inference
        region = self._infer_region(text, user_profile)
        
        # Literacy level detection
        literacy_level = self._detect_literacy_level(text, language)
        
        # Farming system inference
        farming_system = self._infer_farming_system(text, region)
        
        # Seasonal context
        seasonal_context = self._get_seasonal_context(region)
        
        # Get cultural terms for the language/region
        cultural_terms = self._get_cultural_terms(language, region)
        
        return CulturalContext(
            language=language,
            region=region,
            literacy_level=literacy_level,
            farming_system=farming_system,
            seasonal_context=seasonal_context,
            cultural_terms=cultural_terms
        )
    
    def adapt_response(self, response: str, context: CulturalContext, query_type: str = 'general') -> str:
        """
        Adapt response based on cultural context
        
        Args:
            response: Original response text
            context: Cultural context for adaptation
            query_type: Type of query (crop, weather, price, etc.)
            
        Returns:
            Culturally adapted response
        """
        adapted_response = response
        
        # Language-specific adaptations
        if context.language != 'en':
            adapted_response = self._translate_agricultural_terms(adapted_response, context)
        
        # Literacy level adaptations
        if context.literacy_level == 'low':
            adapted_response = self._simplify_language(adapted_response, context.language)
        
        # Regional practice adaptations
        adapted_response = self._add_regional_context(adapted_response, context)
        
        # Seasonal adaptations
        adapted_response = self._add_seasonal_context(adapted_response, context)
        
        # Cultural politeness and formatting
        adapted_response = self._add_cultural_politeness(adapted_response, context)
        
        return adapted_response
    
    def _detect_language(self, text: str) -> str:
        """Detect language from text patterns"""
        # Hindi patterns
        hindi_patterns = [
            r'[\u0900-\u097F]',  # Devanagari script
            r'\b(kya|hai|mein|aur|ke|ki|ko|se|me|par)\b',  # Common Hindi words
            r'\b(fasal|khet|kheti|pani|mitti)\b'  # Agricultural Hindi terms
        ]
        
        # Punjabi patterns  
        punjabi_patterns = [
            r'[\u0A00-\u0A7F]',  # Gurmukhi script
            r'\b(ki|hai|te|da|de|nu|ch|nal)\b',  # Common Punjabi words
            r'\b(fasal|khet|kheti|pani|mitti)\b'  # Agricultural Punjabi terms
        ]
        
        # Check for script patterns first
        for pattern in hindi_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 'hi'
        
        for pattern in punjabi_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 'pa'
        
        # Default to English if no patterns match
        return 'en'
    
    def _infer_region(self, text: str, user_profile: Dict[str, Any] = None) -> str:
        """Infer region from text and user profile"""
        if user_profile and 'location' in user_profile:
            return user_profile['location']
        
        # Region keywords
        region_keywords = {
            'punjab': ['punjab', 'ludhiana', 'amritsar', 'jalandhar', 'patiala'],
            'haryana': ['haryana', 'gurgaon', 'faridabad', 'panipat', 'karnal'],
            'uttar_pradesh': ['uttar pradesh', 'up', 'lucknow', 'kanpur', 'agra'],
            'maharashtra': ['maharashtra', 'mumbai', 'pune', 'nagpur', 'nashik'],
            'gujarat': ['gujarat', 'ahmedabad', 'surat', 'vadodara', 'rajkot'],
            'rajasthan': ['rajasthan', 'jaipur', 'jodhpur', 'udaipur', 'kota']
        }
        
        text_lower = text.lower()
        for region, keywords in region_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return region
        
        return 'north_india'  # Default region
    
    def _detect_literacy_level(self, text: str, language: str) -> str:
        """Detect literacy level from text complexity"""
        # Simple heuristics for literacy level detection
        word_count = len(text.split())
        
        # Check for complex sentence structures
        complex_indicators = [
            r'\b(however|therefore|consequently|furthermore|moreover)\b',
            r'\b(because|although|whereas|nevertheless)\b'
        ]
        
        has_complex_structure = any(re.search(pattern, text, re.IGNORECASE) for pattern in complex_indicators)
        
        if word_count > 20 and has_complex_structure:
            return 'high'
        elif word_count > 10:
            return 'medium'
        else:
            return 'low'
    
    def _infer_farming_system(self, text: str, region: str) -> str:
        """Infer farming system from text and region"""
        # Farming system keywords
        system_keywords = {
            'organic': ['organic', 'natural', 'chemical-free', 'bio'],
            'traditional': ['traditional', 'desi', 'purani', 'old'],
            'modern': ['modern', 'technology', 'machine', 'scientific'],
            'mixed': ['mixed', 'combination', 'both']
        }
        
        text_lower = text.lower()
        for system, keywords in system_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return system
        
        # Regional defaults
        regional_defaults = {
            'punjab': 'modern',
            'haryana': 'modern', 
            'kerala': 'organic',
            'himachal_pradesh': 'traditional'
        }
        
        return regional_defaults.get(region, 'mixed')
    
    def _get_seasonal_context(self, region: str) -> str:
        """Get current seasonal context for the region"""
        current_month = datetime.now().month
        
        # Seasonal calendar for different regions
        if region in ['punjab', 'haryana', 'uttar_pradesh']:
            if current_month in [11, 12, 1, 2, 3]:
                return 'rabi_season'
            elif current_month in [6, 7, 8, 9, 10]:
                return 'kharif_season'
            else:
                return 'zaid_season'
        else:
            # General seasonal context
            if current_month in [11, 12, 1, 2]:
                return 'winter_season'
            elif current_month in [3, 4, 5]:
                return 'summer_season'
            else:
                return 'monsoon_season'
    
    def _get_cultural_terms(self, language: str, region: str) -> Dict[str, str]:
        """Get cultural terms for language and region"""
        return self.agricultural_terms.get(language, {}).get(region, {})
    
    def _translate_agricultural_terms(self, text: str, context: CulturalContext) -> str:
        """Translate agricultural terms to local language"""
        terms = context.cultural_terms
        
        for english_term, local_term in terms.items():
            # Replace English terms with local equivalents
            text = re.sub(r'\b' + re.escape(english_term) + r'\b', local_term, text, flags=re.IGNORECASE)
        
        return text
    
    def _simplify_language(self, text: str, language: str) -> str:
        """Simplify language for low literacy users"""
        # Replace complex words with simpler alternatives
        simplifications = {
            'en': {
                'recommendation': 'advice',
                'application': 'use',
                'fertilizer': 'khad',
                'pesticide': 'dawa',
                'irrigation': 'pani dena'
            },
            'hi': {
                'अनुशंसा': 'सलाह',
                'प्रयोग': 'इस्तेमाल',
                'उर्वरक': 'खाद'
            }
        }
        
        if language in simplifications:
            for complex_word, simple_word in simplifications[language].items():
                text = re.sub(r'\b' + re.escape(complex_word) + r'\b', simple_word, text, flags=re.IGNORECASE)
        
        return text
    
    def _add_regional_context(self, text: str, context: CulturalContext) -> str:
        """Add regional context to response"""
        regional_additions = {
            'punjab': 'पंजाब के मौसम के अनुसार',
            'haryana': 'हरियाणा की मिट्टी के लिए',
            'maharashtra': 'महाराष्ट्र की जलवायु में'
        }
        
        if context.region in regional_additions and context.language == 'hi':
            addition = regional_additions[context.region]
            text = f"{addition}, {text}"
        
        return text
    
    def _add_seasonal_context(self, text: str, context: CulturalContext) -> str:
        """Add seasonal context to response"""
        seasonal_prefixes = {
            'rabi_season': 'रबी के मौसम में',
            'kharif_season': 'खरीफ के मौसम में',
            'zaid_season': 'जायद के मौसम में'
        }
        
        if context.seasonal_context in seasonal_prefixes and context.language == 'hi':
            prefix = seasonal_prefixes[context.seasonal_context]
            text = f"{prefix} {text}"
        
        return text
    
    def _add_cultural_politeness(self, text: str, context: CulturalContext) -> str:
        """Add cultural politeness markers"""
        if context.language == 'hi':
            # Add respectful greeting
            if not text.startswith(('नमस्ते', 'आदाब')):
                text = f"नमस्ते किसान भाई, {text}"
            
            # Add respectful closing
            if not text.endswith(('धन्यवाद', 'जय किसान')):
                text = f"{text} जय किसान!"
        
        elif context.language == 'pa':
            # Add Punjabi politeness
            if not text.startswith(('ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'ਨਮਸਕਾਰ')):
                text = f"ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਜੀ, {text}"
        
        return text
    
    def _load_agricultural_terms(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Load agricultural terms dictionary"""
        return {
            'hi': {
                'punjab': {
                    'crop': 'फसल',
                    'soil': 'मिट्टी',
                    'water': 'पानी',
                    'fertilizer': 'खाद',
                    'pesticide': 'कीटनाशक',
                    'seed': 'बीज',
                    'harvest': 'फसल काटना',
                    'irrigation': 'सिंचाई',
                    'weather': 'मौसम',
                    'market': 'मंडी',
                    'price': 'भाव',
                    'wheat': 'गेहूं',
                    'rice': 'चावल',
                    'cotton': 'कपास',
                    'sugarcane': 'गन्ना'
                }
            },
            'pa': {
                'punjab': {
                    'crop': 'ਫਸਲ',
                    'soil': 'ਮਿੱਟੀ',
                    'water': 'ਪਾਣੀ',
                    'fertilizer': 'ਖਾਦ',
                    'seed': 'ਬੀਜ',
                    'wheat': 'ਕਣਕ',
                    'rice': 'ਚਾਵਲ'
                }
            }
        }
    
    def _load_seasonal_calendar(self) -> Dict[str, Dict[str, List[str]]]:
        """Load seasonal calendar for different regions"""
        return {
            'punjab': {
                'rabi': ['wheat', 'barley', 'mustard', 'gram'],
                'kharif': ['rice', 'cotton', 'sugarcane', 'maize'],
                'zaid': ['fodder', 'vegetables']
            }
        }
    
    def _load_regional_practices(self) -> Dict[str, Dict[str, Any]]:
        """Load regional farming practices"""
        return {
            'punjab': {
                'main_crops': ['wheat', 'rice', 'cotton'],
                'soil_type': 'alluvial',
                'irrigation': 'canal_tubewell',
                'common_issues': ['water_scarcity', 'soil_degradation']
            }
        }
    
    def get_language_stats(self) -> Dict[str, Any]:
        """Get statistics about language usage"""
        return {
            'supported_languages': len(self.supported_languages),
            'agricultural_terms_count': sum(
                len(terms) for lang_terms in self.agricultural_terms.values() 
                for terms in lang_terms.values()
            ),
            'regional_coverage': list(self.regional_practices.keys())
        }