"""
Query Analyzer Agent
Analyzes incoming queries to determine complexity, required tools, and reasoning approach
"""

from typing import Dict, Any, List
import json
import logging
from .base_agent import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

class QueryAnalyzerAgent(BaseAgent):
    """
    Specialized agent for analyzing query complexity and determining reasoning approach
    Enhances the existing analyze_task functionality with multi-step reasoning detection
    """
    
    def __init__(self, cerebras_service):
        super().__init__(
            name="Query Analyzer",
            description="Analyzes query complexity and determines multi-step reasoning requirements"
        )
        self.cerebras_service = cerebras_service
        
    async def _process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Analyze query and determine reasoning approach"""
        
        user_message = input_data.get("user_message", "")
        conversation_history = input_data.get("conversation_history", [])
        
        # Enhanced system prompt for multi-step reasoning detection
        system_prompt = """You are an advanced agricultural query analyzer. Analyze the user's query and determine:

1. Query complexity level (simple, moderate, complex)
2. Required agricultural tools and data sources
3. Multi-step reasoning requirements
4. Expected reasoning chain depth

IMPORTANT RULES:
1. ONLY respond to agricultural queries about: farming, agriculture, crops, livestock, soil, irrigation, pesticides, fertilizers, agricultural markets, farm equipment, weather, crop diseases, etc.
2. If NOT agricultural, respond with: {"is_agricultural": false, "language": "detected_language"}
3. For agricultural queries, determine tools needed:
   - crop-price: Current/recent market prices of crops
   - web-search: Recent news, current events, new research, time-sensitive information
   - soil-health: Soil analysis, NPK levels, pH testing (if available)
   - weather-predictor: Weather forecasts, climate data (if available)
   - pest-identifier: Pest/disease identification (if available)
   - mandi-tracker: Market price comparisons (if available)
   - none: Use base LLM knowledge for general farming advice

4. Detect language: en, hi, ta, te, mr, bn, gu, kn, ml, pa
5. Determine reasoning complexity:
   - simple: Single tool or direct knowledge
   - moderate: 2-3 tools with basic correlation
   - complex: Multiple tools with multi-step analysis and synthesis

Respond in JSON format:
{
  "is_agricultural": true/false,
  "language": "en|hi|ta|te|mr|bn|gu|kn|ml|pa",
  "complexity_level": "simple|moderate|complex",
  "reasoning_chain_depth": 1-5,
  "needs_crop_price": true/false,
  "needs_web_search": true/false,
  "needs_soil_health": true/false,
  "needs_weather": true/false,
  "needs_pest_id": true/false,
  "needs_mandi_tracker": true/false,
  "crop_price_params": {"state": "...", "commodity": "...", "district": "..."},
  "search_query": "...",
  "reasoning_steps": ["step1", "step2", "step3"],
  "expected_tools_sequence": ["tool1", "tool2", "tool3"],
  "synthesis_requirements": ["correlation1", "correlation2"],
  "confidence": 0.0-1.0
}

Examples of complex queries requiring multi-step reasoning:
- "Should I plant tomatoes or wheat this season?" → soil analysis + weather + market prices + synthesis
- "My wheat has yellow spots, what should I do?" → pest identification + weather correlation + treatment options
- "Plan my irrigation for the next month" → soil moisture + weather forecast + crop requirements + scheduling

Examples of simple queries:
- "What is wheat price in Punjab?" → crop-price tool only
- "How to prepare soil for planting?" → base knowledge only"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Add conversation context if available
            if conversation_history:
                context_msg = f"Previous conversation context: {conversation_history[-2:]}"
                messages.insert(-1, {"role": "system", "content": context_msg})
            
            response = await self.cerebras_service.generate_response(messages)
            
            # Parse JSON response
            try:
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response.strip()
                
                analysis = json.loads(json_str)
                
                # Validate and enhance analysis
                analysis = self._validate_and_enhance_analysis(analysis, user_message)
                
                return AgentResult(
                    success=True,
                    data=analysis,
                    confidence=analysis.get("confidence", 0.8),
                    reasoning_steps=[],
                    metadata={
                        "query_length": len(user_message),
                        "complexity": analysis.get("complexity_level", "simple"),
                        "tools_required": len(analysis.get("expected_tools_sequence", []))
                    }
                )
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse analysis JSON: {e}, Response: {response}")
                
                # Fallback to simple analysis
                fallback_analysis = self._create_fallback_analysis(user_message)
                
                return AgentResult(
                    success=True,
                    data=fallback_analysis,
                    confidence=0.5,
                    reasoning_steps=[],
                    metadata={"fallback_used": True, "parse_error": str(e)}
                )
                
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            
            # Emergency fallback
            emergency_analysis = self._create_fallback_analysis(user_message)
            
            return AgentResult(
                success=False,
                data=emergency_analysis,
                confidence=0.3,
                reasoning_steps=[],
                error=str(e),
                metadata={"emergency_fallback": True}
            )
    
    def _validate_and_enhance_analysis(self, analysis: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """Validate and enhance the analysis with additional logic"""
        
        # Ensure required fields exist
        required_fields = ["is_agricultural", "language", "complexity_level"]
        for field in required_fields:
            if field not in analysis:
                analysis[field] = self._get_default_value(field, user_message)
        
        # Set default confidence if not provided
        if "confidence" not in analysis:
            analysis["confidence"] = 0.7
        
        # Ensure reasoning chain depth is reasonable
        if "reasoning_chain_depth" not in analysis:
            complexity = analysis.get("complexity_level", "simple")
            analysis["reasoning_chain_depth"] = {
                "simple": 1,
                "moderate": 2,
                "complex": 3
            }.get(complexity, 1)
        
        # Validate tool requirements consistency
        tools_needed = []
        if analysis.get("needs_crop_price"):
            tools_needed.append("crop-price")
        if analysis.get("needs_web_search"):
            tools_needed.append("web-search")
        if analysis.get("needs_soil_health"):
            tools_needed.append("soil-health")
        if analysis.get("needs_weather"):
            tools_needed.append("weather-predictor")
        if analysis.get("needs_pest_id"):
            tools_needed.append("pest-identifier")
        if analysis.get("needs_mandi_tracker"):
            tools_needed.append("mandi-tracker")
        
        analysis["expected_tools_sequence"] = tools_needed
        
        return analysis
    
    def _get_default_value(self, field: str, user_message: str) -> Any:
        """Get default values for missing fields"""
        defaults = {
            "is_agricultural": True,  # Assume agricultural if we got this far
            "language": "en",  # Default to English
            "complexity_level": "simple"  # Default to simple
        }
        return defaults.get(field, None)
    
    def _create_fallback_analysis(self, user_message: str) -> Dict[str, Any]:
        """Create a basic fallback analysis when parsing fails"""
        
        # Simple keyword-based analysis
        message_lower = user_message.lower()
        
        # Detect if it's agricultural
        agricultural_keywords = [
            "crop", "farm", "agriculture", "soil", "plant", "harvest", "irrigation",
            "pest", "disease", "fertilizer", "seed", "weather", "price", "market",
            "wheat", "rice", "cotton", "tomato", "potato", "mandi"
        ]
        
        is_agricultural = any(keyword in message_lower for keyword in agricultural_keywords)
        
        # Detect language (basic)
        hindi_keywords = ["क्या", "कैसे", "मेरे", "की", "है", "में"]
        language = "hi" if any(keyword in user_message for keyword in hindi_keywords) else "en"
        
        # Detect if crop price query
        price_keywords = ["price", "cost", "rate", "कीमत", "दाम", "भाव"]
        needs_crop_price = any(keyword in message_lower for keyword in price_keywords)
        
        # Detect if search needed
        search_keywords = ["latest", "news", "recent", "current", "new", "research"]
        needs_web_search = any(keyword in message_lower for keyword in search_keywords)
        
        return {
            "is_agricultural": is_agricultural,
            "language": language,
            "complexity_level": "simple",
            "reasoning_chain_depth": 1,
            "needs_crop_price": needs_crop_price,
            "needs_web_search": needs_web_search,
            "needs_soil_health": False,
            "needs_weather": False,
            "needs_pest_id": False,
            "needs_mandi_tracker": False,
            "crop_price_params": {"state": "", "commodity": "", "district": ""},
            "search_query": user_message if needs_web_search else "",
            "reasoning_steps": ["analyze_query", "provide_response"],
            "expected_tools_sequence": [],
            "synthesis_requirements": [],
            "confidence": 0.5
        }