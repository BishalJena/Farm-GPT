"""
Data Synthesizer Agent
Combines and correlates data from multiple sources for comprehensive analysis
"""

from typing import Dict, Any, List
import json
import logging
from .base_agent import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

class DataSynthesizerAgent(BaseAgent):
    """
    Specialized agent for synthesizing data from multiple agricultural tools
    Correlates weather, soil, market, and pest data for comprehensive insights
    """
    
    def __init__(self, cerebras_service):
        super().__init__(
            name="Data Synthesizer",
            description="Synthesizes and correlates data from multiple agricultural sources"
        )
        self.cerebras_service = cerebras_service
        
    async def _process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Synthesize data from multiple tools"""
        
        query = input_data.get("query", "")
        tool_results = input_data.get("tool_results", {})
        user_context = input_data.get("context", {})
        
        # Enhanced system prompt for data synthesis
        system_prompt = """You are an expert agricultural data synthesizer. Your job is to:

1. Analyze data from multiple agricultural tools
2. Find correlations and patterns across different data sources
3. Identify conflicts or inconsistencies in the data
4. Synthesize insights that wouldn't be apparent from individual tools
5. Provide integrated recommendations based on all available data

Available data sources may include:
- Crop prices and market trends
- Weather forecasts and climate data
- Soil health analysis (NPK, pH, organic matter)
- Pest/disease identification results
- Web search results for current agricultural news
- Mandi price comparisons

SYNTHESIS RULES:
1. Look for correlations (e.g., weather impact on pest activity, soil conditions affecting crop choice)
2. Identify timing considerations (e.g., market prices vs planting seasons)
3. Flag any conflicting recommendations from different sources
4. Consider regional and seasonal context
5. Prioritize actionable insights over raw data

Respond in JSON format:
{
  "synthesis_summary": "Brief overview of key insights from all data sources",
  "correlations_found": [
    {
      "sources": ["tool1", "tool2"],
      "correlation": "Description of how data relates",
      "impact": "What this means for the farmer",
      "confidence": 0.0-1.0
    }
  ],
  "conflicts_identified": [
    {
      "conflicting_sources": ["tool1", "tool2"],
      "conflict_description": "What data conflicts",
      "resolution": "How to resolve or which to prioritize"
    }
  ],
  "integrated_insights": [
    {
      "insight": "Key insight from data synthesis",
      "supporting_data": ["source1", "source2"],
      "actionability": "high|medium|low",
      "timeline": "immediate|short_term|long_term"
    }
  ],
  "risk_factors": [
    {
      "risk": "Identified risk from data analysis",
      "probability": "high|medium|low",
      "mitigation": "Suggested mitigation strategy"
    }
  ],
  "confidence_score": 0.0-1.0,
  "data_quality_assessment": {
    "completeness": 0.0-1.0,
    "reliability": 0.0-1.0,
    "timeliness": 0.0-1.0
  }
}"""

        try:
            # Prepare tool results summary for the LLM
            tools_summary = self._prepare_tools_summary(tool_results)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""
Query: {query}

Available Data:
{tools_summary}

User Context:
{json.dumps(user_context, indent=2)}

Please synthesize this data and provide integrated insights.
"""}
            ]
            
            response = await self.cerebras_service.generate_response(messages)
            
            # Parse JSON response
            try:
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response.strip()
                
                synthesis = json.loads(json_str)
                
                # Validate and enhance synthesis
                synthesis = self._validate_synthesis(synthesis, tool_results)
                
                return AgentResult(
                    success=True,
                    data=synthesis,
                    confidence=synthesis.get("confidence_score", 0.8),
                    reasoning_steps=self._extract_reasoning_steps(synthesis),
                    metadata={
                        "tools_synthesized": list(tool_results.keys()),
                        "correlations_count": len(synthesis.get("correlations_found", [])),
                        "insights_count": len(synthesis.get("integrated_insights", []))
                    }
                )
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse synthesis JSON: {e}, Response: {response}")
                
                # Fallback to simple synthesis
                fallback_synthesis = self._create_fallback_synthesis(tool_results, query)
                
                return AgentResult(
                    success=True,
                    data=fallback_synthesis,
                    confidence=0.6,
                    reasoning_steps=[],
                    metadata={"fallback_used": True, "parse_error": str(e)}
                )
                
        except Exception as e:
            logger.error(f"Error in data synthesis: {e}")
            
            # Emergency fallback
            emergency_synthesis = self._create_fallback_synthesis(tool_results, query)
            
            return AgentResult(
                success=False,
                data=emergency_synthesis,
                confidence=0.4,
                reasoning_steps=[],
                error=str(e),
                metadata={"emergency_fallback": True}
            )
    
    def _prepare_tools_summary(self, tool_results: Dict[str, Any]) -> str:
        """Prepare a formatted summary of tool results for the LLM"""
        summary_parts = []
        
        for tool_name, result in tool_results.items():
            summary_parts.append(f"\n{tool_name.upper()} DATA:")
            
            if isinstance(result, dict):
                for key, value in result.items():
                    summary_parts.append(f"  {key}: {value}")
            else:
                summary_parts.append(f"  {result}")
        
        return "\n".join(summary_parts)
    
    def _validate_synthesis(self, synthesis: Dict[str, Any], tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance the synthesis results"""
        
        # Ensure required fields exist
        required_fields = [
            "synthesis_summary", "correlations_found", "conflicts_identified",
            "integrated_insights", "risk_factors", "confidence_score"
        ]
        
        for field in required_fields:
            if field not in synthesis:
                synthesis[field] = self._get_default_synthesis_value(field)
        
        # Validate confidence score
        if not isinstance(synthesis["confidence_score"], (int, float)) or not (0 <= synthesis["confidence_score"] <= 1):
            synthesis["confidence_score"] = 0.7
        
        # Ensure data quality assessment exists
        if "data_quality_assessment" not in synthesis:
            synthesis["data_quality_assessment"] = {
                "completeness": len(tool_results) / 5.0,  # Assume 5 possible tools
                "reliability": 0.8,  # Default reliability
                "timeliness": 0.9   # Assume recent data
            }
        
        return synthesis
    
    def _get_default_synthesis_value(self, field: str) -> Any:
        """Get default values for missing synthesis fields"""
        defaults = {
            "synthesis_summary": "Data synthesis completed with available information",
            "correlations_found": [],
            "conflicts_identified": [],
            "integrated_insights": [],
            "risk_factors": [],
            "confidence_score": 0.7
        }
        return defaults.get(field, None)
    
    def _create_fallback_synthesis(self, tool_results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Create a basic fallback synthesis when AI processing fails"""
        
        # Simple rule-based synthesis
        insights = []
        correlations = []
        risks = []
        
        # Basic correlations based on available tools
        if "weather" in tool_results and "crop-price" in tool_results:
            correlations.append({
                "sources": ["weather", "crop-price"],
                "correlation": "Weather conditions may affect crop prices",
                "impact": "Monitor weather for market timing decisions",
                "confidence": 0.7
            })
        
        if "soil-health" in tool_results and "crop-price" in tool_results:
            insights.append({
                "insight": "Soil conditions should be considered alongside market prices for crop selection",
                "supporting_data": ["soil-health", "crop-price"],
                "actionability": "high",
                "timeline": "immediate"
            })
        
        # Basic risk assessment
        if "pest-identifier" in tool_results:
            pest_data = tool_results["pest-identifier"]
            if isinstance(pest_data, dict) and pest_data.get("severity") in ["high", "severe"]:
                risks.append({
                    "risk": "Pest infestation detected",
                    "probability": "high",
                    "mitigation": "Immediate treatment recommended"
                })
        
        return {
            "synthesis_summary": f"Analyzed {len(tool_results)} data sources for agricultural insights",
            "correlations_found": correlations,
            "conflicts_identified": [],
            "integrated_insights": insights,
            "risk_factors": risks,
            "confidence_score": 0.6,
            "data_quality_assessment": {
                "completeness": min(len(tool_results) / 3.0, 1.0),
                "reliability": 0.7,
                "timeliness": 0.8
            }
        }
    
    def _extract_reasoning_steps(self, synthesis: Dict[str, Any]) -> List[str]:
        """Extract reasoning steps from synthesis results"""
        steps = []
        
        if synthesis.get("correlations_found"):
            steps.append("Identified correlations between data sources")
        
        if synthesis.get("conflicts_identified"):
            steps.append("Detected and resolved data conflicts")
        
        if synthesis.get("integrated_insights"):
            steps.append("Generated integrated insights from multiple sources")
        
        if synthesis.get("risk_factors"):
            steps.append("Assessed risks based on combined data")
        
        return steps or ["Performed basic data synthesis"]