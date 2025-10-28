"""
Simplified LlamaVisionService for agricultural image analysis using OpenRouter API.
Hackathon-ready implementation with in-memory processing and agricultural prompts.
"""

import base64
import json
import logging
from typing import Dict, Any, Optional
import httpx
import asyncio
import os
from treatments_database import treatments_db

logger = logging.getLogger(__name__)


class LlamaVisionServiceError(Exception):
    """Custom exception for LlamaVisionService errors"""
    pass


class LlamaVisionService:
    """Simplified service for agricultural image analysis using OpenRouter API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.model = "meta-llama/llama-3.2-11b-vision-instruct"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.timeout = 30.0  # Reduced timeout for hackathon
        
        # Simplified HTTP client
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    def get_agricultural_prompt(self) -> str:
        """Get specialized agricultural analysis prompt"""
        return """
You are an expert agricultural advisor analyzing this farm image. Please provide a detailed analysis in JSON format with the following structure:

{
    "analysis_type": "pest|crop_health|soil|document",
    "diagnosis": "Detailed description of what you observe in the image",
    "confidence_score": 0.85,
    "severity": "low|medium|high",
    "treatment": "Specific treatment recommendations and next steps",
    "cost_estimate": "₹200-500 per acre",
    "additional_info": {
        "affected_area": "percentage or description",
        "timing": "when to apply treatment",
        "prevention": "how to prevent in future"
    }
}

Analysis Type Guidelines:
- Use "pest" for insect infestations, bug damage, or pest-related issues
- Use "crop_health" for plant diseases, fungal infections, nutrient deficiencies, or general plant health
- Use "soil" for soil quality, texture, or soil-related problems
- Use "document" for text documents or certificates

Focus on:
- Identifying pests, diseases, or crop health issues
- Providing practical, actionable advice for Indian farmers
- Including cost estimates in Indian Rupees
- Suggesting organic/sustainable solutions when possible
- Considering common crops like wheat, rice, cotton, sugarcane

If you cannot clearly identify agricultural issues, provide general crop health assessment and basic farming advice.
"""

    def encode_image_to_base64(self, image_data: bytes) -> str:
        """Encode image bytes to base64 string"""
        try:
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image to base64: {e}")
            raise LlamaVisionServiceError(f"Image encoding failed: {e}")
    
    def build_api_request(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        """Build the API request payload for OpenRouter"""
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 800,
            "temperature": 0.2,  # Low temperature for consistent analysis
            "top_p": 0.9
        }
    
    async def make_api_request(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request to OpenRouter - simplified for hackathon"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://farmchat.ai",
            "X-Title": "FarmChat Agricultural Analysis"
        }
        
        try:
            logger.info("Making OpenRouter API request")
            
            response = await self.client.post(
                self.base_url,
                headers=headers,
                json=request_payload
            )
            
            if response.status_code == 429:
                raise LlamaVisionServiceError("Rate limit exceeded. Please try again in a moment.")
            
            if response.status_code != 200:
                error_msg = f"API request failed with status {response.status_code}"
                logger.error(error_msg)
                raise LlamaVisionServiceError(error_msg)
            
            response_data = response.json()
            logger.info("OpenRouter API request successful")
            return response_data
            
        except httpx.TimeoutException:
            raise LlamaVisionServiceError("Request timeout. Please try again.")
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise LlamaVisionServiceError(f"Network error: {e}")
    
    def parse_analysis_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenRouter API response and extract analysis results"""
        try:
            # Extract the content from the response
            choices = response_data.get("choices", [])
            if not choices:
                raise LlamaVisionServiceError("No choices in API response")
            
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                raise LlamaVisionServiceError("No content in API response")
            
            logger.info(f"Raw API response content: {content}")
            
            # Try to parse JSON from the response
            try:
                # Look for JSON in the response (it might be wrapped in markdown code blocks)
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    analysis_data = json.loads(json_str)
                else:
                    # If no JSON found, create a structured response from the text
                    analysis_data = {
                        "analysis_type": "general_agriculture",
                        "diagnosis": content[:300] + "..." if len(content) > 300 else content,
                        "confidence_score": 0.7,
                        "severity": "medium",
                        "treatment": "Please consult with an agricultural expert for specific treatment recommendations.",
                        "cost_estimate": "₹200-500 per acre",
                        "additional_info": {}
                    }
            
            except json.JSONDecodeError:
                # Fallback: create structured response from text content
                analysis_data = {
                    "analysis_type": "general_agriculture",
                    "diagnosis": content[:300] + "..." if len(content) > 300 else content,
                    "confidence_score": 0.7,
                    "severity": "medium",
                    "treatment": "Please consult with an agricultural expert for specific treatment recommendations.",
                    "cost_estimate": "₹200-500 per acre",
                    "additional_info": {}
                }
            
            # Validate and normalize the response
            # Map AI response analysis types to valid types
            analysis_type_mapping = {
                "disease_identification": "crop_health",
                "pest_identification": "pest", 
                "pest_analysis": "pest",
                "soil_analysis": "soil",
                "soil_health": "soil",
                "crop_health": "crop_health",
                "crop_disease": "crop_health",
                "document_analysis": "document",
                "general_agriculture": "crop_health"
            }
            
            raw_analysis_type = analysis_data.get("analysis_type", "general_agriculture")
            mapped_analysis_type = analysis_type_mapping.get(raw_analysis_type, "crop_health")
            logger.info(f"Analysis completed successfully: {raw_analysis_type} -> {mapped_analysis_type}")
            analysis_data["analysis_type"] = mapped_analysis_type
            
            analysis_data["diagnosis"] = analysis_data.get("diagnosis", "Analysis completed")
            analysis_data["confidence_score"] = max(0.0, min(1.0, float(analysis_data.get("confidence_score", 0.7))))
            
            severity = analysis_data.get("severity", "medium").lower()
            if severity not in ["low", "medium", "high"]:
                severity = "medium"
            analysis_data["severity"] = severity
            
            analysis_data["treatment"] = analysis_data.get("treatment", "Consult agricultural expert")
            analysis_data["cost_estimate"] = analysis_data.get("cost_estimate", "₹200-500 per acre")
            analysis_data["additional_info"] = analysis_data.get("additional_info", {})
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            raise LlamaVisionServiceError(f"Response parsing failed: {e}")
    
    async def analyze_agricultural_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze agricultural image using Llama Vision model
        
        Args:
            image_data: Raw image bytes (processed in memory only)
            
        Returns:
            Dict with analysis results
        """
        try:
            # Encode image to base64
            image_base64 = self.encode_image_to_base64(image_data)
            
            # Get agricultural analysis prompt
            prompt = self.get_agricultural_prompt()
            
            # Build API request
            request_payload = self.build_api_request(image_base64, prompt)
            
            # Make API request
            response_data = await self.make_api_request(request_payload)
            
            # Parse response
            result = self.parse_analysis_response(response_data)
            
            # Enhance with treatment recommendations from database
            analysis_type = result.get('analysis_type', 'general_agriculture')
            severity = result.get('severity', 'medium')
            
            # Get enhanced treatment recommendations
            treatment_data = treatments_db.get_treatment_recommendation(analysis_type, result.get('diagnosis', ''))
            
            # Update result with enhanced data
            result['treatment'] = treatment_data['treatment']
            result['cost_estimate'] = treatments_db.get_cost_estimate(analysis_type, severity)
            result['additional_info'].update({
                'organic_alternative': treatment_data['organic_alternative'],
                'prevention': treatment_data['prevention'],
                'timing': treatment_data['timing'],
                'suppliers': treatment_data['suppliers']
            })
            
            logger.info(f"Analysis completed successfully: {analysis_type}")
            return result
            
        except Exception as e:
            logger.error(f"Agricultural image analysis failed: {e}")
            # Return fallback response with treatment database
            fallback_treatment = treatments_db.get_treatment_recommendation("crop_health")
            return {
                "analysis_type": "general_agriculture",
                "diagnosis": f"Unable to analyze image: {str(e)}. Please try uploading a clearer image of your crops, soil, or agricultural area.",
                "confidence_score": 0.3,
                "severity": "medium",
                "treatment": fallback_treatment['treatment'],
                "cost_estimate": fallback_treatment['cost_estimate'],
                "additional_info": {
                    "error": "Analysis failed",
                    "suggestion": "Try uploading a well-lit, clear image of the affected area",
                    "organic_alternative": fallback_treatment['organic_alternative'],
                    "prevention": fallback_treatment['prevention'],
                    "timing": fallback_treatment['timing'],
                    "suppliers": fallback_treatment['suppliers']
                }
            }