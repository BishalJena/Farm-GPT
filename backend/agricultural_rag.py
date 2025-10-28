# Agricultural Knowledge RAG System
# Retrieval-Augmented Generation for domain-specific agricultural information

from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)

class AgriculturalRAG:
    """
    Simple RAG system for agricultural knowledge
    Stores and retrieves domain-specific agricultural information
    """
    
    def __init__(self):
        # Agricultural knowledge base (in production, this would be a vector database)
        self.knowledge_base = {
            "crop_rotation": {
                "content": "Crop rotation is the practice of growing different types of crops in the same area across different seasons. Benefits include improved soil health, reduced pest and disease pressure, and better nutrient management.",
                "keywords": ["rotation", "different crops", "seasons", "soil health"],
                "crops": ["wheat", "rice", "legumes", "cotton"]
            },
            "rabi_crops": {
                "content": "Rabi crops are winter season crops sown in October-December and harvested in March-May. Main rabi crops include wheat, barley, peas, gram, mustard. They require cool weather for growth and warm weather for ripening.",
                "keywords": ["rabi", "winter", "wheat", "barley", "cool weather"],
                "crops": ["wheat", "barley", "peas", "gram", "mustard"]
            },
            "kharif_crops": {
                "content": "Kharif crops are monsoon season crops sown in June-July and harvested in September-October. Main kharif crops include rice, cotton, sugarcane, maize, bajra. They depend on monsoon rainfall.",
                "keywords": ["kharif", "monsoon", "rice", "cotton", "rainfall"],
                "crops": ["rice", "cotton", "sugarcane", "maize", "bajra"]
            },
            "soil_health": {
                "content": "Soil health refers to the continued capacity of soil to function as a vital living ecosystem. Key indicators include organic matter content, pH level, nutrient availability (NPK), soil structure, and biological activity.",
                "keywords": ["soil", "health", "organic matter", "pH", "NPK", "nutrients"],
                "crops": ["all"]
            },
            "irrigation_methods": {
                "content": "Common irrigation methods include flood irrigation, sprinkler irrigation, drip irrigation, and furrow irrigation. Drip irrigation is most water-efficient, while flood irrigation is traditional but less efficient.",
                "keywords": ["irrigation", "water", "drip", "sprinkler", "flood", "efficient"],
                "crops": ["all"]
            },
            "pest_management": {
                "content": "Integrated Pest Management (IPM) combines biological, cultural, physical, and chemical tools to manage pests effectively while minimizing environmental impact. Prevention is better than cure.",
                "keywords": ["pest", "IPM", "management", "biological", "chemical", "prevention"],
                "crops": ["all"]
            },
            "fertilizer_application": {
                "content": "Balanced fertilizer application based on soil testing is crucial. NPK (Nitrogen, Phosphorus, Potassium) are primary nutrients. Organic fertilizers improve soil structure while chemical fertilizers provide quick nutrients.",
                "keywords": ["fertilizer", "NPK", "nitrogen", "phosphorus", "potassium", "organic"],
                "crops": ["all"]
            },
            "water_management": {
                "content": "Efficient water management includes proper irrigation scheduling, mulching to reduce evaporation, rainwater harvesting, and choosing drought-resistant varieties. Water is precious in agriculture.",
                "keywords": ["water", "irrigation", "mulching", "rainwater", "drought", "efficient"],
                "crops": ["all"]
            }
        }
        
        # Crop-specific knowledge
        self.crop_specific_knowledge = {
            "wheat": {
                "sowing_time": "October-December (Rabi season)",
                "harvesting_time": "March-May",
                "soil_requirement": "Well-drained loamy soil, pH 6.0-7.5",
                "water_requirement": "450-650mm during growing season",
                "common_pests": ["aphids", "stem borer", "rust"],
                "fertilizer_schedule": "Basal: 60kg N, 30kg P, 20kg K per hectare"
            },
            "rice": {
                "sowing_time": "June-July (Kharif season)",
                "harvesting_time": "September-October",
                "soil_requirement": "Clay or clay loam, pH 5.5-6.5",
                "water_requirement": "1200-1500mm, requires standing water",
                "common_pests": ["stem borer", "leaf folder", "brown plant hopper"],
                "fertilizer_schedule": "120kg N, 60kg P, 40kg K per hectare in splits"
            },
            "cotton": {
                "sowing_time": "April-June (Kharif season)",
                "harvesting_time": "October-January (multiple picks)",
                "soil_requirement": "Deep black cotton soil, pH 6.0-8.0",
                "water_requirement": "700-1200mm depending on variety",
                "common_pests": ["bollworm", "aphids", "whitefly"],
                "fertilizer_schedule": "150kg N, 75kg P, 75kg K per hectare"
            }
        }
    
    def retrieve_relevant_knowledge(self, query: str, crop: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant agricultural knowledge based on query"""
        
        query_lower = query.lower()
        relevant_docs = []
        
        # Search in general knowledge base
        for topic, info in self.knowledge_base.items():
            relevance_score = 0
            
            # Check keyword matches
            for keyword in info["keywords"]:
                if keyword in query_lower:
                    relevance_score += 1
            
            # Check crop relevance
            if crop and (crop in info["crops"] or "all" in info["crops"]):
                relevance_score += 0.5
            
            if relevance_score > 0:
                relevant_docs.append({
                    "topic": topic,
                    "content": info["content"],
                    "relevance_score": relevance_score,
                    "source": "general_knowledge"
                })
        
        # Search in crop-specific knowledge
        if crop and crop in self.crop_specific_knowledge:
            crop_info = self.crop_specific_knowledge[crop]
            
            # Check which aspects are relevant to the query
            aspects = {
                "sowing": ["sowing", "planting", "seeding", "when to sow"],
                "harvesting": ["harvest", "harvesting", "when to harvest", "maturity"],
                "soil": ["soil", "land", "field preparation"],
                "water": ["water", "irrigation", "watering"],
                "pest": ["pest", "insect", "disease", "problem"],
                "fertilizer": ["fertilizer", "nutrient", "feeding", "manure"]
            }
            
            for aspect, keywords in aspects.items():
                if any(keyword in query_lower for keyword in keywords):
                    aspect_key = f"{aspect}_time" if aspect in ["sowing", "harvesting"] else f"{aspect}_requirement" if aspect in ["soil", "water"] else f"common_{aspect}s" if aspect == "pest" else f"{aspect}_schedule"
                    
                    if aspect_key in crop_info:
                        relevant_docs.append({
                            "topic": f"{crop}_{aspect}",
                            "content": f"{crop.title()} {aspect}: {crop_info[aspect_key]}",
                            "relevance_score": 2.0,  # Higher score for crop-specific info
                            "source": "crop_specific"
                        })
        
        # Sort by relevance score
        relevant_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return relevant_docs[:5]  # Return top 5 most relevant documents
    
    def enhance_response_with_knowledge(self, query: str, base_response: str, crop: Optional[str] = None) -> str:
        """Enhance response with retrieved agricultural knowledge"""
        
        relevant_knowledge = self.retrieve_relevant_knowledge(query, crop)
        
        if not relevant_knowledge:
            return base_response
        
        # Build knowledge context
        knowledge_context = "\n\nRELEVANT AGRICULTURAL KNOWLEDGE:\n"
        
        for doc in relevant_knowledge:
            knowledge_context += f"- {doc['content']}\n"
        
        # Add instruction to use the knowledge
        instruction = "\n\nPlease incorporate the above agricultural knowledge into your response where relevant, ensuring accuracy and practical applicability for Indian farming conditions."
        
        return base_response + knowledge_context + instruction
    
    def get_crop_calendar(self, crop: str, region: Optional[str] = None) -> Dict[str, Any]:
        """Get crop calendar information"""
        
        if crop not in self.crop_specific_knowledge:
            return {}
        
        crop_info = self.crop_specific_knowledge[crop]
        
        calendar = {
            "crop": crop,
            "sowing_period": crop_info.get("sowing_time", "Information not available"),
            "harvesting_period": crop_info.get("harvesting_time", "Information not available"),
            "growing_season": self._determine_season(crop_info.get("sowing_time", "")),
            "duration": self._calculate_duration(crop_info.get("sowing_time", ""), crop_info.get("harvesting_time", "")),
            "soil_requirements": crop_info.get("soil_requirement", "Information not available"),
            "water_needs": crop_info.get("water_requirement", "Information not available")
        }
        
        return calendar
    
    def _determine_season(self, sowing_time: str) -> str:
        """Determine crop season based on sowing time"""
        sowing_lower = sowing_time.lower()
        
        if any(month in sowing_lower for month in ["june", "july", "august"]):
            return "Kharif (Monsoon season)"
        elif any(month in sowing_lower for month in ["october", "november", "december"]):
            return "Rabi (Winter season)"
        elif any(month in sowing_lower for month in ["march", "april", "may"]):
            return "Zaid (Summer season)"
        else:
            return "Season information not available"
    
    def _calculate_duration(self, sowing_time: str, harvesting_time: str) -> str:
        """Calculate approximate crop duration"""
        # Simple duration calculation based on typical patterns
        duration_map = {
            ("june", "september"): "3-4 months",
            ("june", "october"): "4-5 months", 
            ("october", "march"): "5-6 months",
            ("october", "april"): "6-7 months",
            ("april", "october"): "6-7 months"
        }
        
        sowing_lower = sowing_time.lower()
        harvesting_lower = harvesting_time.lower()
        
        for (sow_month, harvest_month), duration in duration_map.items():
            if sow_month in sowing_lower and harvest_month in harvesting_lower:
                return duration
        
        return "Duration information not available"