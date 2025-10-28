"""
Static treatments database for agricultural recommendations
Hackathon-ready mock data for common crop issues
"""

from typing import Dict, List, Any
import random

class TreatmentsDatabase:
    """Static database of agricultural treatments and recommendations"""
    
    def __init__(self):
        self.treatments = {
            "pest_detection": {
                "aphids": {
                    "treatment": "Apply neem oil spray (5ml per liter water) or use insecticidal soap. Spray early morning or evening.",
                    "cost_range": "₹150-300 per acre",
                    "organic_alternative": "Mix garlic and chili spray, or introduce ladybugs as natural predators",
                    "prevention": "Regular monitoring, avoid over-fertilization with nitrogen",
                    "timing": "Apply every 7-10 days until infestation is controlled"
                },
                "bollworm": {
                    "treatment": "Use Bt cotton varieties or apply Bacillus thuringiensis spray. For severe cases, use approved insecticides.",
                    "cost_range": "₹400-800 per acre",
                    "organic_alternative": "Pheromone traps and neem-based products",
                    "prevention": "Crop rotation, intercropping with marigold or basil",
                    "timing": "Monitor during flowering stage, apply at first sign of larvae"
                },
                "whitefly": {
                    "treatment": "Yellow sticky traps and neem oil application. Use reflective mulch to deter flies.",
                    "cost_range": "₹200-400 per acre",
                    "organic_alternative": "Soap spray solution and companion planting with basil",
                    "prevention": "Remove weeds, avoid excessive nitrogen fertilization",
                    "timing": "Early detection is key, apply treatments weekly"
                }
            },
            "disease_identification": {
                "leaf_blight": {
                    "treatment": "Apply copper-based fungicide or mancozeb. Ensure proper drainage and air circulation.",
                    "cost_range": "₹300-600 per acre",
                    "organic_alternative": "Baking soda spray (1 tsp per liter) or compost tea application",
                    "prevention": "Avoid overhead watering, practice crop rotation",
                    "timing": "Apply at first symptoms, repeat every 10-14 days"
                },
                "powdery_mildew": {
                    "treatment": "Sulfur-based fungicide or potassium bicarbonate spray. Improve air circulation.",
                    "cost_range": "₹250-500 per acre",
                    "organic_alternative": "Milk spray (1:10 ratio with water) or neem oil",
                    "prevention": "Avoid overcrowding plants, water at soil level",
                    "timing": "Apply preventively during humid conditions"
                },
                "bacterial_wilt": {
                    "treatment": "Remove infected plants immediately. Apply copper sulfate to soil. Use resistant varieties.",
                    "cost_range": "₹400-700 per acre",
                    "organic_alternative": "Biocontrol agents like Pseudomonas fluorescens",
                    "prevention": "Soil solarization, avoid waterlogging",
                    "timing": "Immediate action required upon detection"
                }
            },
            "soil_analysis": {
                "nitrogen_deficiency": {
                    "treatment": "Apply urea (46-0-0) at 100-150 kg/hectare or use organic compost.",
                    "cost_range": "₹800-1200 per acre",
                    "organic_alternative": "Vermicompost, green manure, or nitrogen-fixing cover crops",
                    "prevention": "Regular soil testing, balanced fertilization",
                    "timing": "Apply during active growth periods"
                },
                "phosphorus_deficiency": {
                    "treatment": "Apply DAP (18-46-0) or single super phosphate at recommended rates.",
                    "cost_range": "₹600-1000 per acre",
                    "organic_alternative": "Bone meal, rock phosphate, or compost",
                    "prevention": "Maintain soil pH between 6.0-7.0 for optimal P availability",
                    "timing": "Apply before planting or during early growth"
                },
                "potassium_deficiency": {
                    "treatment": "Apply muriate of potash (0-0-60) or sulfate of potash.",
                    "cost_range": "₹500-900 per acre",
                    "organic_alternative": "Wood ash, kelp meal, or banana peel compost",
                    "prevention": "Regular soil testing and balanced NPK application",
                    "timing": "Apply during fruit development stage"
                },
                "acidic_soil": {
                    "treatment": "Apply agricultural lime at 1-2 tons per hectare to raise pH.",
                    "cost_range": "₹300-600 per acre",
                    "organic_alternative": "Wood ash or crushed eggshells for small areas",
                    "prevention": "Regular pH monitoring, avoid over-use of acidic fertilizers",
                    "timing": "Apply 2-3 months before planting"
                }
            },
            "crop_health": {
                "stunted_growth": {
                    "treatment": "Soil test for nutrient deficiencies. Apply balanced NPK fertilizer and improve drainage.",
                    "cost_range": "₹400-800 per acre",
                    "organic_alternative": "Compost, vermicompost, and mycorrhizal inoculants",
                    "prevention": "Proper soil preparation, adequate spacing, regular monitoring",
                    "timing": "Address immediately upon detection"
                },
                "yellowing_leaves": {
                    "treatment": "Check for nitrogen deficiency or waterlogging. Apply appropriate fertilizer or improve drainage.",
                    "cost_range": "₹300-700 per acre",
                    "organic_alternative": "Compost tea, fish emulsion, or seaweed extract",
                    "prevention": "Balanced nutrition, proper irrigation management",
                    "timing": "Act quickly to prevent further yellowing"
                },
                "poor_flowering": {
                    "treatment": "Apply phosphorus-rich fertilizer and ensure adequate pollination support.",
                    "cost_range": "₹350-650 per acre",
                    "organic_alternative": "Bone meal, rock phosphate, and encourage beneficial insects",
                    "prevention": "Balanced nutrition, avoid excessive nitrogen during flowering",
                    "timing": "Apply before flowering stage begins"
                }
            }
        }
        
        self.suppliers = [
            {
                "name": "Punjab Agri Store",
                "contact": "+91-98765-43210",
                "location": "Ludhiana, Punjab",
                "specialties": ["Pesticides", "Fertilizers", "Seeds"],
                "rating": 4.8
            },
            {
                "name": "Farmer's Choice",
                "contact": "+91-98765-43211", 
                "location": "Amritsar, Punjab",
                "specialties": ["Organic Solutions", "Bio-fertilizers", "Equipment"],
                "rating": 4.6
            },
            {
                "name": "Krishi Kendra",
                "contact": "+91-98765-43212",
                "location": "Jalandhar, Punjab", 
                "specialties": ["Government Schemes", "Soil Testing", "Training"],
                "rating": 4.9
            },
            {
                "name": "Green Valley Supplies",
                "contact": "+91-98765-43213",
                "location": "Patiala, Punjab",
                "specialties": ["Organic Inputs", "Micronutrients", "Growth Promoters"],
                "rating": 4.7
            },
            {
                "name": "Modern Agri Solutions",
                "contact": "+91-98765-43214",
                "location": "Bathinda, Punjab",
                "specialties": ["Precision Agriculture", "Drones", "Smart Irrigation"],
                "rating": 4.5
            }
        ]
    
    def get_treatment_recommendation(self, analysis_type: str, issue_detected: str = None) -> Dict[str, Any]:
        """Get treatment recommendation based on analysis type and detected issue"""
        
        # If specific issue is detected, try to find exact match
        if issue_detected and analysis_type in self.treatments:
            issue_key = issue_detected.lower().replace(" ", "_")
            for key, treatment in self.treatments[analysis_type].items():
                if key in issue_key or issue_key in key:
                    return self._format_treatment_response(treatment, analysis_type)
        
        # Fallback to random treatment from category
        if analysis_type in self.treatments:
            treatments_list = list(self.treatments[analysis_type].values())
            if treatments_list:
                selected_treatment = random.choice(treatments_list)
                return self._format_treatment_response(selected_treatment, analysis_type)
        
        # Generic fallback
        return {
            "treatment": "Please consult with a local agricultural expert for specific treatment recommendations based on your crop and local conditions.",
            "cost_estimate": "₹200-500 per acre",
            "organic_alternative": "Consider organic and sustainable farming practices",
            "prevention": "Regular monitoring and preventive measures",
            "timing": "Consult expert for optimal timing",
            "suppliers": self.get_nearby_suppliers()
        }
    
    def _format_treatment_response(self, treatment_data: Dict[str, str], analysis_type: str) -> Dict[str, Any]:
        """Format treatment data into response format"""
        return {
            "treatment": treatment_data["treatment"],
            "cost_estimate": treatment_data["cost_range"],
            "organic_alternative": treatment_data["organic_alternative"],
            "prevention": treatment_data["prevention"],
            "timing": treatment_data["timing"],
            "suppliers": self.get_nearby_suppliers(analysis_type)
        }
    
    def get_nearby_suppliers(self, analysis_type: str = None) -> List[Dict[str, Any]]:
        """Get list of nearby suppliers with contact information"""
        # Filter suppliers based on analysis type if provided
        if analysis_type:
            relevant_suppliers = []
            for supplier in self.suppliers:
                if analysis_type == "pest_detection" and any(spec in ["Pesticides", "Organic Solutions"] for spec in supplier["specialties"]):
                    relevant_suppliers.append(supplier)
                elif analysis_type == "soil_analysis" and any(spec in ["Fertilizers", "Soil Testing", "Micronutrients"] for spec in supplier["specialties"]):
                    relevant_suppliers.append(supplier)
                elif analysis_type == "disease_identification" and any(spec in ["Pesticides", "Bio-fertilizers", "Organic Solutions"] for spec in supplier["specialties"]):
                    relevant_suppliers.append(supplier)
                elif analysis_type == "crop_health" and any(spec in ["Fertilizers", "Growth Promoters", "Micronutrients"] for spec in supplier["specialties"]):
                    relevant_suppliers.append(supplier)
            
            if relevant_suppliers:
                return random.sample(relevant_suppliers, min(3, len(relevant_suppliers)))
        
        # Return random 3 suppliers if no specific filtering
        return random.sample(self.suppliers, min(3, len(self.suppliers)))
    
    def get_cost_estimate(self, analysis_type: str, severity: str = "medium") -> str:
        """Get cost estimate based on analysis type and severity"""
        base_costs = {
            "pest_detection": {"low": "₹150-300", "medium": "₹300-600", "high": "₹600-1200"},
            "disease_identification": {"low": "₹200-400", "medium": "₹400-800", "high": "₹800-1500"},
            "soil_analysis": {"low": "₹300-500", "medium": "₹500-1000", "high": "₹1000-2000"},
            "crop_health": {"low": "₹200-400", "medium": "₹400-800", "high": "₹800-1600"}
        }
        
        if analysis_type in base_costs and severity in base_costs[analysis_type]:
            return base_costs[analysis_type][severity] + " per acre"
        
        return "₹300-600 per acre"

# Global instance for easy access
treatments_db = TreatmentsDatabase()