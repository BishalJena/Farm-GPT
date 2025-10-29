"""
Agricultural Tools Module
Integrated MCP tools directly into the backend for simplified deployment
"""

import os
import json
import httpx
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio
from dotenv import load_dotenv

# Load environment variables from the correct path
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

logger = logging.getLogger(__name__)

class AgriculturalTools:
    """Integrated agricultural tools for crop analysis, pricing, and assistance"""
    
    def __init__(self):
        self.datagovin_api_key = os.environ.get('DATAGOVIN_API_KEY')
        self.exa_api_key = os.environ.get('EXA_API_KEY')
        
    async def crop_price_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch crop price data from data.gov.in"""
        try:
            if not self.datagovin_api_key:
                return {"error": "Configuration error: DATAGOVIN_API_KEY not set in environment."}

            state = params.get('state')
            district = params.get('district')
            commodity = params.get('commodity')
            limit = params.get('limit', 50)
            offset = params.get('offset', 0)
            
            resource_id = "35985678-0d79-46b4-9ed6-6f13308a1d24"
            base_url = f"https://api.data.gov.in/resource/{resource_id}"
            
            # Build URL parameters with proper encoding for filters
            url_params = {
                "api-key": self.datagovin_api_key,
                "format": "json",
                "limit": str(limit),
                "offset": str(offset)
            }
            
            # Add filters with proper encoding
            if state:
                url_params["filters[State]"] = state
            if district:
                url_params["filters[District]"] = district
            if commodity:
                url_params["filters[Commodity]"] = commodity
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(base_url, params=url_params)
                
                if response.status_code != 200:
                    return {"error": f"HTTP {response.status_code} fetching data.gov.in: {response.text}"}
                
                data = response.json()
                records = data.get('records', [])
                total = data.get('total', len(records))
                
                return {
                    "success": True,
                    "data": {
                        "records": records,
                        "total": total,
                        "limit": limit,
                        "offset": offset,
                        "query": {"state": state, "district": district, "commodity": commodity}
                    }
                }
                
        except Exception as e:
            logger.error(f"Crop price tool error: {e}")
            return {"error": f"Server error: {str(e)}"}
    
    async def search_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search the web for agricultural information using EXA API"""
        try:
            if not self.exa_api_key:
                return {"error": "Configuration error: EXA_API_KEY not set in environment."}
            
            query = params.get('query')
            if not query:
                return {"error": "Query parameter is required"}
            
            num_results = params.get('num_results', 5)
            include_domains = params.get('include_domains')
            exclude_domains = params.get('exclude_domains')
            
            request_body = {
                "query": query,
                "num_results": num_results,
                "use_autoprompt": True,
                "contents": {"text": True}
            }
            
            if include_domains:
                request_body["include_domains"] = include_domains
            if exclude_domains:
                request_body["exclude_domains"] = exclude_domains
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.exa.ai/search",
                    json=request_body,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.exa_api_key
                    }
                )
                
                if response.status_code != 200:
                    return {"error": f"HTTP {response.status_code} fetching EXA API: {response.text}"}
                
                data = response.json()
                results = data.get('results', [])
                
                formatted_results = []
                for result in results:
                    text = result.get('text', '')
                    formatted_results.append({
                        "title": result.get('title'),
                        "url": result.get('url'),
                        "text": text[:500] + ('...' if len(text) > 500 else ''),
                        "score": result.get('score'),
                        "published_date": result.get('published_date')
                    })
                
                return {
                    "success": True,
                    "data": {
                        "results": formatted_results,
                        "total_results": len(results),
                        "query": query
                    }
                }
                
        except Exception as e:
            logger.error(f"Search tool error: {e}")
            return {"error": f"Server error: {str(e)}"}
    
    async def soil_health_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze soil health parameters and provide crop recommendations"""
        try:
            state = params.get('state')
            district = params.get('district')
            soil_type = params.get('soil_type', 'Unknown')
            npk_values = params.get('npk_values', {})
            ph_level = params.get('ph_level', 6.5)
            organic_content = params.get('organic_content', 0.75)
            
            # Extract NPK values
            nitrogen = npk_values.get('nitrogen', 280)
            phosphorus = npk_values.get('phosphorus', 23)
            potassium = npk_values.get('potassium', 280)
            
            soil_analysis = {
                "location": {"state": state, "district": district},
                "soil_parameters": {
                    "type": soil_type,
                    "ph_level": ph_level,
                    "nitrogen": nitrogen,
                    "phosphorus": phosphorus,
                    "potassium": potassium,
                    "organic_content": organic_content
                },
                "health_score": 0,
                "recommendations": [],
                "suitable_crops": []
            }
            
            # Calculate health score
            health_score = 0
            
            # pH scoring (optimal range 6.0-7.5)
            if 6.0 <= ph_level <= 7.5:
                health_score += 25
            elif 5.5 <= ph_level <= 8.0:
                health_score += 15
            else:
                health_score += 5
            
            # NPK scoring
            if nitrogen >= 280:
                health_score += 20
            elif nitrogen >= 200:
                health_score += 15
            else:
                health_score += 5
                
            if phosphorus >= 20:
                health_score += 20
            elif phosphorus >= 15:
                health_score += 15
            else:
                health_score += 5
                
            if potassium >= 280:
                health_score += 20
            elif potassium >= 200:
                health_score += 15
            else:
                health_score += 5
            
            # Organic content scoring
            if organic_content >= 0.75:
                health_score += 15
            elif organic_content >= 0.5:
                health_score += 10
            else:
                health_score += 3
            
            soil_analysis["health_score"] = min(health_score, 100)
            
            # Generate recommendations
            recommendations = []
            if ph_level < 6.0:
                recommendations.append("Apply lime to increase soil pH")
            if ph_level > 7.5:
                recommendations.append("Apply organic matter to reduce soil pH")
            if nitrogen < 200:
                recommendations.append("Apply nitrogen-rich fertilizers or compost")
            if phosphorus < 15:
                recommendations.append("Apply phosphorus fertilizers (DAP/SSP)")
            if potassium < 200:
                recommendations.append("Apply potassium fertilizers (MOP)")
            if organic_content < 0.5:
                recommendations.append("Increase organic matter through compost and crop residues")
            
            soil_analysis["recommendations"] = recommendations
            
            # Suggest suitable crops
            suitable_crops = []
            if 6.0 <= ph_level <= 7.5:
                if nitrogen >= 250:
                    suitable_crops.extend(["Wheat", "Rice", "Maize"])
                if phosphorus >= 20:
                    suitable_crops.extend(["Cotton", "Sugarcane"])
                if potassium >= 250:
                    suitable_crops.extend(["Potato", "Tomato"])
            
            if 5.5 <= ph_level <= 6.5:
                suitable_crops.extend(["Tea", "Coffee"])
            if organic_content >= 0.75:
                suitable_crops.extend(["Organic vegetables", "Pulses"])
            
            soil_analysis["suitable_crops"] = list(set(suitable_crops))
            
            return {
                "success": True,
                "data": {
                    "analysis": soil_analysis,
                    "timestamp": datetime.now().isoformat(),
                    "source": "Soil Health Card Scheme simulation"
                }
            }
            
        except Exception as e:
            logger.error(f"Soil health tool error: {e}")
            return {"error": f"Soil analysis error: {str(e)}"}
    
    async def weather_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get farming-specific weather forecast with irrigation and pest risk alerts"""
        try:
            location = params.get('location')
            days = params.get('days', 7)
            include_farming_alerts = params.get('include_farming_alerts', True)
            
            if not location:
                return {"error": "Location parameter is required"}
            
            # Generate realistic weather forecast
            current_date = datetime.now()
            forecast = []
            
            for i in range(days):
                date = current_date + timedelta(days=i)
                
                # Generate realistic weather data for Indian conditions
                import random
                temp_max = 28 + random.random() * 10  # 28-38°C
                temp_min = temp_max - 8 - random.random() * 4  # 8-12°C difference
                humidity = 60 + random.random() * 30  # 60-90%
                rainfall = random.random() * 25 if random.random() > 0.7 else 0  # 30% chance of rain
                wind_speed = 5 + random.random() * 15  # 5-20 km/h
                
                forecast.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "temperature": {
                        "max": round(temp_max),
                        "min": round(temp_min)
                    },
                    "humidity": round(humidity),
                    "rainfall": round(rainfall * 10) / 10,
                    "wind_speed": round(wind_speed),
                    "conditions": "Rainy" if rainfall > 0 else ("Cloudy" if humidity > 80 else "Clear")
                })
            
            # Generate farming alerts
            alerts = []
            if include_farming_alerts:
                for day in forecast:
                    if day["rainfall"] > 10:
                        alerts.append({
                            "date": day["date"],
                            "type": "irrigation",
                            "severity": "info",
                            "message": "Heavy rainfall expected. Skip irrigation for the day."
                        })
                    
                    if day["humidity"] > 85 and day["temperature"]["max"] > 30:
                        alerts.append({
                            "date": day["date"],
                            "type": "pest_risk",
                            "severity": "warning",
                            "message": "High humidity and temperature. Monitor for pest activity."
                        })
                    
                    if day["rainfall"] == 0 and day["temperature"]["max"] > 35:
                        alerts.append({
                            "date": day["date"],
                            "type": "heat_stress",
                            "severity": "warning",
                            "message": "High temperature with no rain. Ensure adequate irrigation."
                        })
                    
                    if day["wind_speed"] > 15:
                        alerts.append({
                            "date": day["date"],
                            "type": "wind",
                            "severity": "caution",
                            "message": "Strong winds expected. Secure tall crops and structures."
                        })
            
            # Calculate irrigation schedule
            irrigation_schedule = []
            for day in forecast:
                if day["rainfall"] < 5 and day["temperature"]["max"] > 30:
                    irrigation_schedule.append({
                        "date": day["date"],
                        "recommendation": "irrigate",
                        "timing": "early_morning",
                        "duration": "extended" if day["temperature"]["max"] > 35 else "normal",
                        "reason": f"Low rainfall ({day['rainfall']}mm) and high temperature ({day['temperature']['max']}°C)"
                    })
                elif day["rainfall"] > 10:
                    irrigation_schedule.append({
                        "date": day["date"],
                        "recommendation": "skip",
                        "reason": f"Adequate rainfall expected ({day['rainfall']}mm)"
                    })
            
            return {
                "success": True,
                "data": {
                    "location": location,
                    "forecast": forecast,
                    "alerts": alerts,
                    "irrigation_schedule": irrigation_schedule,
                    "summary": {
                        "total_rainfall": sum(day["rainfall"] for day in forecast),
                        "avg_temperature": round(sum(day["temperature"]["max"] for day in forecast) / len(forecast)),
                        "high_risk_days": len([alert for alert in alerts if alert["severity"] == "warning"]),
                        "irrigation_days": len([day for day in irrigation_schedule if day["recommendation"] == "irrigate"])
                    },
                    "timestamp": datetime.now().isoformat(),
                    "source": "IMD Weather Forecast simulation"
                }
            }
            
        except Exception as e:
            logger.error(f"Weather tool error: {e}")
            return {"error": f"Weather prediction error: {str(e)}"}
    
    async def pest_identifier_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Identify crop pests and diseases based on symptoms"""
        try:
            crop = params.get('crop')
            symptoms = params.get('symptoms')
            image_description = params.get('image_description')
            location = params.get('location')
            
            if not crop or not symptoms:
                return {"error": "Both crop and symptoms parameters are required"}
            
            # Pest database
            pest_database = {
                "rice": [
                    {
                        "name": "Brown Plant Hopper",
                        "symptoms": ["yellowing leaves", "stunted growth", "hopper burn"],
                        "treatment": ["Use neem oil spray", "Apply imidacloprid", "Maintain proper water levels"],
                        "severity": "high",
                        "season": "kharif"
                    },
                    {
                        "name": "Stem Borer",
                        "symptoms": ["dead hearts", "white ears", "holes in stem"],
                        "treatment": ["Use pheromone traps", "Apply cartap hydrochloride", "Remove affected tillers"],
                        "severity": "medium",
                        "season": "kharif"
                    }
                ],
                "wheat": [
                    {
                        "name": "Aphids",
                        "symptoms": ["curled leaves", "sticky honeydew", "yellowing"],
                        "treatment": ["Spray neem oil", "Use ladybird beetles", "Apply dimethoate"],
                        "severity": "medium",
                        "season": "rabi"
                    },
                    {
                        "name": "Rust Disease",
                        "symptoms": ["orange spots", "leaf yellowing", "reduced yield"],
                        "treatment": ["Apply propiconazole", "Use resistant varieties", "Proper field sanitation"],
                        "severity": "high",
                        "season": "rabi"
                    }
                ],
                "cotton": [
                    {
                        "name": "Bollworm",
                        "symptoms": ["holes in bolls", "damaged squares", "frass presence"],
                        "treatment": ["Use Bt cotton varieties", "Apply spinosad", "Pheromone traps"],
                        "severity": "high",
                        "season": "kharif"
                    },
                    {
                        "name": "Whitefly",
                        "symptoms": ["yellowing leaves", "sooty mold", "leaf curl"],
                        "treatment": ["Yellow sticky traps", "Spray acetamiprid", "Reflective mulch"],
                        "severity": "medium",
                        "season": "kharif"
                    }
                ]
            }
            
            crop_pests = pest_database.get(crop.lower(), [])
            
            # Match symptoms to identify pest
            identified_pest = None
            confidence = 0
            
            if symptoms:
                symptoms_list = [s.strip().lower() for s in symptoms.split(',')]
                
                for pest in crop_pests:
                    matches = 0
                    for symptom in symptoms_list:
                        if any(symptom in ps.lower() or ps.lower() in symptom for ps in pest["symptoms"]):
                            matches += 1
                    
                    pest_confidence = (matches / max(len(symptoms_list), len(pest["symptoms"]))) * 100
                    if pest_confidence > confidence:
                        confidence = pest_confidence
                        identified_pest = pest
            
            # If no specific match, provide general recommendations
            if not identified_pest and crop_pests:
                identified_pest = crop_pests[0]
                confidence = 30
            
            result = {
                "crop": crop,
                "location": location,
                "identification": {
                    "pest_name": identified_pest["name"] if identified_pest else "Unknown",
                    "confidence_score": round(confidence),
                    "severity": identified_pest["severity"] if identified_pest else "unknown",
                    "season": identified_pest["season"] if identified_pest else "unknown"
                } if identified_pest else None,
                "treatment_recommendations": identified_pest["treatment"] if identified_pest else [
                    "Consult local agricultural extension officer",
                    "Take clear photos of affected plants",
                    "Monitor pest population regularly"
                ],
                "prevention_measures": [
                    "Regular field monitoring",
                    "Crop rotation practices",
                    "Maintain field hygiene",
                    "Use resistant varieties when available",
                    "Integrated Pest Management (IPM)"
                ],
                "alternative_pests": [
                    {
                        "name": p["name"],
                        "symptoms": p["symptoms"],
                        "severity": p["severity"]
                    } for p in crop_pests if p != identified_pest
                ][:2]
            }
            
            return {
                "success": True,
                "data": {
                    "analysis": result,
                    "timestamp": datetime.now().isoformat(),
                    "source": "Agricultural Pest Database simulation",
                    "note": "For accurate identification, consult with local agricultural experts"
                }
            }
            
        except Exception as e:
            logger.error(f"Pest identifier tool error: {e}")
            return {"error": f"Pest identification error: {str(e)}"}
    
    async def mandi_price_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Track mandi prices with trends, predictions, and market recommendations"""
        try:
            commodity = params.get('commodity')
            state = params.get('state')
            district = params.get('district')
            days_back = params.get('days_back', 30)
            include_predictions = params.get('include_predictions', True)
            
            if not commodity:
                return {"error": "Commodity parameter is required"}
            
            # Base prices for different commodities
            base_prices = {
                "wheat": 2100,
                "rice": 1800,
                "cotton": 5500,
                "maize": 1600,
                "soybean": 4200,
                "sugarcane": 350,
                "onion": 2500,
                "potato": 1200
            }
            
            commodity_price = base_prices.get(commodity.lower(), 2000)
            current_date = datetime.now()
            price_history = []
            
            # Generate price history
            import random
            for i in range(days_back, -1, -1):
                date = current_date - timedelta(days=i)
                
                # Add realistic price variations
                variation = (random.random() - 0.5) * 0.2  # ±10% variation
                seasonal_factor = __import__('math').sin((date.month / 12) * 2 * __import__('math').pi) * 0.1
                trend_factor = (days_back - i) / days_back * 0.05
                
                price = commodity_price * (1 + variation + seasonal_factor + trend_factor)
                
                price_history.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "price": round(price),
                    "market": f"{district or 'Local'} Mandi",
                    "volume": round(50 + random.random() * 200),
                    "quality": random.choice(["FAQ", "Good", "Average"])
                })
            
            # Calculate trends
            recent_prices = [p["price"] for p in price_history[-7:]]
            older_prices = [p["price"] for p in price_history[-14:-7]]
            
            recent_avg = sum(recent_prices) / len(recent_prices)
            older_avg = sum(older_prices) / len(older_prices) if older_prices else recent_avg
            
            trend = "increasing" if recent_avg > older_avg else ("decreasing" if recent_avg < older_avg else "stable")
            trend_percentage = abs(((recent_avg - older_avg) / older_avg) * 100) if older_avg else 0
            
            # Price predictions
            predictions = []
            if include_predictions:
                last_price = price_history[-1]["price"]
                trend_factor = 1.02 if trend == "increasing" else (0.98 if trend == "decreasing" else 1.0)
                
                for i in range(1, 8):
                    future_date = current_date + timedelta(days=i)
                    predicted_price = last_price * (trend_factor ** i) * (1 + (random.random() - 0.5) * 0.05)
                    
                    predictions.append({
                        "date": future_date.strftime('%Y-%m-%d'),
                        "predicted_price": round(predicted_price),
                        "confidence": max(60 - i * 5, 30)
                    })
            
            # Market recommendations
            current_price = price_history[-1]["price"]
            avg_price = sum(p["price"] for p in price_history) / len(price_history)
            
            recommendations = []
            if current_price > avg_price * 1.1:
                recommendations.append({
                    "action": "sell",
                    "reason": f"Current price (₹{current_price}) is {round(((current_price - avg_price) / avg_price) * 100)}% above average",
                    "urgency": "high"
                })
            elif current_price < avg_price * 0.9:
                recommendations.append({
                    "action": "hold",
                    "reason": f"Current price (₹{current_price}) is below average. Wait for better rates",
                    "urgency": "medium"
                })
            else:
                recommendations.append({
                    "action": "monitor",
                    "reason": "Price is near average. Monitor for trend changes",
                    "urgency": "low"
                })
            
            # Best markets
            nearby_markets = [
                {"name": f"{district or 'Local'} Mandi", "price": current_price, "distance": 0},
                {"name": f"{state or 'State'} Central Market", "price": round(current_price * 1.05), "distance": 25},
                {"name": "Regional Hub Market", "price": round(current_price * 1.08), "distance": 45}
            ]
            nearby_markets.sort(key=lambda x: x["price"], reverse=True)
            
            return {
                "success": True,
                "data": {
                    "commodity": commodity,
                    "location": {"state": state, "district": district},
                    "current_price": current_price,
                    "price_history": price_history[-14:],  # Last 2 weeks
                    "trend_analysis": {
                        "direction": trend,
                        "percentage_change": round(trend_percentage * 100) / 100,
                        "period": "7 days"
                    },
                    "predictions": predictions,
                    "recommendations": recommendations,
                    "best_markets": nearby_markets,
                    "market_summary": {
                        "highest_price": max(p["price"] for p in price_history),
                        "lowest_price": min(p["price"] for p in price_history),
                        "average_price": round(avg_price),
                        "total_volume": sum(p["volume"] for p in price_history)
                    },
                    "timestamp": datetime.now().isoformat(),
                    "source": "Multi-Mandi Price Aggregation simulation"
                }
            }
            
        except Exception as e:
            logger.error(f"Mandi price tool error: {e}")
            return {"error": f"Mandi price tracking error: {str(e)}"}   
 
    async def scheme_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Help farmers with crop damage schemes, insurance claims, and government relief programs"""
        try:
            damage_type = params.get('damage_type')
            crop_type = params.get('crop_type')
            state = params.get('state')
            district = params.get('district')
            damage_extent = params.get('damage_extent', 'moderate')
            has_insurance = params.get('has_insurance', False)
            insurance_type = params.get('insurance_type', 'none')
            land_size_acres = params.get('land_size_acres', 2.5)
            farmer_category = params.get('farmer_category', 'small')
            
            if not damage_type or not crop_type:
                return {"error": "Both damage_type and crop_type parameters are required"}
            
            # Comprehensive scheme database
            scheme_database = {
                "crop_insurance": [
                    {
                        "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
                        "coverage": ["drought", "flood", "cyclone", "hailstorm", "pest_attack", "disease", "fire"],
                        "premium_subsidy": "Up to 90% government subsidy",
                        "max_coverage": "Sum Insured based on Scale of Finance",
                        "claim_process": [
                            "Report crop loss within 72 hours",
                            "Submit claim form with required documents",
                            "Crop Cutting Experiments (CCE) conducted",
                            "Settlement based on yield data"
                        ],
                        "documents_required": [
                            "Land records (Khatauni/Khewat)",
                            "Aadhaar card",
                            "Bank account details",
                            "Sowing certificate",
                            "Crop loss photos"
                        ],
                        "helpline": "1800-200-7710",
                        "website": "pmfby.gov.in",
                        "eligibility": "All farmers (loanee and non-loanee)"
                    },
                    {
                        "name": "Weather Based Crop Insurance Scheme (WBCIS)",
                        "coverage": ["drought", "flood", "cyclone", "hailstorm"],
                        "premium_subsidy": "Up to 90% for small farmers",
                        "max_coverage": "Based on weather parameters",
                        "claim_process": [
                            "Automatic trigger based on weather data",
                            "No need to report individual losses",
                            "Settlement within 45 days of harvest"
                        ],
                        "documents_required": [
                            "Land records",
                            "Aadhaar card",
                            "Bank account details"
                        ],
                        "helpline": "1800-200-7710",
                        "website": "pmfby.gov.in",
                        "eligibility": "Farmers in notified areas"
                    }
                ],
                "relief_schemes": [
                    {
                        "name": "State Disaster Response Fund (SDRF)",
                        "coverage": ["drought", "flood", "cyclone", "hailstorm", "fire"],
                        "compensation": "₹6,800 per hectare for food crops, ₹18,000 for cash crops",
                        "process": [
                            "Village level damage assessment",
                            "Application through Tehsildar",
                            "Joint verification by officials",
                            "Compensation disbursement"
                        ],
                        "documents_required": [
                            "Land ownership documents",
                            "Crop loss certificate from Patwari",
                            "Bank account details",
                            "Aadhaar card"
                        ],
                        "timeline": "45-60 days from application",
                        "eligibility": "All farmers with land records"
                    },
                    {
                        "name": "National Disaster Response Fund (NDRF)",
                        "coverage": ["severe_drought", "major_flood", "cyclone"],
                        "compensation": "Additional support for severe calamities",
                        "process": [
                            "State government recommendation",
                            "Central team assessment",
                            "Additional compensation approval"
                        ],
                        "documents_required": [
                            "SDRF application documents",
                            "Severity assessment report"
                        ],
                        "timeline": "90-120 days",
                        "eligibility": "Farmers in severely affected areas"
                    }
                ],
                "input_schemes": [
                    {
                        "name": "Seed Subsidy Scheme",
                        "coverage": ["drought", "flood", "pest_attack", "disease"],
                        "benefit": "50% subsidy on certified seeds",
                        "process": [
                            "Apply at nearest Krishi Vigyan Kendra",
                            "Submit crop loss certificate",
                            "Collect subsidized seeds"
                        ],
                        "documents_required": [
                            "Land records",
                            "Crop loss certificate",
                            "Aadhaar card"
                        ],
                        "timeline": "15-30 days",
                        "eligibility": "Small and marginal farmers"
                    },
                    {
                        "name": "Fertilizer Subsidy for Affected Farmers",
                        "coverage": ["drought", "flood", "pest_attack"],
                        "benefit": "Additional 25% subsidy on fertilizers",
                        "process": [
                            "Apply through Primary Agricultural Credit Society",
                            "Submit damage assessment report",
                            "Purchase from authorized dealers"
                        ],
                        "documents_required": [
                            "Damage assessment certificate",
                            "Soil health card",
                            "Aadhaar card"
                        ],
                        "timeline": "Immediate after approval",
                        "eligibility": "All affected farmers"
                    }
                ]
            }
            
            # Analyze farmer situation
            farmer_situation = {
                "damage_type": damage_type,
                "crop_type": crop_type,
                "state": state,
                "district": district,
                "damage_extent": damage_extent,
                "has_insurance": has_insurance,
                "insurance_type": insurance_type,
                "land_size_acres": land_size_acres,
                "farmer_category": farmer_category
            }
            
            # Find applicable schemes
            applicable_schemes = []
            
            # Check insurance schemes
            if has_insurance and insurance_type != "none":
                insurance_schemes = [
                    scheme for scheme in scheme_database["crop_insurance"]
                    if damage_type in scheme["coverage"] or damage_type == "natural_calamity"
                ]
                applicable_schemes.extend([
                    {**scheme, "category": "insurance", "priority": "high", "immediate_action": True}
                    for scheme in insurance_schemes
                ])
            
            # Check relief schemes
            relief_schemes = [
                scheme for scheme in scheme_database["relief_schemes"]
                if damage_type in scheme["coverage"] or damage_type == "natural_calamity"
            ]
            applicable_schemes.extend([
                {**scheme, "category": "relief", "priority": "high" if not has_insurance else "medium", "immediate_action": not has_insurance}
                for scheme in relief_schemes
            ])
            
            # Check input schemes
            input_schemes = [
                scheme for scheme in scheme_database["input_schemes"]
                if damage_type in scheme["coverage"]
            ]
            applicable_schemes.extend([
                {**scheme, "category": "input_support", "priority": "medium", "immediate_action": False}
                for scheme in input_schemes
            ])
            
            # Calculate estimated compensation
            estimated_compensation = 0
            
            if has_insurance and insurance_type == "pmfby":
                crop_multipliers = {
                    "rice": 40000,
                    "wheat": 35000,
                    "cotton": 60000,
                    "sugarcane": 80000,
                    "maize": 30000
                }
                estimated_compensation = crop_multipliers.get(crop_type.lower(), 35000) * land_size_acres
            else:
                # SDRF compensation
                is_cash_crop = crop_type.lower() in ["cotton", "sugarcane", "tobacco"]
                per_hectare_rate = 18000 if is_cash_crop else 6800
                estimated_compensation = per_hectare_rate * land_size_acres * 0.4047  # Convert acres to hectares
            
            # Adjust based on damage extent
            damage_multipliers = {
                "minor": 0.3,
                "moderate": 0.6,
                "severe": 0.8,
                "complete": 1.0
            }
            estimated_compensation *= damage_multipliers.get(damage_extent, 0.6)
            
            # Generate action plan
            action_plan = {
                "immediate_actions": [],
                "short_term_actions": [],
                "long_term_actions": []
            }
            
            # Immediate actions (0-7 days)
            if has_insurance:
                action_plan["immediate_actions"].append({
                    "action": "Report crop loss to insurance company",
                    "timeline": "Within 72 hours",
                    "contact": "1800-200-7710 (PMFBY Helpline)",
                    "documents": ["Land records", "Crop photos", "Sowing certificate"]
                })
            
            action_plan["immediate_actions"].extend([
                {
                    "action": "Visit Tehsildar office for damage assessment",
                    "timeline": "Within 7 days",
                    "contact": "Local Tehsildar office",
                    "documents": ["Land ownership documents", "Aadhaar card"]
                },
                {
                    "action": "Document crop damage with photos",
                    "timeline": "Immediately",
                    "contact": "Self or local photographer",
                    "documents": ["GPS-tagged photos", "Date-stamped images"]
                }
            ])
            
            # Short-term actions (1-4 weeks)
            action_plan["short_term_actions"].extend([
                {
                    "action": "Apply for SDRF compensation",
                    "timeline": "Within 15 days",
                    "contact": "District Collector office",
                    "documents": ["Crop loss certificate", "Bank details", "Land records"]
                },
                {
                    "action": "Apply for subsidized seeds/inputs",
                    "timeline": "Within 30 days",
                    "contact": "Krishi Vigyan Kendra",
                    "documents": ["Damage certificate", "Soil health card"]
                }
            ])
            
            # Long-term actions (1-3 months)
            action_plan["long_term_actions"].extend([
                {
                    "action": "Plan next season crop with climate-resilient varieties",
                    "timeline": "Before next sowing season",
                    "contact": "Agricultural extension officer",
                    "documents": ["Soil test report", "Weather advisory"]
                },
                {
                    "action": "Enroll in crop insurance for next season",
                    "timeline": "Before next sowing",
                    "contact": "Bank or insurance agent",
                    "documents": ["Land records", "Bank account", "Aadhaar"]
                }
            ])
            
            # Search for recent relief announcements using EXA (if available)
            recent_announcements = []
            try:
                if self.exa_api_key:
                    search_query = f"{state} crop damage relief scheme {damage_type} {datetime.now().year}"
                    search_result = await self.search_tool({
                        "query": search_query,
                        "num_results": 3
                    })
                    
                    if search_result.get("success"):
                        recent_announcements = [
                            {
                                "title": result["title"],
                                "url": result["url"],
                                "summary": result["text"][:200] + "...",
                                "published_date": result.get("published_date")
                            }
                            for result in search_result["data"]["results"][:3]
                        ]
            except Exception as e:
                logger.warning(f"Could not fetch recent announcements: {e}")
            
            return {
                "success": True,
                "data": {
                    "farmer_situation": farmer_situation,
                    "estimated_compensation": round(estimated_compensation),
                    "applicable_schemes_count": len(applicable_schemes),
                    "recommendations": sorted(applicable_schemes, key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x["priority"], 0), reverse=True),
                    "action_plan": action_plan,
                    "recent_relief_announcements": recent_announcements,
                    "important_contacts": {
                        "pmfby_helpline": "1800-200-7710",
                        "kisan_call_center": "1800-180-1551",
                        "district_collector": f"Contact local District Collector office in {district or state}",
                        "krishi_vigyan_kendra": f"Contact nearest KVK in {district or state}"
                    },
                    "next_steps_summary": [
                        "File insurance claim within 72 hours" if has_insurance else "Apply for SDRF relief immediately",
                        "Get official damage assessment from Tehsildar",
                        "Document all crop damage with photos",
                        "Apply for input subsidies for next season",
                        "Consider crop insurance for future protection"
                    ],
                    "timestamp": datetime.now().isoformat(),
                    "source": "Comprehensive Agricultural Scheme Database"
                }
            }
            
        except Exception as e:
            logger.error(f"Scheme tool error: {e}")
            return {"error": f"Scheme tool error: {str(e)}"}
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific agricultural tool"""
        tool_methods = {
            'crop-price': self.crop_price_tool,
            'search': self.search_tool,
            'soil-health': self.soil_health_tool,
            'weather': self.weather_tool,
            'pest-identifier': self.pest_identifier_tool,
            'mandi-price': self.mandi_price_tool,
            'scheme-tool': self.scheme_tool
        }
        
        if tool_name not in tool_methods:
            return {"error": f"Tool '{tool_name}' not found. Available tools: {list(tool_methods.keys())}"}
        
        try:
            return await tool_methods[tool_name](parameters)
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {"error": f"Error calling tool {tool_name}: {str(e)}"}
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools with their descriptions"""
        return [
            {
                "name": "crop-price",
                "description": "Fetch crop price data from data.gov.in with state/district/commodity filters",
                "parameters": ["state", "district", "commodity", "limit", "offset"]
            },
            {
                "name": "search",
                "description": "Search the web for agricultural information using EXA API",
                "parameters": ["query", "num_results", "include_domains", "exclude_domains"]
            },
            {
                "name": "soil-health",
                "description": "Analyze soil health parameters and provide crop recommendations",
                "parameters": ["state", "district", "soil_type", "npk_values", "ph_level", "organic_content"]
            },
            {
                "name": "weather",
                "description": "Get farming-specific weather forecast with irrigation and pest risk alerts",
                "parameters": ["location", "days", "include_farming_alerts"]
            },
            {
                "name": "pest-identifier",
                "description": "Identify crop pests and diseases based on symptoms and provide treatment recommendations",
                "parameters": ["crop", "symptoms", "image_description", "location"]
            },
            {
                "name": "mandi-price",
                "description": "Track mandi prices with trends, predictions, and market recommendations",
                "parameters": ["commodity", "state", "district", "days_back", "include_predictions"]
            },
            {
                "name": "scheme-tool",
                "description": "Help farmers with crop damage schemes, insurance claims, and government relief programs",
                "parameters": ["damage_type", "crop_type", "state", "district", "damage_extent", "has_insurance", "insurance_type", "land_size_acres", "farmer_category"]
            }
        ]