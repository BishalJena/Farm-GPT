// src/server.ts - Agricultural AI MCP Server for Docker MCP Gateway
import { config } from "dotenv";
config(); // Load .env file

import fetch from "node-fetch";
import { createServer, IncomingMessage, ServerResponse } from 'http';

// MCP Protocol Types
interface MCPRequest {
    jsonrpc: "2.0";
    id: string | number;
    method: string;
    params?: any;
}

interface MCPResponse {
    jsonrpc: "2.0";
    id: string | number;
    result?: any;
    error?: {
        code: number;
        message: string;
        data?: any;
    };
}

interface MCPTool {
    name: string;
    description: string;
    inputSchema: {
        type: "object";
        properties: Record<string, any>;
        required?: string[];
    };
}

const PORT = process.env.PORT || 10000;

// Type definitions for better TypeScript support
interface PestData {
    name: string;
    symptoms: string[];
    treatment: string[];
    severity: string;
    season: string;
}

interface PestDatabase {
    [key: string]: PestData[];
}

interface BasePrice {
    [key: string]: number;
}

/**
 * Crop Price Tool Handler
 */
const cropPriceHandler = async (params: any) => {
    try {
        const API_KEY = process.env.DATAGOVIN_API_KEY;
        const RESOURCE_ID = process.env.DATAGOVIN_RESOURCE_ID ?? "35985678-0d79-46b4-9ed6-6f13308a1d24";

        if (!API_KEY) {
            return {
                error: "Configuration error: DATAGOVIN_API_KEY not set in environment."
            };
        }

        const { state, district, commodity } = params;
        const limit = params.limit ?? 50;
        const offset = params.offset ?? 0;

        // Build URL with filters[...] parameters
        const base = `https://api.data.gov.in/resource/${encodeURIComponent(RESOURCE_ID)}`;
        const urlParams = new URLSearchParams();
        urlParams.set("api-key", API_KEY);
        urlParams.set("format", "json");
        urlParams.set("limit", String(limit));
        urlParams.set("offset", String(offset));

        if (state) urlParams.append("filters[State]", state);
        if (district) urlParams.append("filters[District]", district);
        if (commodity) urlParams.append("filters[Commodity]", commodity);

        const url = `${base}?${urlParams.toString()}`;

        // Fetch data
        const res = await fetch(url, { method: "GET" });
        const text = await res.text();

        if (!res.ok) {
            return {
                error: `HTTP ${res.status} fetching data.gov.in: ${text}`
            };
        }

        // parse JSON (defensive)
        let json;
        try {
            json = JSON.parse(text);
        } catch (err) {
            return { error: `Invalid JSON response: ${text}` };
        }

        // Format response for better readability
        const records = json.records || [];
        const total = json.total || records.length;

        return {
            success: true,
            data: {
                records,
                total,
                limit,
                offset,
                query: { state, district, commodity }
            }
        };
    } catch (err) {
        return { error: `Server error: ${String(err)}` };
    }
};

/**
 * Soil Health Analyzer Tool Handler
 */
const soilHealthHandler = async (params: any) => {
    try {
        const { state, district, soil_type, npk_values, ph_level, organic_content } = params;

        // Simulate soil health analysis (in production, this would connect to government databases)
        const soilAnalysis = {
            location: { state, district },
            soil_parameters: {
                type: soil_type || "Unknown",
                ph_level: ph_level || 6.5,
                nitrogen: npk_values?.nitrogen || 280,
                phosphorus: npk_values?.phosphorus || 23,
                potassium: npk_values?.potassium || 280,
                organic_content: organic_content || 0.75
            },
            health_score: 0,
            recommendations: [] as string[],
            suitable_crops: [] as string[]
        };

        // Calculate health score based on parameters
        let healthScore = 0;
        const ph = soilAnalysis.soil_parameters.ph_level;
        const nitrogen = soilAnalysis.soil_parameters.nitrogen;
        const phosphorus = soilAnalysis.soil_parameters.phosphorus;
        const potassium = soilAnalysis.soil_parameters.potassium;
        const organic = soilAnalysis.soil_parameters.organic_content;

        // pH scoring (optimal range 6.0-7.5)
        if (ph >= 6.0 && ph <= 7.5) healthScore += 25;
        else if (ph >= 5.5 && ph <= 8.0) healthScore += 15;
        else healthScore += 5;

        // NPK scoring
        if (nitrogen >= 280) healthScore += 20;
        else if (nitrogen >= 200) healthScore += 15;
        else healthScore += 5;

        if (phosphorus >= 20) healthScore += 20;
        else if (phosphorus >= 15) healthScore += 15;
        else healthScore += 5;

        if (potassium >= 280) healthScore += 20;
        else if (potassium >= 200) healthScore += 15;
        else healthScore += 5;

        // Organic content scoring
        if (organic >= 0.75) healthScore += 15;
        else if (organic >= 0.5) healthScore += 10;
        else healthScore += 3;

        soilAnalysis.health_score = Math.min(healthScore, 100);

        // Generate recommendations based on analysis
        const recommendations: string[] = [];
        if (ph < 6.0) recommendations.push("Apply lime to increase soil pH");
        if (ph > 7.5) recommendations.push("Apply organic matter to reduce soil pH");
        if (nitrogen < 200) recommendations.push("Apply nitrogen-rich fertilizers or compost");
        if (phosphorus < 15) recommendations.push("Apply phosphorus fertilizers (DAP/SSP)");
        if (potassium < 200) recommendations.push("Apply potassium fertilizers (MOP)");
        if (organic < 0.5) recommendations.push("Increase organic matter through compost and crop residues");

        soilAnalysis.recommendations = recommendations;

        // Suggest suitable crops based on soil conditions
        const suitableCrops: string[] = [];
        if (ph >= 6.0 && ph <= 7.5) {
            if (nitrogen >= 250) suitableCrops.push("Wheat", "Rice", "Maize");
            if (phosphorus >= 20) suitableCrops.push("Cotton", "Sugarcane");
            if (potassium >= 250) suitableCrops.push("Potato", "Tomato");
        }
        
        if (ph >= 5.5 && ph <= 6.5) suitableCrops.push("Tea", "Coffee");
        if (organic >= 0.75) suitableCrops.push("Organic vegetables", "Pulses");

        soilAnalysis.suitable_crops = [...new Set(suitableCrops)]; // Remove duplicates

        return {
            success: true,
            data: {
                analysis: soilAnalysis,
                timestamp: new Date().toISOString(),
                source: "Soil Health Card Scheme simulation"
            }
        };
    } catch (err) {
        return { error: `Soil analysis error: ${String(err)}` };
    }
};

/**
 * Weather Predictor Tool Handler
 */
const weatherHandler = async (params: any) => {
    try {
        const { location, days = 7, include_farming_alerts = true } = params;

        // Simulate weather data (in production, this would connect to IMD API)
        const currentDate = new Date();
        const forecast = [];

        for (let i = 0; i < days; i++) {
            const date = new Date(currentDate);
            date.setDate(date.getDate() + i);

            // Generate realistic weather data for Indian conditions
            const temp_max = 28 + Math.random() * 10; // 28-38°C
            const temp_min = temp_max - 8 - Math.random() * 4; // 8-12°C difference
            const humidity = 60 + Math.random() * 30; // 60-90%
            const rainfall = Math.random() > 0.7 ? Math.random() * 25 : 0; // 30% chance of rain
            const wind_speed = 5 + Math.random() * 15; // 5-20 km/h

            forecast.push({
                date: date.toISOString().split('T')[0],
                temperature: {
                    max: Math.round(temp_max),
                    min: Math.round(temp_min)
                },
                humidity: Math.round(humidity),
                rainfall: Math.round(rainfall * 10) / 10,
                wind_speed: Math.round(wind_speed),
                conditions: rainfall > 0 ? "Rainy" : humidity > 80 ? "Cloudy" : "Clear"
            });
        }

        // Generate farming alerts based on weather conditions
        const alerts: any[] = [];
        if (include_farming_alerts) {
            forecast.forEach((day, index) => {
                if (day.rainfall > 10) {
                    alerts.push({
                        date: day.date,
                        type: "irrigation",
                        severity: "info",
                        message: "Heavy rainfall expected. Skip irrigation for the day."
                    });
                }
                
                if (day.humidity > 85 && day.temperature.max > 30) {
                    alerts.push({
                        date: day.date,
                        type: "pest_risk",
                        severity: "warning",
                        message: "High humidity and temperature. Monitor for pest activity."
                    });
                }
                
                if (day.rainfall === 0 && day.temperature.max > 35) {
                    alerts.push({
                        date: day.date,
                        type: "heat_stress",
                        severity: "warning",
                        message: "High temperature with no rain. Ensure adequate irrigation."
                    });
                }
                
                if (day.wind_speed > 15) {
                    alerts.push({
                        date: day.date,
                        type: "wind",
                        severity: "caution",
                        message: "Strong winds expected. Secure tall crops and structures."
                    });
                }
            });
        }

        // Calculate irrigation recommendations
        const irrigation_schedule: any[] = [];
        forecast.forEach((day, index) => {
            if (day.rainfall < 5 && day.temperature.max > 30) {
                irrigation_schedule.push({
                    date: day.date,
                    recommendation: "irrigate",
                    timing: "early_morning",
                    duration: day.temperature.max > 35 ? "extended" : "normal",
                    reason: `Low rainfall (${day.rainfall}mm) and high temperature (${day.temperature.max}°C)`
                });
            } else if (day.rainfall > 10) {
                irrigation_schedule.push({
                    date: day.date,
                    recommendation: "skip",
                    reason: `Adequate rainfall expected (${day.rainfall}mm)`
                });
            }
        });

        return {
            success: true,
            data: {
                location,
                forecast,
                alerts,
                irrigation_schedule,
                summary: {
                    total_rainfall: forecast.reduce((sum, day) => sum + day.rainfall, 0),
                    avg_temperature: Math.round(forecast.reduce((sum, day) => sum + day.temperature.max, 0) / forecast.length),
                    high_risk_days: alerts.filter(alert => alert.severity === "warning").length,
                    irrigation_days: irrigation_schedule.filter(day => day.recommendation === "irrigate").length
                },
                timestamp: new Date().toISOString(),
                source: "IMD Weather Forecast simulation"
            }
        };
    } catch (err) {
        return { error: `Weather prediction error: ${String(err)}` };
    }
};

/**
 * Pest Identifier Tool Handler
 */
const pestIdentifierHandler = async (params: any) => {
    try {
        const { crop, symptoms, image_description, location } = params;

        // Simulate pest identification (in production, this would use image recognition AI)
        const pestDatabase: PestDatabase = {
            "rice": [
                {
                    name: "Brown Plant Hopper",
                    symptoms: ["yellowing leaves", "stunted growth", "hopper burn"],
                    treatment: ["Use neem oil spray", "Apply imidacloprid", "Maintain proper water levels"],
                    severity: "high",
                    season: "kharif"
                },
                {
                    name: "Stem Borer",
                    symptoms: ["dead hearts", "white ears", "holes in stem"],
                    treatment: ["Use pheromone traps", "Apply cartap hydrochloride", "Remove affected tillers"],
                    severity: "medium",
                    season: "kharif"
                }
            ],
            "wheat": [
                {
                    name: "Aphids",
                    symptoms: ["curled leaves", "sticky honeydew", "yellowing"],
                    treatment: ["Spray neem oil", "Use ladybird beetles", "Apply dimethoate"],
                    severity: "medium",
                    season: "rabi"
                },
                {
                    name: "Rust Disease",
                    symptoms: ["orange spots", "leaf yellowing", "reduced yield"],
                    treatment: ["Apply propiconazole", "Use resistant varieties", "Proper field sanitation"],
                    severity: "high",
                    season: "rabi"
                }
            ],
            "cotton": [
                {
                    name: "Bollworm",
                    symptoms: ["holes in bolls", "damaged squares", "frass presence"],
                    treatment: ["Use Bt cotton varieties", "Apply spinosad", "Pheromone traps"],
                    severity: "high",
                    season: "kharif"
                },
                {
                    name: "Whitefly",
                    symptoms: ["yellowing leaves", "sooty mold", "leaf curl"],
                    treatment: ["Yellow sticky traps", "Spray acetamiprid", "Reflective mulch"],
                    severity: "medium",
                    season: "kharif"
                }
            ]
        };

        const cropPests = pestDatabase[crop?.toLowerCase()] || [];
        
        // Match symptoms to identify pest
        let identifiedPest: PestData | null = null;
        let confidence = 0;

        if (symptoms) {
            const symptomsList = symptoms.toLowerCase().split(',').map((s: string) => s.trim());
            
            for (const pest of cropPests) {
                let matches = 0;
                for (const symptom of symptomsList) {
                    if (pest.symptoms.some((ps: string) => ps.includes(symptom) || symptom.includes(ps))) {
                        matches++;
                    }
                }
                
                const pestConfidence = (matches / Math.max(symptomsList.length, pest.symptoms.length)) * 100;
                if (pestConfidence > confidence) {
                    confidence = pestConfidence;
                    identifiedPest = pest;
                }
            }
        }

        // If no specific match, provide general recommendations
        if (!identifiedPest && cropPests.length > 0) {
            identifiedPest = cropPests[0]; // Default to first common pest
            confidence = 30; // Low confidence
        }

        const result = {
            crop: crop || "Unknown",
            location,
            identification: identifiedPest ? {
                pest_name: identifiedPest.name,
                confidence_score: Math.round(confidence),
                severity: identifiedPest.severity,
                season: identifiedPest.season
            } : null,
            treatment_recommendations: identifiedPest ? identifiedPest.treatment : [
                "Consult local agricultural extension officer",
                "Take clear photos of affected plants",
                "Monitor pest population regularly"
            ],
            prevention_measures: [
                "Regular field monitoring",
                "Crop rotation practices",
                "Maintain field hygiene",
                "Use resistant varieties when available",
                "Integrated Pest Management (IPM)"
            ],
            alternative_pests: cropPests.filter((p: PestData) => p !== identifiedPest).slice(0, 2).map((p: PestData) => ({
                name: p.name,
                symptoms: p.symptoms,
                severity: p.severity
            }))
        };

        return {
            success: true,
            data: {
                analysis: result,
                timestamp: new Date().toISOString(),
                source: "Agricultural Pest Database simulation",
                note: "For accurate identification, consult with local agricultural experts"
            }
        };
    } catch (err) {
        return { error: `Pest identification error: ${String(err)}` };
    }
};

/**
 * Mandi Price Tracker Tool Handler
 */
const mandiPriceHandler = async (params: any) => {
    try {
        const { commodity, state, district, days_back = 30, include_predictions = true } = params;

        // Simulate mandi price data with trends (in production, this would aggregate from multiple sources)
        const currentDate = new Date();
        const priceHistory = [];
        const basePrice: BasePrice = {
            "wheat": 2100,
            "rice": 1800,
            "cotton": 5500,
            "maize": 1600,
            "soybean": 4200,
            "sugarcane": 350,
            "onion": 2500,
            "potato": 1200
        };

        const commodityPrice = basePrice[commodity?.toLowerCase()] || 2000;

        // Generate price history
        for (let i = days_back; i >= 0; i--) {
            const date = new Date(currentDate);
            date.setDate(date.getDate() - i);

            // Add realistic price variations
            const variation = (Math.random() - 0.5) * 0.2; // ±10% variation
            const seasonalFactor = Math.sin((date.getMonth() / 12) * 2 * Math.PI) * 0.1; // Seasonal variation
            const trendFactor = (days_back - i) / days_back * 0.05; // Slight upward trend

            const price = commodityPrice * (1 + variation + seasonalFactor + trendFactor);

            priceHistory.push({
                date: date.toISOString().split('T')[0],
                price: Math.round(price),
                market: `${district || 'Local'} Mandi`,
                volume: Math.round(50 + Math.random() * 200), // Quintals
                quality: ["FAQ", "Good", "Average"][Math.floor(Math.random() * 3)]
            });
        }

        // Calculate trends
        const recentPrices = priceHistory.slice(-7).map(p => p.price);
        const olderPrices = priceHistory.slice(-14, -7).map(p => p.price);
        
        const recentAvg = recentPrices.reduce((a, b) => a + b, 0) / recentPrices.length;
        const olderAvg = olderPrices.reduce((a, b) => a + b, 0) / olderPrices.length;
        
        const trend = recentAvg > olderAvg ? "increasing" : recentAvg < olderAvg ? "decreasing" : "stable";
        const trendPercentage = Math.abs(((recentAvg - olderAvg) / olderAvg) * 100);

        // Price predictions (simple trend-based)
        const predictions = [];
        if (include_predictions) {
            const lastPrice = priceHistory[priceHistory.length - 1].price;
            const trendFactor = trend === "increasing" ? 1.02 : trend === "decreasing" ? 0.98 : 1.0;

            for (let i = 1; i <= 7; i++) {
                const futureDate = new Date(currentDate);
                futureDate.setDate(futureDate.getDate() + i);
                
                const predictedPrice = lastPrice * Math.pow(trendFactor, i) * (1 + (Math.random() - 0.5) * 0.05);
                
                predictions.push({
                    date: futureDate.toISOString().split('T')[0],
                    predicted_price: Math.round(predictedPrice),
                    confidence: Math.max(60 - i * 5, 30) // Decreasing confidence over time
                });
            }
        }

        // Market recommendations
        const currentPrice = priceHistory[priceHistory.length - 1].price;
        const avgPrice = priceHistory.map(p => p.price).reduce((a, b) => a + b, 0) / priceHistory.length;
        
        const recommendations = [];
        if (currentPrice > avgPrice * 1.1) {
            recommendations.push({
                action: "sell",
                reason: `Current price (₹${currentPrice}) is ${Math.round(((currentPrice - avgPrice) / avgPrice) * 100)}% above average`,
                urgency: "high"
            });
        } else if (currentPrice < avgPrice * 0.9) {
            recommendations.push({
                action: "hold",
                reason: `Current price (₹${currentPrice}) is below average. Wait for better rates`,
                urgency: "medium"
            });
        } else {
            recommendations.push({
                action: "monitor",
                reason: "Price is near average. Monitor for trend changes",
                urgency: "low"
            });
        }

        // Best markets (simulate multiple mandis)
        const nearbyMarkets = [
            { name: `${district || 'Local'} Mandi`, price: currentPrice, distance: 0 },
            { name: `${state || 'State'} Central Market`, price: Math.round(currentPrice * 1.05), distance: 25 },
            { name: "Regional Hub Market", price: Math.round(currentPrice * 1.08), distance: 45 }
        ].sort((a, b) => b.price - a.price);

        return {
            success: true,
            data: {
                commodity: commodity || "Unknown",
                location: { state, district },
                current_price: currentPrice,
                price_history: priceHistory.slice(-14), // Last 2 weeks
                trend_analysis: {
                    direction: trend,
                    percentage_change: Math.round(trendPercentage * 100) / 100,
                    period: "7 days"
                },
                predictions: predictions,
                recommendations: recommendations,
                best_markets: nearbyMarkets,
                market_summary: {
                    highest_price: Math.max(...priceHistory.map(p => p.price)),
                    lowest_price: Math.min(...priceHistory.map(p => p.price)),
                    average_price: Math.round(avgPrice),
                    total_volume: priceHistory.reduce((sum, p) => sum + p.volume, 0)
                },
                timestamp: new Date().toISOString(),
                source: "Multi-Mandi Price Aggregation simulation"
            }
        };
    } catch (err) {
        return { error: `Mandi price tracking error: ${String(err)}` };
    }
};

/**
 * EXA Search Tool Handler
 */
const searchHandler = async (params: any) => {
    try {
        const API_KEY = process.env.EXA_API_KEY;

        if (!API_KEY) {
            return {
                error: "Configuration error: EXA_API_KEY not set in environment."
            };
        }

        const { query, num_results = 5, include_domains, exclude_domains, start_crawl_date, end_crawl_date } = params;

        // Build EXA API request
        const requestBody: any = {
            query,
            num_results,
            use_autoprompt: true,
            contents: {
                text: true
            }
        };

        if (include_domains) requestBody.include_domains = include_domains;
        if (exclude_domains) requestBody.exclude_domains = exclude_domains;
        if (start_crawl_date) requestBody.start_crawl_date = start_crawl_date;
        if (end_crawl_date) requestBody.end_crawl_date = end_crawl_date;

        const res = await fetch("https://api.exa.ai/search", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "x-api-key": API_KEY
            },
            body: JSON.stringify(requestBody)
        });

        const text = await res.text();

        if (!res.ok) {
            return {
                error: `HTTP ${res.status} fetching EXA API: ${text}`
            };
        }

        let json;
        try {
            json = JSON.parse(text);
        } catch (err) {
            return { error: `Invalid JSON response from EXA: ${text}` };
        }

        // Format response
        const results = json.results || [];

        return {
            success: true,
            data: {
                results: results.map((result: any) => ({
                    title: result.title,
                    url: result.url,
                    text: result.text?.substring(0, 500) + (result.text?.length > 500 ? '...' : ''),
                    score: result.score,
                    published_date: result.published_date
                })),
                total_results: results.length,
                query
            }
        };
    } catch (err) {
        return { error: `Server error: ${String(err)}` };
    }
};

// Store tool handlers
const toolHandlers = new Map();
toolHandlers.set('crop-price', cropPriceHandler);
toolHandlers.set('search', searchHandler);
toolHandlers.set('soil-health', soilHealthHandler);
toolHandlers.set('weather', weatherHandler);
toolHandlers.set('pest-identifier', pestIdentifierHandler);
toolHandlers.set('mandi-price', mandiPriceHandler);

// MCP Tool Definitions
const mcpTools: MCPTool[] = [
    {
        name: "crop-price",
        description: "Fetch crop price data from data.gov.in with state/district/commodity filters",
        inputSchema: {
            type: "object",
            properties: {
                state: {
                    type: "string",
                    description: "State filter (e.g., Punjab, Maharashtra)"
                },
                district: {
                    type: "string", 
                    description: "District filter (e.g., Ludhiana, Mumbai)"
                },
                commodity: {
                    type: "string",
                    description: "Commodity filter (e.g., Wheat, Rice, Cotton)"
                },
                limit: {
                    type: "number",
                    description: "Max records to return (default: 50)",
                    default: 50
                },
                offset: {
                    type: "number", 
                    description: "Records to skip (default: 0)",
                    default: 0
                }
            }
        }
    },
    {
        name: "search",
        description: "Search the web for agricultural information using EXA API",
        inputSchema: {
            type: "object",
            properties: {
                query: {
                    type: "string",
                    description: "Search query for agricultural information"
                },
                num_results: {
                    type: "number",
                    description: "Number of results to return (default: 5)",
                    default: 5
                },
                include_domains: {
                    type: "array",
                    items: { type: "string" },
                    description: "Domains to include in search"
                },
                exclude_domains: {
                    type: "array", 
                    items: { type: "string" },
                    description: "Domains to exclude from search"
                }
            },
            required: ["query"]
        }
    },
    {
        name: "soil-health",
        description: "Analyze soil health parameters and provide crop recommendations based on NPK, pH, and organic content",
        inputSchema: {
            type: "object",
            properties: {
                state: {
                    type: "string",
                    description: "State where soil sample is from"
                },
                district: {
                    type: "string",
                    description: "District where soil sample is from"
                },
                soil_type: {
                    type: "string",
                    description: "Type of soil (e.g., Alluvial, Black Cotton, Red, Laterite)"
                },
                npk_values: {
                    type: "object",
                    properties: {
                        nitrogen: { type: "number", description: "Nitrogen content (kg/ha)" },
                        phosphorus: { type: "number", description: "Phosphorus content (kg/ha)" },
                        potassium: { type: "number", description: "Potassium content (kg/ha)" }
                    },
                    description: "NPK values from soil test"
                },
                ph_level: {
                    type: "number",
                    description: "Soil pH level (0-14 scale)"
                },
                organic_content: {
                    type: "number",
                    description: "Organic carbon content percentage"
                }
            }
        }
    },
    {
        name: "weather",
        description: "Get farming-specific weather forecast with irrigation and pest risk alerts",
        inputSchema: {
            type: "object",
            properties: {
                location: {
                    type: "string",
                    description: "Location for weather forecast (city, district, or coordinates)"
                },
                days: {
                    type: "number",
                    description: "Number of days to forecast (default: 7, max: 14)",
                    default: 7
                },
                include_farming_alerts: {
                    type: "boolean",
                    description: "Include farming-specific alerts and recommendations",
                    default: true
                }
            },
            required: ["location"]
        }
    },
    {
        name: "pest-identifier",
        description: "Identify crop pests and diseases based on symptoms and provide treatment recommendations",
        inputSchema: {
            type: "object",
            properties: {
                crop: {
                    type: "string",
                    description: "Type of crop affected (e.g., rice, wheat, cotton, maize)"
                },
                symptoms: {
                    type: "string",
                    description: "Comma-separated list of observed symptoms (e.g., 'yellowing leaves, holes in stem, stunted growth')"
                },
                image_description: {
                    type: "string",
                    description: "Description of what is visible in pest/disease images"
                },
                location: {
                    type: "string",
                    description: "Location where pest/disease is observed"
                }
            },
            required: ["crop", "symptoms"]
        }
    },
    {
        name: "mandi-price",
        description: "Track mandi prices with trends, predictions, and market recommendations",
        inputSchema: {
            type: "object",
            properties: {
                commodity: {
                    type: "string",
                    description: "Commodity to track (e.g., wheat, rice, cotton, maize, soybean)"
                },
                state: {
                    type: "string",
                    description: "State for price tracking"
                },
                district: {
                    type: "string",
                    description: "District for local mandi prices"
                },
                days_back: {
                    type: "number",
                    description: "Number of days of historical data (default: 30)",
                    default: 30
                },
                include_predictions: {
                    type: "boolean",
                    description: "Include 7-day price predictions",
                    default: true
                }
            },
            required: ["commodity"]
        }
    }
];

// MCP Protocol Handlers
const handleMCPRequest = async (request: MCPRequest): Promise<MCPResponse> => {
    try {
        switch (request.method) {
            case "initialize":
                return {
                    jsonrpc: "2.0",
                    id: request.id,
                    result: {
                        protocolVersion: "2024-11-05",
                        capabilities: {
                            tools: {}
                        },
                        serverInfo: {
                            name: "agricultural-ai-mcp",
                            version: "1.0.0"
                        }
                    }
                };

            case "tools/list":
                return {
                    jsonrpc: "2.0",
                    id: request.id,
                    result: {
                        tools: mcpTools
                    }
                };

            case "tools/call":
                const { name, arguments: args } = request.params;
                const handler = toolHandlers.get(name);
                
                if (!handler) {
                    return {
                        jsonrpc: "2.0",
                        id: request.id,
                        error: {
                            code: -32601,
                            message: `Tool '${name}' not found`,
                            data: { availableTools: Array.from(toolHandlers.keys()) }
                        }
                    };
                }

                const result = await handler(args);
                
                if (result.error) {
                    return {
                        jsonrpc: "2.0",
                        id: request.id,
                        error: {
                            code: -32603,
                            message: result.error
                        }
                    };
                }

                return {
                    jsonrpc: "2.0",
                    id: request.id,
                    result: {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify(result.data, null, 2)
                            }
                        ]
                    }
                };

            default:
                return {
                    jsonrpc: "2.0",
                    id: request.id,
                    error: {
                        code: -32601,
                        message: `Method '${request.method}' not found`
                    }
                };
        }
    } catch (error) {
        return {
            jsonrpc: "2.0",
            id: request.id,
            error: {
                code: -32603,
                message: `Internal error: ${String(error)}`
            }
        };
    }
};

// Hybrid Server: HTTP + MCP Protocol Support
const httpServer = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    // Enable CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }

    // MCP Protocol Endpoint (for Docker MCP Gateway)
    if (req.url === '/mcp' && req.method === 'POST') {
        let body = '';
        req.on('data', (chunk: any) => {
            body += chunk.toString();
        });

        req.on('end', async () => {
            try {
                const mcpRequest: MCPRequest = JSON.parse(body);
                const mcpResponse = await handleMCPRequest(mcpRequest);
                
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(mcpResponse));
            } catch (error) {
                const errorResponse: MCPResponse = {
                    jsonrpc: "2.0",
                    id: 0,
                    error: {
                        code: -32700,
                        message: `Parse error: ${String(error)}`
                    }
                };
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(errorResponse));
            }
        });
        return;
    }

    // Health check endpoint
    if (req.url === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            status: 'healthy',
            server: 'agricultural-ai-mcp',
            protocols: ['http', 'mcp'],
            tools: ['crop-price', 'search', 'soil-health', 'weather', 'pest-identifier', 'mandi-price'],
            timestamp: new Date().toISOString(),
            environment: {
                datagovin_key_set: !!process.env.DATAGOVIN_API_KEY,
                exa_key_set: !!process.env.EXA_API_KEY,
                port: PORT
            },
            mcp: {
                endpoint: '/mcp',
                protocol_version: '2024-11-05',
                capabilities: ['tools']
            }
        }));
        return;
    }

    // Root endpoint with API info
    if (req.url === '/') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            name: 'Agricultural AI MCP Server',
            version: '1.0.0',
            description: 'Hybrid server: HTTP API + MCP Protocol for Docker MCP Gateway integration',
            protocols: {
                http: 'Direct HTTP API access',
                mcp: 'Model Context Protocol via /mcp endpoint'
            },
            tools: [
                {
                    name: 'crop-price',
                    description: 'Fetch crop price data from data.gov.in',
                    endpoint: '/tools/crop-price',
                    method: 'POST',
                    parameters: {
                        state: 'string (optional) - State filter (e.g., Punjab, Maharashtra)',
                        district: 'string (optional) - District filter (e.g., Ludhiana, Mumbai)',
                        commodity: 'string (optional) - Commodity filter (e.g., Wheat, Rice, Cotton)',
                        limit: 'number (optional) - Max records to return (default: 50)',
                        offset: 'number (optional) - Records to skip (default: 0)'
                    }
                },
                {
                    name: 'search',
                    description: 'Search the web for agricultural information',
                    endpoint: '/tools/search',
                    method: 'POST',
                    parameters: {
                        query: 'string (required) - Search query',
                        num_results: 'number (optional) - Number of results (default: 5)',
                        include_domains: 'array (optional) - Domains to include',
                        exclude_domains: 'array (optional) - Domains to exclude'
                    }
                },
                {
                    name: 'soil-health',
                    description: 'Analyze soil health and provide crop recommendations',
                    endpoint: '/tools/soil-health',
                    method: 'POST',
                    parameters: {
                        state: 'string (optional) - State location',
                        district: 'string (optional) - District location',
                        soil_type: 'string (optional) - Type of soil',
                        npk_values: 'object (optional) - NPK values from soil test',
                        ph_level: 'number (optional) - Soil pH level',
                        organic_content: 'number (optional) - Organic carbon percentage'
                    }
                },
                {
                    name: 'weather',
                    description: 'Get farming-specific weather forecast with alerts',
                    endpoint: '/tools/weather',
                    method: 'POST',
                    parameters: {
                        location: 'string (required) - Location for forecast',
                        days: 'number (optional) - Days to forecast (default: 7)',
                        include_farming_alerts: 'boolean (optional) - Include farming alerts'
                    }
                },
                {
                    name: 'pest-identifier',
                    description: 'Identify pests and diseases with treatment recommendations',
                    endpoint: '/tools/pest-identifier',
                    method: 'POST',
                    parameters: {
                        crop: 'string (required) - Type of crop affected',
                        symptoms: 'string (required) - Observed symptoms',
                        image_description: 'string (optional) - Description of images',
                        location: 'string (optional) - Location of observation'
                    }
                },
                {
                    name: 'mandi-price',
                    description: 'Track mandi prices with trends and predictions',
                    endpoint: '/tools/mandi-price',
                    method: 'POST',
                    parameters: {
                        commodity: 'string (required) - Commodity to track',
                        state: 'string (optional) - State for tracking',
                        district: 'string (optional) - District for local prices',
                        days_back: 'number (optional) - Historical data days',
                        include_predictions: 'boolean (optional) - Include predictions'
                    }
                }
            ],
            usage: {
                http: 'POST to /tools/{tool-name} with JSON body containing tool parameters',
                mcp: 'POST to /mcp with MCP protocol JSON-RPC requests'
            },
            examples: {
                'crop-price': {
                    url: '/tools/crop-price',
                    method: 'POST',
                    body: { state: 'Punjab', commodity: 'Wheat', limit: 10 }
                },
                'search': {
                    url: '/tools/search',
                    method: 'POST',
                    body: { query: 'Indian agriculture news 2024', num_results: 5 }
                },
                'soil-health': {
                    url: '/tools/soil-health',
                    method: 'POST',
                    body: { state: 'Punjab', ph_level: 6.5, npk_values: { nitrogen: 280, phosphorus: 23, potassium: 280 } }
                },
                'weather': {
                    url: '/tools/weather',
                    method: 'POST',
                    body: { location: 'Ludhiana, Punjab', days: 7, include_farming_alerts: true }
                },
                'pest-identifier': {
                    url: '/tools/pest-identifier',
                    method: 'POST',
                    body: { crop: 'rice', symptoms: 'yellowing leaves, stunted growth', location: 'Punjab' }
                },
                'mandi-price': {
                    url: '/tools/mandi-price',
                    method: 'POST',
                    body: { commodity: 'wheat', state: 'Punjab', district: 'Ludhiana', include_predictions: true }
                }
            }
        }));
        return;
    }

    // Tool endpoints
    if (req.url?.startsWith('/tools/') && req.method === 'POST') {
        const toolName = req.url.split('/tools/')[1];

        let body = '';
        req.on('data', (chunk: any) => {
            body += chunk.toString();
        });

        req.on('end', async () => {
            try {
                const params = JSON.parse(body || '{}');
                const handler = toolHandlers.get(toolName);

                if (!handler) {
                    res.writeHead(404, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({
                        error: `Tool '${toolName}' not found`,
                        available_tools: Array.from(toolHandlers.keys())
                    }));
                    return;
                }

                const result = await handler(params);
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(result));
            } catch (error) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: `Invalid request: ${error}` }));
            }
        });
        return;
    }

    // 404 for other routes
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
});

// Start HTTP server
httpServer.listen(PORT, () => {
    const isProduction = process.env.NODE_ENV === 'production';
    const baseUrl = isProduction ? 'https://fs-gate.onrender.com' : `http://localhost:${PORT}`;

    console.log(`🚀 Agricultural AI MCP Server running on port ${PORT}`);
    console.log(`🌾 HTTP: Crop price tool: ${baseUrl}/tools/crop-price`);
    console.log(`🔍 HTTP: Search tool: ${baseUrl}/tools/search`);
    console.log(`🤖 MCP: Protocol endpoint: ${baseUrl}/mcp`);
    console.log(`❤️  Health check: ${baseUrl}/health`);
    console.log(`📖 API docs: ${baseUrl}/`);

    if (isProduction) {
        console.log(`🎯 Live with Docker MCP Gateway support! Ready for hackathon!`);
    } else {
        console.log(`🎯 Ready for Docker MCP Gateway integration and cloud deployment!`);
    }
});