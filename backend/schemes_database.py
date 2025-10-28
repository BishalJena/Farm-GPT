"""
Static schemes database for government agricultural schemes and subsidies
Hackathon-ready mock data for Indian agricultural schemes
"""

from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta

class SchemesDatabase:
    """Static database of government agricultural schemes and subsidies"""
    
    def __init__(self):
        self.schemes = {
            "pm_kisan": {
                "id": "pm_kisan",
                "name": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "cash",
                "benefit_amount": 6000,
                "benefit_description": "₹6,000 per year in 3 equal installments of ₹2,000 each",
                "description": "Direct income support scheme for small and marginal farmers",
                "eligibility_criteria": [
                    "All landholding farmers (small and marginal)",
                    "Land records should be in farmer's name",
                    "Applicable across all states and UTs",
                    "No income limit for eligibility"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Bank Account Details",
                    "Land Ownership Documents",
                    "Mobile Number"
                ],
                "application_process": [
                    "Visit PM-KISAN portal or nearest CSC",
                    "Fill registration form with required details",
                    "Upload necessary documents",
                    "Submit application and get acknowledgment"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 0, "max": None},
                "deadline": None,
                "website_url": "https://pmkisan.gov.in",
                "enrollment_rate": 0.15  # 15% farmers typically enrolled
            },
            "pmfby": {
                "id": "pmfby",
                "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "insurance",
                "benefit_amount": 200000,
                "benefit_description": "Crop insurance coverage up to ₹2 lakh per hectare",
                "description": "Comprehensive crop insurance scheme for all farmers",
                "eligibility_criteria": [
                    "All farmers growing notified crops",
                    "Both loanee and non-loanee farmers eligible",
                    "Premium: 2% for Kharif, 1.5% for Rabi crops",
                    "Covers pre-sowing to post-harvest losses"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Bank Account Details", 
                    "Land Records",
                    "Sowing Certificate",
                    "Premium Payment Receipt"
                ],
                "application_process": [
                    "Apply through bank, CSC, or insurance company",
                    "Submit before cut-off date (varies by crop/state)",
                    "Pay farmer's share of premium",
                    "Get policy document and coverage details"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 0, "max": None},
                "deadline": "2025-03-31",
                "website_url": "https://pmfby.gov.in",
                "enrollment_rate": 0.10  # 10% farmers typically enrolled
            },
            "soil_health_card": {
                "id": "soil_health_card",
                "name": "Soil Health Card Scheme",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "subsidy",
                "benefit_amount": 0,
                "benefit_description": "Free soil testing and health card (worth ₹500-1000)",
                "description": "Free soil testing for farmers every 3 years",
                "eligibility_criteria": [
                    "All farmers eligible",
                    "One soil health card per 2.5 acres",
                    "Valid for 3 years",
                    "Covers 12 soil parameters"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Mobile Number",
                    "Soil Sample"
                ],
                "application_process": [
                    "Contact local agriculture office",
                    "Collect soil sample as per guidelines",
                    "Submit sample at designated center",
                    "Receive soil health card within 30 days"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 0, "max": None},
                "deadline": None,
                "website_url": "https://soilhealth.dac.gov.in",
                "enrollment_rate": 0.05  # 5% farmers typically enrolled
            },
            "kisan_credit_card": {
                "id": "kisan_credit_card",
                "name": "Kisan Credit Card (KCC)",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "credit",
                "benefit_amount": 300000,
                "benefit_description": "Credit limit up to ₹3 lakh at subsidized interest rates",
                "description": "Flexible credit facility for farmers' cultivation and other needs",
                "eligibility_criteria": [
                    "All farmers (individual/joint borrowers)",
                    "Tenant farmers, oral lessees, and sharecroppers",
                    "Self Help Group members",
                    "Interest subvention available"
                ],
                "required_documents": [
                    "Application Form",
                    "Identity Proof (Aadhaar)",
                    "Address Proof",
                    "Land Documents",
                    "Income Proof"
                ],
                "application_process": [
                    "Visit nearest bank branch",
                    "Fill KCC application form",
                    "Submit required documents",
                    "Bank verification and approval",
                    "Receive KCC within 15 days"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 0, "max": None},
                "deadline": None,
                "website_url": "https://www.nabard.org/kcc.aspx",
                "enrollment_rate": 0.08  # 8% farmers typically enrolled
            },
            "organic_farming_scheme": {
                "id": "organic_farming_scheme",
                "name": "National Programme for Organic Production (NPOP)",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "subsidy",
                "benefit_amount": 50000,
                "benefit_description": "₹50,000 per hectare for organic farming conversion",
                "description": "Support for farmers to convert to organic farming practices",
                "eligibility_criteria": [
                    "Farmers willing to convert to organic farming",
                    "Minimum 1 hectare land",
                    "3-year conversion period commitment",
                    "Group certification preferred"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Bank Account Details",
                    "Organic Farming Plan"
                ],
                "application_process": [
                    "Contact local agriculture department",
                    "Submit organic farming conversion plan",
                    "Get land inspection done",
                    "Receive certification and subsidy"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 1, "max": None},
                "deadline": "2025-06-30",
                "website_url": "https://www.apeda.gov.in/apedawebsite/organic/Organic_Products.htm",
                "enrollment_rate": 0.25
            },
            "micro_irrigation_scheme": {
                "id": "micro_irrigation_scheme", 
                "name": "Per Drop More Crop (Micro Irrigation)",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "subsidy",
                "benefit_amount": 80000,
                "benefit_description": "Up to 55% subsidy on drip/sprinkler irrigation systems",
                "description": "Subsidy for water-efficient irrigation systems",
                "eligibility_criteria": [
                    "All categories of farmers",
                    "Minimum 0.5 hectare land",
                    "Water source availability required",
                    "Technical feasibility assessment"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Water Source Certificate",
                    "Technical Estimate",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Apply through state agriculture department",
                    "Get technical assessment done",
                    "Install system through approved vendor",
                    "Claim subsidy after verification"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 0.5, "max": None},
                "deadline": "2025-03-31",
                "website_url": "https://pmksy.gov.in",
                "enrollment_rate": 0.30
            },
            "wheat_procurement_scheme": {
                "id": "wheat_procurement_scheme",
                "name": "Minimum Support Price for Wheat",
                "category": "central", 
                "department": "Food Corporation of India",
                "benefit_type": "price_support",
                "benefit_amount": 2275,
                "benefit_description": "Guaranteed price of ₹2,275 per quintal for wheat",
                "description": "Minimum support price guarantee for wheat farmers",
                "eligibility_criteria": [
                    "Wheat farmers in procurement states",
                    "Quality specifications must be met",
                    "Valid land records required",
                    "Moisture content below 12%"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Crop Cutting Certificate",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Register at nearest procurement center",
                    "Get crop quality tested",
                    "Sell wheat at MSP rate",
                    "Receive payment within 72 hours"
                ],
                "target_states": ["punjab", "haryana", "uttar-pradesh", "madhya-pradesh", "rajasthan"],
                "land_size_criteria": {"min": 0, "max": None},
                "deadline": "2025-05-31",
                "website_url": "https://fci.gov.in",
                "enrollment_rate": 0.12
            },
            "rice_procurement_scheme": {
                "id": "rice_procurement_scheme",
                "name": "Minimum Support Price for Rice",
                "category": "central",
                "department": "Food Corporation of India", 
                "benefit_type": "price_support",
                "benefit_amount": 2300,
                "benefit_description": "Guaranteed price of ₹2,300 per quintal for paddy",
                "description": "Minimum support price guarantee for rice farmers",
                "eligibility_criteria": [
                    "Rice farmers in procurement states",
                    "Quality specifications must be met",
                    "Valid land records required",
                    "Moisture content below 17%"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records", 
                    "Crop Cutting Certificate",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Register at nearest procurement center",
                    "Get crop quality tested",
                    "Sell paddy at MSP rate",
                    "Receive payment within 72 hours"
                ],
                "target_states": ["punjab", "haryana", "uttar-pradesh", "west-bengal", "odisha", "andhra-pradesh"],
                "land_size_criteria": {"min": 0, "max": None},
                "deadline": "2025-02-28",
                "website_url": "https://fci.gov.in",
                "enrollment_rate": 0.15
            },
            "cotton_technology_mission": {
                "id": "cotton_technology_mission",
                "name": "Technology Mission on Cotton",
                "category": "central",
                "department": "Ministry of Textiles",
                "benefit_type": "subsidy",
                "benefit_amount": 25000,
                "benefit_description": "₹25,000 per hectare for cotton productivity enhancement",
                "description": "Support for improving cotton productivity and quality",
                "eligibility_criteria": [
                    "Cotton farmers in designated states",
                    "Minimum 2 hectare cotton cultivation",
                    "Use of certified seeds mandatory",
                    "Integrated pest management adoption"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Seed Purchase Receipt",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Apply through cotton development office",
                    "Submit cultivation plan",
                    "Get field inspection done",
                    "Receive subsidy in installments"
                ],
                "target_states": ["gujarat", "maharashtra", "andhra-pradesh", "telangana", "karnataka"],
                "land_size_criteria": {"min": 2, "max": None},
                "deadline": "2025-07-31",
                "website_url": "https://texmin.nic.in",
                "enrollment_rate": 0.05
            },
            "sugarcane_development_scheme": {
                "id": "sugarcane_development_scheme",
                "name": "Sustainable Sugarcane Initiative",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "subsidy",
                "benefit_amount": 35000,
                "benefit_description": "₹35,000 per hectare for sustainable sugarcane farming",
                "description": "Support for sustainable sugarcane cultivation practices",
                "eligibility_criteria": [
                    "Sugarcane farmers",
                    "Minimum 1 hectare sugarcane area",
                    "Water-efficient irrigation methods",
                    "Soil health card mandatory"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Soil Health Card",
                    "Water Source Certificate",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Apply through sugar development office",
                    "Submit sustainable farming plan",
                    "Install recommended practices",
                    "Get verification and receive subsidy"
                ],
                "target_states": ["uttar-pradesh", "maharashtra", "karnataka", "tamil-nadu", "gujarat"],
                "land_size_criteria": {"min": 1, "max": None},
                "deadline": "2025-04-30",
                "website_url": "https://dfpd.gov.in",
                "enrollment_rate": 0.08
            },
            "maize_development_program": {
                "id": "maize_development_program",
                "name": "National Food Security Mission - Maize",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "subsidy",
                "benefit_amount": 20000,
                "benefit_description": "₹20,000 per hectare for maize productivity enhancement",
                "description": "Support for increasing maize production and productivity",
                "eligibility_criteria": [
                    "Maize farmers in focus districts",
                    "Minimum 0.5 hectare maize cultivation",
                    "Use of hybrid seeds",
                    "Balanced fertilizer application"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Seed Purchase Receipt",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Register with agriculture department",
                    "Submit maize cultivation plan",
                    "Implement recommended practices",
                    "Get field verification and subsidy"
                ],
                "target_states": ["karnataka", "andhra-pradesh", "telangana", "maharashtra", "rajasthan"],
                "land_size_criteria": {"min": 0.5, "max": None},
                "deadline": "2025-08-31",
                "website_url": "https://nfsm.gov.in",
                "enrollment_rate": 0.06
            },
            "vegetable_cluster_development": {
                "id": "vegetable_cluster_development",
                "name": "Vegetable Cluster Development Programme",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "subsidy",
                "benefit_amount": 75000,
                "benefit_description": "₹75,000 per hectare for vegetable cluster development",
                "description": "Support for developing vegetable production clusters",
                "eligibility_criteria": [
                    "Vegetable farmers in cluster areas",
                    "Minimum 5 hectare cluster size",
                    "Group farming approach",
                    "Market linkage commitment"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Group Formation Certificate",
                    "Market Linkage Agreement",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Form farmer producer group",
                    "Apply through horticulture department",
                    "Submit cluster development plan",
                    "Implement activities and claim subsidy"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 0.2, "max": None},
                "deadline": "2025-09-30",
                "website_url": "https://midh.gov.in",
                "enrollment_rate": 0.04
            },
            "pradhan_mantri_krishi_sinchai": {
                "id": "pradhan_mantri_krishi_sinchai",
                "name": "Pradhan Mantri Krishi Sinchayee Yojana (PMKSY)",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "subsidy",
                "benefit_amount": 50000,
                "benefit_description": "Up to 55% subsidy on drip/sprinkler irrigation systems",
                "description": "Irrigation efficiency improvement scheme",
                "eligibility_criteria": [
                    "All categories of farmers",
                    "Minimum 0.5 hectare land holding",
                    "Water source availability required",
                    "Priority to SC/ST and small farmers"
                ],
                "required_documents": [
                    "Application Form",
                    "Land Documents",
                    "Water Source Certificate",
                    "Bank Account Details",
                    "Aadhaar Card"
                ],
                "application_process": [
                    "Apply through state agriculture department",
                    "Technical feasibility assessment",
                    "Approval and subsidy sanction",
                    "Installation by approved vendors",
                    "Subsidy release after verification"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 0.5, "max": None},
                "deadline": "2025-06-30",
                "website_url": "https://pmksy.gov.in",
                "enrollment_rate": 0.25  # 25% farmers typically enrolled
            },
            "national_agriculture_market": {
                "id": "national_agriculture_market",
                "name": "National Agriculture Market (e-NAM)",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "platform",
                "benefit_amount": 0,
                "benefit_description": "Online trading platform for better price discovery",
                "description": "Unified national market for agricultural commodities",
                "eligibility_criteria": [
                    "All farmers can register",
                    "Valid mobile number required",
                    "Bank account mandatory",
                    "Quality testing facilities available"
                ],
                "required_documents": [
                    "Mobile Number",
                    "Bank Account Details",
                    "Aadhaar Card",
                    "Land Records (optional)"
                ],
                "application_process": [
                    "Register on e-NAM portal",
                    "Mobile verification",
                    "Upload required documents",
                    "Start trading after approval"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 0, "max": None},
                "deadline": None,
                "website_url": "https://enam.gov.in",
                "enrollment_rate": 0.20  # 20% farmers typically enrolled
            },
            "paramparagat_krishi_vikas": {
                "id": "paramparagat_krishi_vikas",
                "name": "Paramparagat Krishi Vikas Yojana (PKVY)",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "subsidy",
                "benefit_amount": 50000,
                "benefit_description": "₹50,000 per hectare for 3 years for organic farming",
                "description": "Promotion of organic farming practices",
                "eligibility_criteria": [
                    "Farmers willing to adopt organic farming",
                    "Group formation of 50 farmers minimum",
                    "50 hectare cluster area",
                    "3-year commitment required"
                ],
                "required_documents": [
                    "Group Formation Certificate",
                    "Land Records",
                    "Organic Farming Plan",
                    "Bank Account Details",
                    "Aadhaar Cards of all members"
                ],
                "application_process": [
                    "Form farmer groups (FPO/SHG)",
                    "Apply through state agriculture department",
                    "Cluster identification and approval",
                    "Training and input distribution",
                    "Certification and marketing support"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 1, "max": None},
                "deadline": "2025-04-30",
                "website_url": "https://pgsindia-ncof.gov.in",
                "enrollment_rate": 0.15  # 15% farmers typically enrolled
            },
            "rashtriya_krishi_vikas": {
                "id": "rashtriya_krishi_vikas",
                "name": "Rashtriya Krishi Vikas Yojana (RKVY)",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "subsidy",
                "benefit_amount": 100000,
                "benefit_description": "Various subsidies for agriculture infrastructure development",
                "description": "State-specific agriculture development schemes",
                "eligibility_criteria": [
                    "Farmers, FPOs, and cooperatives",
                    "State-specific implementation",
                    "Infrastructure development focus",
                    "Technology adoption support"
                ],
                "required_documents": [
                    "Project Proposal",
                    "Land Documents",
                    "Technical Feasibility Report",
                    "Bank Account Details",
                    "Registration Certificates"
                ],
                "application_process": [
                    "Submit project proposal to state government",
                    "Technical and financial evaluation",
                    "Approval by state committee",
                    "Implementation and monitoring",
                    "Subsidy release in installments"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 1, "max": None},
                "deadline": "2025-03-31",
                "website_url": "https://rkvy.nic.in",
                "enrollment_rate": 0.10  # 10% farmers typically enrolled
            },
            # Punjab-specific schemes
            "punjab_crop_diversification": {
                "id": "punjab_crop_diversification",
                "name": "Punjab Crop Diversification Scheme",
                "category": "state",
                "department": "Punjab Agriculture Department",
                "benefit_type": "subsidy",
                "benefit_amount": 15000,
                "benefit_description": "₹15,000 per hectare for shifting from paddy to alternative crops",
                "description": "Incentive for crop diversification in Punjab",
                "eligibility_criteria": [
                    "Punjab farmers only",
                    "Must shift from paddy cultivation",
                    "Alternative crops: maize, cotton, sugarcane, basmati",
                    "Minimum 1 hectare area"
                ],
                "required_documents": [
                    "Land Records (Punjab)",
                    "Previous Year Crop Details",
                    "Aadhaar Card",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Apply at district agriculture office",
                    "Verification of previous paddy cultivation",
                    "Crop planning and approval",
                    "Subsidy release after sowing verification"
                ],
                "target_states": ["punjab"],
                "land_size_criteria": {"min": 1, "max": None},
                "deadline": "2025-05-31",
                "website_url": "https://agri.punjab.gov.in",
                "enrollment_rate": 0.30  # 30% Punjab farmers enrolled
            },
            "punjab_farm_mechanization": {
                "id": "punjab_farm_mechanization",
                "name": "Punjab Farm Mechanization Scheme",
                "category": "state",
                "department": "Punjab Agriculture Department",
                "benefit_type": "subsidy",
                "benefit_amount": 200000,
                "benefit_description": "Up to 50% subsidy on agricultural machinery",
                "description": "Subsidy for purchase of agricultural equipment",
                "eligibility_criteria": [
                    "Punjab farmers and FPOs",
                    "Priority to small and marginal farmers",
                    "Custom hiring centers eligible",
                    "One-time benefit per farmer"
                ],
                "required_documents": [
                    "Application Form",
                    "Land Records",
                    "Machinery Quotation",
                    "Bank Account Details",
                    "Caste Certificate (if applicable)"
                ],
                "application_process": [
                    "Online application on Punjab Agri portal",
                    "Document verification",
                    "Approval and purchase authorization",
                    "Purchase from approved dealers",
                    "Subsidy credit to bank account"
                ],
                "target_states": ["punjab"],
                "land_size_criteria": {"min": 0.5, "max": None},
                "deadline": "2025-08-31",
                "website_url": "https://agri.punjab.gov.in",
                "enrollment_rate": 0.40  # 40% Punjab farmers enrolled
            },
            "pulses_production_scheme": {
                "id": "pulses_production_scheme",
                "name": "National Food Security Mission - Pulses",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "subsidy",
                "benefit_amount": 15000,
                "benefit_description": "₹15,000 per hectare for pulses cultivation",
                "description": "Support for increasing pulses production and productivity",
                "eligibility_criteria": [
                    "Farmers growing pulses in focus districts",
                    "Minimum 0.4 hectare pulses cultivation",
                    "Use of certified seeds",
                    "Integrated nutrient management"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Seed Purchase Receipt",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Apply through agriculture department",
                    "Submit pulses cultivation plan",
                    "Implement recommended practices",
                    "Get verification and receive subsidy"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 0.4, "max": None},
                "deadline": "2025-07-15",
                "website_url": "https://nfsm.gov.in",
                "enrollment_rate": 0.45
            },
            "oilseeds_development_program": {
                "id": "oilseeds_development_program", 
                "name": "National Mission on Oilseeds and Oil Palm",
                "category": "central",
                "department": "Ministry of Agriculture & Farmers Welfare",
                "benefit_type": "subsidy",
                "benefit_amount": 18000,
                "benefit_description": "₹18,000 per hectare for oilseeds cultivation",
                "description": "Support for increasing oilseeds production",
                "eligibility_criteria": [
                    "Farmers growing oilseeds",
                    "Minimum 0.5 hectare oilseeds area",
                    "Quality seed usage mandatory",
                    "Soil testing certificate required"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Soil Health Card",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Register with oilseeds development office",
                    "Submit cultivation plan",
                    "Purchase inputs from approved sources",
                    "Claim subsidy after harvest verification"
                ],
                "target_states": ["rajasthan", "gujarat", "madhya-pradesh", "maharashtra", "karnataka"],
                "land_size_criteria": {"min": 0.5, "max": None},
                "deadline": "2025-06-30",
                "website_url": "https://nmoop.gov.in",
                "enrollment_rate": 0.35
            },
            "spices_development_scheme": {
                "id": "spices_development_scheme",
                "name": "Spices Development Programme",
                "category": "central",
                "department": "Spices Board of India",
                "benefit_type": "subsidy",
                "benefit_amount": 40000,
                "benefit_description": "₹40,000 per hectare for spices cultivation",
                "description": "Support for spices cultivation and quality improvement",
                "eligibility_criteria": [
                    "Spices farmers",
                    "Minimum 0.25 hectare spices cultivation",
                    "Quality planting material usage",
                    "Organic certification preferred"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Spices Cultivation Plan",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Apply through Spices Board office",
                    "Submit detailed cultivation plan",
                    "Get technical guidance",
                    "Receive subsidy in installments"
                ],
                "target_states": ["kerala", "karnataka", "tamil-nadu", "andhra-pradesh", "gujarat"],
                "land_size_criteria": {"min": 0.25, "max": None},
                "deadline": "2025-08-31",
                "website_url": "https://indianspices.com",
                "enrollment_rate": 0.30
            },
            "coconut_development_board_scheme": {
                "id": "coconut_development_board_scheme",
                "name": "Coconut Development Board Assistance",
                "category": "central",
                "department": "Coconut Development Board",
                "benefit_type": "subsidy",
                "benefit_amount": 60000,
                "benefit_description": "₹60,000 per hectare for coconut plantation",
                "description": "Support for coconut cultivation and productivity enhancement",
                "eligibility_criteria": [
                    "Farmers in coconut growing areas",
                    "Minimum 0.5 hectare for new plantation",
                    "Suitable soil and climate conditions",
                    "Water availability for irrigation"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Soil Test Report",
                    "Water Source Certificate",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Apply through CDB regional office",
                    "Get technical feasibility assessment",
                    "Plant coconut saplings as per guidelines",
                    "Receive subsidy in phases"
                ],
                "target_states": ["kerala", "tamil-nadu", "karnataka", "andhra-pradesh", "goa"],
                "land_size_criteria": {"min": 0.5, "max": None},
                "deadline": "2025-09-30",
                "website_url": "https://coconutboard.gov.in",
                "enrollment_rate": 0.40
            },
            "tea_development_scheme": {
                "id": "tea_development_scheme",
                "name": "Tea Development and Promotion Scheme",
                "category": "central",
                "department": "Tea Board of India",
                "benefit_type": "subsidy",
                "benefit_amount": 80000,
                "benefit_description": "₹80,000 per hectare for tea plantation development",
                "description": "Support for tea cultivation and quality improvement",
                "eligibility_criteria": [
                    "Tea growers in designated areas",
                    "Minimum 1 hectare tea plantation",
                    "Suitable altitude and climate",
                    "Quality planting material usage"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Tea Board Registration",
                    "Plantation Plan",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Register with Tea Board",
                    "Submit plantation development plan",
                    "Implement as per technical guidelines",
                    "Claim subsidy after verification"
                ],
                "target_states": ["assam", "west-bengal", "tamil-nadu", "kerala", "himachal-pradesh"],
                "land_size_criteria": {"min": 1, "max": None},
                "deadline": "2025-10-31",
                "website_url": "https://teaboard.gov.in",
                "enrollment_rate": 0.50
            },
            "rubber_plantation_scheme": {
                "id": "rubber_plantation_scheme",
                "name": "Rubber Plantation Development Scheme",
                "category": "central",
                "department": "Rubber Board",
                "benefit_type": "subsidy",
                "benefit_amount": 70000,
                "benefit_description": "₹70,000 per hectare for rubber plantation",
                "description": "Support for rubber cultivation in suitable areas",
                "eligibility_criteria": [
                    "Farmers in rubber growing regions",
                    "Minimum 2 hectare for new plantation",
                    "Suitable soil and rainfall conditions",
                    "Long-term commitment (25+ years)"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Soil Suitability Certificate",
                    "Plantation Plan",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Apply through Rubber Board office",
                    "Get land suitability assessment",
                    "Plant rubber saplings as per norms",
                    "Receive subsidy in installments"
                ],
                "target_states": ["kerala", "tamil-nadu", "karnataka", "goa", "assam"],
                "land_size_criteria": {"min": 2, "max": None},
                "deadline": "2025-07-31",
                "website_url": "https://rubberboard.org.in",
                "enrollment_rate": 0.25
            },
            "fisheries_development_scheme": {
                "id": "fisheries_development_scheme",
                "name": "Pradhan Mantri Matsya Sampada Yojana",
                "category": "central",
                "department": "Ministry of Fisheries, Animal Husbandry and Dairying",
                "benefit_type": "subsidy",
                "benefit_amount": 100000,
                "benefit_description": "Up to ₹1 lakh subsidy for fish farming",
                "description": "Support for fish farming and aquaculture development",
                "eligibility_criteria": [
                    "Fish farmers and entrepreneurs",
                    "Minimum 0.5 hectare water area",
                    "Technical feasibility clearance",
                    "Water quality suitable for fish farming"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land/Water Body Records",
                    "Water Quality Report",
                    "Project Report",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Apply through fisheries department",
                    "Submit detailed project report",
                    "Get technical approval",
                    "Implement project and claim subsidy"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 0.5, "max": None},
                "deadline": "2025-03-31",
                "website_url": "https://pmmsy.dof.gov.in",
                "enrollment_rate": 0.35
            },
            "dairy_development_scheme": {
                "id": "dairy_development_scheme",
                "name": "National Programme for Dairy Development",
                "category": "central",
                "department": "Ministry of Fisheries, Animal Husbandry and Dairying",
                "benefit_type": "subsidy",
                "benefit_amount": 50000,
                "benefit_description": "₹50,000 subsidy for dairy unit establishment",
                "description": "Support for dairy farming and milk production",
                "eligibility_criteria": [
                    "Dairy farmers and entrepreneurs",
                    "Minimum 2 milch animals",
                    "Adequate fodder arrangement",
                    "Veterinary care access"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land Records",
                    "Animal Purchase Receipt",
                    "Veterinary Certificate",
                    "Bank Account Details"
                ],
                "application_process": [
                    "Apply through animal husbandry department",
                    "Submit dairy project proposal",
                    "Get animals and infrastructure",
                    "Claim subsidy after verification"
                ],
                "target_states": ["all"],
                "land_size_criteria": {"min": 0.1, "max": None},
                "deadline": "2025-12-31",
                "website_url": "https://dahd.nic.in",
                "enrollment_rate": 0.45
            }
        }
    
    def get_all_schemes(self) -> List[Dict[str, Any]]:
        """Get all available schemes"""
        return list(self.schemes.values())
    
    def find_matching_schemes(self, farmer_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find schemes matching farmer's profile"""
        state = farmer_details.get('state', '').lower()
        district = farmer_details.get('district', '').lower()
        land_size = float(farmer_details.get('landSize', 0))
        crop_types = farmer_details.get('cropTypes', [])
        
        matching_schemes = []
        
        for scheme in self.schemes.values():
            # Check state eligibility
            if scheme['target_states'] != ['all'] and state not in scheme['target_states']:
                continue
            
            # Check land size criteria
            land_criteria = scheme.get('land_size_criteria', {})
            min_land = land_criteria.get('min', 0)
            max_land = land_criteria.get('max')
            
            if land_size < min_land:
                continue
            if max_land and land_size > max_land:
                continue
            
            # Add scheme with eligibility status
            scheme_copy = scheme.copy()
            scheme_copy['eligibility_status'] = 'eligible'
            matching_schemes.append(scheme_copy)
        
        return matching_schemes
    
    def generate_mock_enrollment_status(self, user_id: str, scheme_id: str) -> Dict[str, Any]:
        """Generate realistic mock enrollment status for a user and scheme"""
        scheme = self.schemes.get(scheme_id)
        if not scheme:
            return {"status": "not_found"}
        
        # All schemes start as eligible (not applied) - this is a scheme finder
        return {
            "status": "eligible",
            "enrollment_date": None,
            "amount_received": 0,
            "next_payment_date": None,
            "application_id": None,
            "total_received_this_year": 0
        }
    
    def get_user_enrollment_summary(self, user_id: str) -> Dict[str, Any]:
        """Get complete enrollment summary for a user"""
        total_received = 0
        total_pending = 0
        enrolled_count = 0
        eligible_count = len(self.schemes)
        upcoming_deadlines = []
        
        enrollment_data = {}
        
        for scheme_id, scheme in self.schemes.items():
            enrollment_status = self.generate_mock_enrollment_status(user_id, scheme_id)
            enrollment_data[scheme_id] = enrollment_status
            
            if enrollment_status['status'] == 'enrolled':
                enrolled_count += 1
                total_received += enrollment_status.get('amount_received', 0)
            
            elif enrollment_status['status'] == 'pending':
                total_pending += scheme.get('benefit_amount', 0) * 0.5  # Estimated pending amount
            
            # Check for upcoming deadlines
            if scheme.get('deadline'):
                try:
                    deadline_date = datetime.strptime(scheme['deadline'], "%Y-%m-%d")
                    days_left = (deadline_date - datetime.now()).days
                    if 0 < days_left <= 90:  # Next 90 days
                        upcoming_deadlines.append({
                            "scheme_name": scheme['name'],
                            "deadline": scheme['deadline'],
                            "days_left": days_left
                        })
                except:
                    pass
        
        return {
            "total_received_this_year": round(total_received, 2),
            "total_pending": round(total_pending, 2),
            "enrolled_schemes_count": enrolled_count,
            "available_schemes_count": eligible_count,
            "upcoming_deadlines": sorted(upcoming_deadlines, key=lambda x: x['days_left']),
            "enrollment_details": enrollment_data
        }

# Global instance for easy access
schemes_db = SchemesDatabase()