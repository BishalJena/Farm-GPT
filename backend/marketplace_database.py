"""
Marketplace database for surplus produce trading
Hackathon-ready mock data for B2B marketplace functionality
"""

from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta
import uuid

class MarketplaceDatabase:
    """Database for surplus marketplace functionality"""
    
    def __init__(self):
        # In-memory storage for hackathon demo
        self.listings = {}
        self.mock_buyers = [
            {
                "id": "buyer_001",
                "name": "Zomato Kitchens",
                "type": "restaurant",
                "location": "Ludhiana, Punjab",
                "contact": "+91-98765-43210",
                "email": "procurement@zomato.com",
                "rating": 4.8,
                "verified": True,
                "typical_order_size": "500kg-2 tons weekly",
                "payment_terms": "Net 15 days",
                "preferred_crops": ["wheat", "rice", "potato", "onion", "tomato"],
                "quality_requirements": ["A", "B"],
                "max_distance": 50  # km
            },
            {
                "id": "buyer_002", 
                "name": "Hotel Taj Palace",
                "type": "hotel",
                "location": "Chandigarh, Punjab",
                "contact": "+91-98765-43211",
                "email": "purchase@tajpalace.com",
                "rating": 4.9,
                "verified": True,
                "typical_order_size": "200-500kg weekly",
                "payment_terms": "Net 7 days",
                "preferred_crops": ["rice", "wheat", "vegetables", "fruits"],
                "quality_requirements": ["A"],
                "max_distance": 30
            },
            {
                "id": "buyer_003",
                "name": "BigBasket Punjab",
                "type": "retail_chain", 
                "location": "Amritsar, Punjab",
                "contact": "+91-98765-43212",
                "email": "sourcing@bigbasket.com",
                "rating": 4.7,
                "verified": True,
                "typical_order_size": "5-10 tons monthly",
                "payment_terms": "Net 30 days",
                "preferred_crops": ["all"],
                "quality_requirements": ["A", "B"],
                "max_distance": 100
            },
            {
                "id": "buyer_004",
                "name": "Punjab Food Processing Co.",
                "type": "food_processor",
                "location": "Jalandhar, Punjab",
                "contact": "+91-98765-43213",
                "email": "raw.materials@punjabfood.com",
                "rating": 4.6,
                "verified": True,
                "typical_order_size": "10-50 tons monthly",
                "payment_terms": "Net 45 days",
                "preferred_crops": ["wheat", "rice", "maize", "sugarcane"],
                "quality_requirements": ["A", "B", "C"],
                "max_distance": 200
            },
            {
                "id": "buyer_005",
                "name": "Fresh & More Supermarket",
                "type": "retail_chain",
                "location": "Patiala, Punjab",
                "contact": "+91-98765-43214",
                "email": "procurement@freshandmore.com",
                "rating": 4.5,
                "verified": True,
                "typical_order_size": "1-3 tons weekly",
                "payment_terms": "Net 21 days",
                "preferred_crops": ["vegetables", "fruits", "grains"],
                "quality_requirements": ["A", "B"],
                "max_distance": 75
            },
            {
                "id": "buyer_006",
                "name": "Domino's Pizza Supply Chain",
                "type": "restaurant",
                "location": "Ludhiana, Punjab",
                "contact": "+91-98765-43215",
                "email": "supply@dominos.co.in",
                "rating": 4.4,
                "verified": True,
                "typical_order_size": "300-800kg weekly",
                "payment_terms": "Net 10 days",
                "preferred_crops": ["wheat", "tomato", "onion", "capsicum"],
                "quality_requirements": ["A"],
                "max_distance": 40
            },
            {
                "id": "buyer_007",
                "name": "Reliance Fresh",
                "type": "retail_chain",
                "location": "Bathinda, Punjab",
                "contact": "+91-98765-43216",
                "email": "vendor@reliancefresh.com",
                "rating": 4.6,
                "verified": True,
                "typical_order_size": "2-5 tons weekly",
                "payment_terms": "Net 30 days",
                "preferred_crops": ["all"],
                "quality_requirements": ["A", "B"],
                "max_distance": 120
            },
            {
                "id": "buyer_008",
                "name": "McDonald's India Supply",
                "type": "restaurant",
                "location": "Chandigarh, Punjab",
                "contact": "+91-98765-43217",
                "email": "sourcing@mcdonalds.co.in",
                "rating": 4.7,
                "verified": True,
                "typical_order_size": "1-2 tons weekly",
                "payment_terms": "Net 14 days",
                "preferred_crops": ["potato", "onion", "lettuce", "tomato"],
                "quality_requirements": ["A"],
                "max_distance": 60
            }
        ]
    
    def create_listing(self, user_id: str, listing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new surplus listing"""
        listing_id = str(uuid.uuid4())
        
        listing = {
            "id": listing_id,
            "user_id": user_id,
            "crop_type": listing_data["cropType"],
            "quantity": float(listing_data["quantity"]),
            "price_per_unit": float(listing_data["pricePerUnit"]),
            "ready_date": listing_data["readyDate"],
            "quality_grade": listing_data["qualityGrade"],
            "description": listing_data.get("description", ""),
            "status": "active",
            "views": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.listings[listing_id] = listing
        return listing
    
    def get_user_listings(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all listings for a specific user"""
        user_listings = []
        for listing in self.listings.values():
            if listing["user_id"] == user_id:
                # Add mock offers for each listing
                listing_with_offers = listing.copy()
                listing_with_offers["offers"] = self.generate_mock_offers(listing)
                listing_with_offers["views"] = random.randint(5, 50)
                user_listings.append(listing_with_offers)
        
        # Sort by creation date (newest first)
        user_listings.sort(key=lambda x: x["created_at"], reverse=True)
        return user_listings
    
    def update_listing(self, listing_id: str, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a listing"""
        if listing_id not in self.listings:
            return None
        
        listing = self.listings[listing_id]
        if listing["user_id"] != user_id:
            return None  # User can only update their own listings
        
        # Update allowed fields
        allowed_fields = ["crop_type", "quantity", "price_per_unit", "ready_date", "quality_grade", "description", "status"]
        for field in allowed_fields:
            if field in updates:
                listing[field] = updates[field]
        
        listing["updated_at"] = datetime.now().isoformat()
        return listing
    
    def delete_listing(self, listing_id: str, user_id: str) -> bool:
        """Delete a listing"""
        if listing_id not in self.listings:
            return False
        
        listing = self.listings[listing_id]
        if listing["user_id"] != user_id:
            return False  # User can only delete their own listings
        
        del self.listings[listing_id]
        return True
    
    def generate_mock_offers(self, listing: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate realistic mock offers for a listing"""
        crop_type = listing["crop_type"].lower()
        quality_grade = listing["quality_grade"]
        base_price = listing["price_per_unit"]
        quantity = listing["quantity"]
        
        # Filter buyers based on crop preferences and quality requirements
        interested_buyers = []
        for buyer in self.mock_buyers:
            if ("all" in buyer["preferred_crops"] or 
                any(crop in crop_type for crop in buyer["preferred_crops"]) or
                crop_type in buyer["preferred_crops"]):
                if quality_grade in buyer["quality_requirements"]:
                    interested_buyers.append(buyer)
        
        # Generate 0-4 offers randomly
        num_offers = random.randint(0, min(4, len(interested_buyers)))
        if num_offers == 0:
            return []
        
        selected_buyers = random.sample(interested_buyers, num_offers)
        offers = []
        
        for buyer in selected_buyers:
            # Price variation based on buyer type and quality
            price_multiplier = 1.0
            if buyer["type"] == "retail_chain":
                price_multiplier = random.uniform(0.95, 1.05)  # Retail chains negotiate
            elif buyer["type"] == "restaurant":
                price_multiplier = random.uniform(1.02, 1.08)  # Restaurants pay premium for quality
            elif buyer["type"] == "food_processor":
                price_multiplier = random.uniform(0.90, 0.98)  # Processors buy in bulk, lower price
            
            offered_price = round(base_price * price_multiplier, 2)
            
            # Quantity needed (usually less than or equal to available)
            max_quantity = min(quantity, self._get_typical_order_quantity(buyer["typical_order_size"]))
            quantity_needed = random.randint(int(max_quantity * 0.5), int(max_quantity))
            
            # Pickup date (usually within a week of ready date)
            ready_date = datetime.fromisoformat(listing["ready_date"])
            pickup_date = ready_date + timedelta(days=random.randint(0, 7))
            
            offer = {
                "id": f"offer_{uuid.uuid4().hex[:8]}",
                "buyer_name": buyer["name"],
                "buyer_type": buyer["type"],
                "buyer_id": buyer["id"],
                "offered_price": offered_price,
                "quantity_needed": quantity_needed,
                "pickup_date": pickup_date.strftime("%Y-%m-%d"),
                "contact_phone": buyer["contact"],
                "contact_email": buyer["email"],
                "payment_terms": buyer["payment_terms"],
                "buyer_rating": buyer["rating"],
                "verified": buyer["verified"],
                "created_at": datetime.now().isoformat()
            }
            offers.append(offer)
        
        # Sort offers by price (highest first)
        offers.sort(key=lambda x: x["offered_price"], reverse=True)
        return offers
    
    def _get_typical_order_quantity(self, order_size_str: str) -> int:
        """Extract typical quantity from order size string"""
        # Simple parsing of order size strings
        if "kg" in order_size_str:
            numbers = [int(s) for s in order_size_str.split() if s.isdigit()]
            if numbers:
                return max(numbers)  # Take the higher end
        elif "tons" in order_size_str:
            numbers = [int(s) for s in order_size_str.split() if s.isdigit()]
            if numbers:
                return max(numbers) * 1000  # Convert tons to kg
        
        return 500  # Default fallback
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics"""
        total_listings = len(self.listings)
        active_listings = len([l for l in self.listings.values() if l["status"] == "active"])
        total_buyers = len(self.mock_buyers)
        
        # Calculate total value of active listings
        total_value = sum(
            l["quantity"] * l["price_per_unit"] 
            for l in self.listings.values() 
            if l["status"] == "active"
        )
        
        return {
            "total_listings": total_listings,
            "active_listings": active_listings,
            "total_buyers": total_buyers,
            "total_value": round(total_value, 2),
            "avg_price_per_kg": round(total_value / max(sum(l["quantity"] for l in self.listings.values() if l["status"] == "active"), 1), 2)
        }

# Global instance for easy access
marketplace_db = MarketplaceDatabase()