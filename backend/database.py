from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection with error handling
mongo_url = os.environ.get('MONGO_URL')
if not mongo_url:
    raise ValueError("MONGO_URL environment variable is required. Please set it in your deployment environment.")

try:
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ.get('DB_NAME', 'farmchat')]
    print(f"✅ MongoDB connection configured for: {mongo_url.split('@')[1] if '@' in mongo_url else 'localhost'}")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    raise

async def get_database():
    """Get database instance"""
    return db

async def close_database():
    """Close database connection"""
    client.close()