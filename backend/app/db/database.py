from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import DATABASE_URL, DATABASE_NAME

# Global database client and db instance
client = None
db = None

async def init_db():
    global client, db
    client = AsyncIOMotorClient(DATABASE_URL)
    db = client[DATABASE_NAME]
    
    # Optional: setup indexes
    await db["videos"].create_index("id", unique=True)
    await db["segments"].create_index("id", unique=True)
    await db["segments"].create_index("video_id")
    await db["monetization_rules"].create_index("video_id", unique=True)

async def close_db():
    global client
    if client:
        client.close()

async def get_db():
    global db
    yield db
