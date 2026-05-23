import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import DESCENDING, MongoClient

load_dotenv()

# MongoDB connection
_client = MongoClient(os.getenv("MONGO_URI"))
_db = _client[os.getenv("MONGO_DB", "chatbot_db")]

facts_col = _db["user_facts"]
conv_col = _db["conversations"]

# Compound indexes for efficient lookups
facts_col.create_index([("user_id", 1), ("created_at", DESCENDING)])
conv_col.create_index([("thread_id", 1), ("created_at", 1)])


# Facts(long-term user memory)
def save_fact(user_id: str, fact: str) -> None:
    """Persist a single user fact to MongoDB."""
    facts_col.insert_one({
        "user_id":user_id,
        "fact":fact,
        "created_at":datetime.now(timezone.utc),
    })

def get_facts(user_id: str, limit: int = 20) -> list[str]:
    """Return the most recent `limit` facts about a user."""
    docs = facts_col.find(
        {"user_id": user_id},
        sort=[("created_at", DESCENDING)],
        limit=limit,
    )
    return [d["fact"] for d in docs]


# Conversation history 
def save_message(thread_id: str, role: str, content: str) -> None:
    """Append a single message to a conversation thread."""
    conv_col.insert_one({
        "thread_id":thread_id,
        "role":role,
        "content":content,
        "created_at":datetime.now(timezone.utc)
    })


def get_conversation(thread_id: str) -> list[dict]:
    """Return the full ordered history for a thread."""
    return list(
        conv_col.find(
            {"thread_id": thread_id},
            sort=[("created_at", 1)],
            projection={"_id": 0, "role": 1, "content": 1}
        )
    )