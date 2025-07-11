from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import random
import uuid

router = APIRouter()

# --- In-memory mock database ---
GIGS_DB: Dict[str, Dict] = {}

def create_mock_gigs():
    tasks = [
        {"title": "Transcribe Audio from 10-minute Meeting", "reward_usd": 5.00, "skills": ["transcription"]},
        {"title": "Categorize 100 Product Images", "reward_usd": 15.00, "skills": ["image_classification"]},
        {"title": "Write a Python script to scrape a website", "reward_usd": 50.00, "skills": ["python", "web_scraping"]},
        {"title": "Summarize a 5-page research paper", "reward_usd": 25.00, "skills": ["summarization"]},
        {"title": "Perform sentiment analysis on 1,000 tweets", "reward_usd": 30.00, "skills": ["data_analysis", "nlp"]}
    ]
    for task in tasks:
        gig_id = f"gig_{uuid.uuid4().hex[:8]}"
        GIGS_DB[gig_id] = {
            "id": gig_id,
            "title": task["title"],
            "description": f"Complete the task: {task['title']}",
            "reward_usd": task["reward_usd"],
            "required_skills": task["skills"],
            "status": "OPEN" # OPEN, ASSIGNED, COMPLETED
        }
create_mock_gigs()

# --- API Models ---
class Gig(BaseModel):
    id: str
    title: str
    description: str
    reward_usd: float
    required_skills: List[str]
    status: str

class BidRequest(BaseModel):
    agent_squad_id: str

# --- API Endpoints ---
@router.get("/gigs", response_model=List[Gig])
async def list_available_gigs():
    """Returns a list of all gigs with 'OPEN' status."""
    open_gigs = [Gig(**gig) for gig in GIGS_DB.values() if gig["status"] == "OPEN"]
    return open_gigs

@router.post("/gigs/{gig_id}/bid", status_code=200)
async def bid_on_gig(gig_id: str, bid: BidRequest):
    """Allows an agent to bid on and be assigned a gig."""
    if gig_id not in GIGS_DB:
        raise HTTPException(status_code=404, detail="Gig not found.")
    
    gig = GIGS_DB[gig_id]
    if gig["status"] != "OPEN":
        raise HTTPException(status_code=400, detail="Gig is not open for bidding.")
        
    gig["status"] = "ASSIGNED"
    gig["assigned_to"] = bid.agent_squad_id
    
    return {"message": "Bid successful. Gig has been assigned.", "gig_id": gig_id} 