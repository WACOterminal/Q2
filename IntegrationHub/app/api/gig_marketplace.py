from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import uuid
import logging
import asyncio
from sqlalchemy import Column, String, Float, DateTime, JSON, Enum as SQLEnum, Boolean, Integer, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, update, and_, or_

from shared.vault_client import VaultClient
from shared.pulsar_client import shared_pulsar_client
from shared.q_auth_parser.parser import UserClaims, get_current_user

logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
vault_client = VaultClient()

# Get database credentials from Vault
db_creds = vault_client.read_secret_data("database/gig_marketplace")
DATABASE_URL = f"postgresql+asyncpg://{db_creds.get('username')}:{db_creds.get('password')}@{db_creds.get('host', 'postgres')}:{db_creds.get('port', 5432)}/{db_creds.get('database', 'gig_marketplace')}"

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# --- Enums ---
class GigStatus(str, Enum):
    DRAFT = "DRAFT"
    OPEN = "OPEN"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    SUBMITTED = "SUBMITTED"
    IN_REVIEW = "IN_REVIEW"
    COMPLETED = "COMPLETED"
    DISPUTED = "DISPUTED"
    CANCELLED = "CANCELLED"

class PaymentStatus(str, Enum):
    PENDING = "PENDING"
    ESCROW = "ESCROW"
    RELEASED = "RELEASED"
    REFUNDED = "REFUNDED"
    DISPUTED = "DISPUTED"

class SkillCategory(str, Enum):
    PROGRAMMING = "PROGRAMMING"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    CONTENT_CREATION = "CONTENT_CREATION"
    DESIGN = "DESIGN"
    TRANSLATION = "TRANSLATION"
    RESEARCH = "RESEARCH"
    TESTING = "TESTING"
    AI_TRAINING = "AI_TRAINING"
    OTHER = "OTHER"

# --- Database Models ---
class GigModel(Base):
    __tablename__ = "gigs"
    
    id = Column(String, primary_key=True, default=lambda: f"gig_{uuid.uuid4().hex[:12]}")
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    requirements = Column(JSON, default=dict)  # Detailed requirements
    
    # Pricing
    reward_usd = Column(Float, nullable=False)
    estimated_hours = Column(Float, default=1.0)
    
    # Skills and categories
    required_skills = Column(JSON, default=list)
    skill_category = Column(SQLEnum(SkillCategory), default=SkillCategory.OTHER)
    difficulty_level = Column(Integer, default=1)  # 1-5 scale
    
    # Status and assignment
    status = Column(SQLEnum(GigStatus), default=GigStatus.DRAFT)
    assigned_to = Column(String, nullable=True)  # Agent or user ID
    assigned_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deadline = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Creator info
    created_by = Column(String, nullable=False)  # User ID
    organization_id = Column(String, nullable=True)
    
    # Quality and ratings
    quality_score = Column(Float, nullable=True)  # 0-5 rating
    reviewer_notes = Column(Text, nullable=True)
    
    # Payment info
    payment_status = Column(SQLEnum(PaymentStatus), default=PaymentStatus.PENDING)
    payment_transaction_id = Column(String, nullable=True)
    
    # Relationships
    bids = relationship("BidModel", back_populates="gig", cascade="all, delete-orphan")
    submissions = relationship("SubmissionModel", back_populates="gig", cascade="all, delete-orphan")

class BidModel(Base):
    __tablename__ = "bids"
    
    id = Column(String, primary_key=True, default=lambda: f"bid_{uuid.uuid4().hex[:12]}")
    gig_id = Column(String, ForeignKey("gigs.id"), nullable=False)
    bidder_id = Column(String, nullable=False)  # Agent or user ID
    
    # Bid details
    proposed_amount = Column(Float, nullable=False)
    estimated_completion_hours = Column(Float, nullable=False)
    cover_letter = Column(Text, nullable=True)
    
    # Bidder qualifications
    relevant_experience = Column(JSON, default=dict)
    skill_match_score = Column(Float, default=0.0)  # 0-1 score
    
    # Status
    is_selected = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    gig = relationship("GigModel", back_populates="bids")

class SubmissionModel(Base):
    __tablename__ = "submissions"
    
    id = Column(String, primary_key=True, default=lambda: f"sub_{uuid.uuid4().hex[:12]}")
    gig_id = Column(String, ForeignKey("gigs.id"), nullable=False)
    submitter_id = Column(String, nullable=False)
    
    # Submission content
    content = Column(JSON, nullable=False)  # Flexible structure for different types
    attachments = Column(JSON, default=list)  # URLs to uploaded files
    notes = Column(Text, nullable=True)
    
    # Review
    is_approved = Column(Boolean, nullable=True)
    review_feedback = Column(Text, nullable=True)
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    
    # Timestamps
    submitted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    gig = relationship("GigModel", back_populates="submissions")

# --- Payment Integration ---
class PaymentProcessor:
    """Handles payment processing for gigs"""
    
    def __init__(self):
        # Get payment provider credentials from Vault
        self.payment_creds = vault_client.read_secret_data("payments/stripe")
        self.stripe_api_key = self.payment_creds.get("api_key")
        
    async def create_escrow(self, gig_id: str, amount: float) -> Dict[str, Any]:
        """Create an escrow payment for a gig"""
        # In production, this would integrate with Stripe or similar
        # For now, simulate escrow creation
        transaction_id = f"txn_{uuid.uuid4().hex[:16]}"
        
        # Log the escrow creation
        await shared_pulsar_client.send_message(
            "persistent://public/default/payment-events",
            {
                "event_type": "ESCROW_CREATED",
                "gig_id": gig_id,
                "amount": amount,
                "transaction_id": transaction_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "transaction_id": transaction_id,
            "status": "ESCROW",
            "amount": amount
        }
    
    async def release_payment(self, transaction_id: str, recipient_id: str) -> bool:
        """Release escrowed payment to recipient"""
        # In production, this would trigger actual payment
        await shared_pulsar_client.send_message(
            "persistent://public/default/payment-events",
            {
                "event_type": "PAYMENT_RELEASED",
                "transaction_id": transaction_id,
                "recipient_id": recipient_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        return True
    
    async def refund_payment(self, transaction_id: str) -> bool:
        """Refund escrowed payment to buyer"""
        # In production, this would trigger actual refund
        await shared_pulsar_client.send_message(
            "persistent://public/default/payment-events",
            {
                "event_type": "PAYMENT_REFUNDED",
                "transaction_id": transaction_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        return True

# --- Skill Matching Service ---
class SkillMatcher:
    """Matches agent/user skills with gig requirements"""
    
    async def calculate_match_score(self, user_skills: List[str], required_skills: List[str]) -> float:
        """Calculate how well user skills match gig requirements"""
        if not required_skills:
            return 1.0
        
        matched_skills = set(user_skills) & set(required_skills)
        return len(matched_skills) / len(required_skills)
    
    async def get_user_skills(self, user_id: str) -> List[str]:
        """Retrieve user/agent skills from profile service"""
        # In production, this would query the user profile service
        # For now, return sample skills
        skill_sets = {
            "developer": ["python", "javascript", "api_development", "testing"],
            "analyst": ["data_analysis", "sql", "visualization", "statistics"],
            "writer": ["content_creation", "copywriting", "research", "editing"],
            "designer": ["ui_design", "graphics", "prototyping", "branding"]
        }
        
        # Assign skills based on user ID hash
        skill_type = list(skill_sets.keys())[hash(user_id) % len(skill_sets)]
        return skill_sets[skill_type]

# Initialize services
payment_processor = PaymentProcessor()
skill_matcher = SkillMatcher()

# --- API Models ---
class CreateGigRequest(BaseModel):
    title: str = Field(..., min_length=10, max_length=200)
    description: str = Field(..., min_length=50)
    requirements: Dict[str, Any] = Field(default_factory=dict)
    reward_usd: float = Field(..., gt=0, le=10000)
    estimated_hours: float = Field(default=1.0, gt=0, le=100)
    required_skills: List[str] = Field(default_factory=list, max_items=10)
    skill_category: SkillCategory = SkillCategory.OTHER
    difficulty_level: int = Field(default=1, ge=1, le=5)
    deadline: Optional[datetime] = None
    
    @validator('deadline')
    def deadline_must_be_future(cls, v):
        if v and v <= datetime.utcnow():
            raise ValueError('Deadline must be in the future')
        return v

class GigResponse(BaseModel):
    id: str
    title: str
    description: str
    requirements: Dict[str, Any]
    reward_usd: float
    estimated_hours: float
    required_skills: List[str]
    skill_category: SkillCategory
    difficulty_level: int
    status: GigStatus
    assigned_to: Optional[str]
    created_at: datetime
    deadline: Optional[datetime]
    created_by: str
    payment_status: PaymentStatus
    bids_count: int = 0
    
    class Config:
        orm_mode = True

class BidRequest(BaseModel):
    proposed_amount: float = Field(..., gt=0)
    estimated_completion_hours: float = Field(..., gt=0, le=100)
    cover_letter: Optional[str] = Field(None, max_length=1000)
    relevant_experience: Dict[str, Any] = Field(default_factory=dict)

class SubmitWorkRequest(BaseModel):
    content: Dict[str, Any]
    attachments: List[str] = Field(default_factory=list, max_items=10)
    notes: Optional[str] = Field(None, max_length=1000)

class ReviewSubmissionRequest(BaseModel):
    is_approved: bool
    feedback: str = Field(..., min_length=10)

# --- Dependency Functions ---
async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

# --- API Router ---
router = APIRouter(prefix="/api/v1/gigs", tags=["gig-marketplace"])

@router.post("/", response_model=GigResponse)
async def create_gig(
    gig: CreateGigRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: UserClaims = Depends(get_current_user)
):
    """Create a new gig listing"""
    db_gig = GigModel(
        **gig.dict(),
        created_by=current_user.sub,
        organization_id=current_user.organization_id
    )
    
    db.add(db_gig)
    await db.commit()
    await db.refresh(db_gig)
    
    # Notify agents about new gig
    background_tasks.add_task(
        notify_agents_about_gig,
        db_gig.id,
        db_gig.required_skills,
        db_gig.reward_usd
    )
    
    return GigResponse.from_orm(db_gig)

@router.get("/", response_model=List[GigResponse])
async def list_gigs(
    status: Optional[GigStatus] = Query(None),
    skill_category: Optional[SkillCategory] = Query(None),
    min_reward: Optional[float] = Query(None, ge=0),
    max_reward: Optional[float] = Query(None, ge=0),
    skills: Optional[List[str]] = Query(None),
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: UserClaims = Depends(get_current_user)
):
    """List available gigs with filtering"""
    query = select(GigModel)
    
    # Apply filters
    filters = []
    if status:
        filters.append(GigModel.status == status)
    else:
        # Default to showing open gigs
        filters.append(GigModel.status == GigStatus.OPEN)
    
    if skill_category:
        filters.append(GigModel.skill_category == skill_category)
    
    if min_reward is not None:
        filters.append(GigModel.reward_usd >= min_reward)
    
    if max_reward is not None:
        filters.append(GigModel.reward_usd <= max_reward)
    
    if skills:
        # Filter gigs that require any of the specified skills
        skill_filters = []
        for skill in skills:
            skill_filters.append(GigModel.required_skills.contains([skill]))
        filters.append(or_(*skill_filters))
    
    if filters:
        query = query.where(and_(*filters))
    
    # Add ordering and pagination
    query = query.order_by(GigModel.created_at.desc()).limit(limit).offset(offset)
    
    result = await db.execute(query)
    gigs = result.scalars().all()
    
    # Add bid counts
    gig_responses = []
    for gig in gigs:
        gig_dict = gig.__dict__
        bid_count_result = await db.execute(
            select(BidModel).where(BidModel.gig_id == gig.id)
        )
        gig_dict['bids_count'] = len(bid_count_result.scalars().all())
        gig_responses.append(GigResponse(**gig_dict))
    
    return gig_responses

@router.get("/recommended", response_model=List[GigResponse])
async def get_recommended_gigs(
    limit: int = Query(10, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: UserClaims = Depends(get_current_user)
):
    """Get gigs recommended based on user skills"""
    # Get user skills
    user_skills = await skill_matcher.get_user_skills(current_user.sub)
    
    # Find open gigs
    result = await db.execute(
        select(GigModel).where(GigModel.status == GigStatus.OPEN)
    )
    open_gigs = result.scalars().all()
    
    # Calculate match scores and sort
    gig_scores = []
    for gig in open_gigs:
        score = await skill_matcher.calculate_match_score(user_skills, gig.required_skills)
        if score > 0:
            gig_scores.append((gig, score))
    
    # Sort by score and return top matches
    gig_scores.sort(key=lambda x: x[1], reverse=True)
    recommended_gigs = [GigResponse.from_orm(gig) for gig, _ in gig_scores[:limit]]
    
    return recommended_gigs

@router.get("/{gig_id}", response_model=GigResponse)
async def get_gig(
    gig_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserClaims = Depends(get_current_user)
):
    """Get details of a specific gig"""
    result = await db.execute(select(GigModel).where(GigModel.id == gig_id))
    gig = result.scalar_one_or_none()
    
    if not gig:
        raise HTTPException(status_code=404, detail="Gig not found")
    
    # Add bid count
    bid_count_result = await db.execute(
        select(BidModel).where(BidModel.gig_id == gig_id)
    )
    gig_dict = gig.__dict__
    gig_dict['bids_count'] = len(bid_count_result.scalars().all())
    
    return GigResponse(**gig_dict)

@router.post("/{gig_id}/bid")
async def bid_on_gig(
    gig_id: str,
    bid: BidRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: UserClaims = Depends(get_current_user)
):
    """Submit a bid for a gig"""
    # Check if gig exists and is open
    result = await db.execute(select(GigModel).where(GigModel.id == gig_id))
    gig = result.scalar_one_or_none()
    
    if not gig:
        raise HTTPException(status_code=404, detail="Gig not found")
    
    if gig.status != GigStatus.OPEN:
        raise HTTPException(status_code=400, detail="Gig is not open for bidding")
    
    # Check if user already bid
    existing_bid_result = await db.execute(
        select(BidModel).where(
            and_(BidModel.gig_id == gig_id, BidModel.bidder_id == current_user.sub)
        )
    )
    if existing_bid_result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="You have already bid on this gig")
    
    # Calculate skill match score
    user_skills = await skill_matcher.get_user_skills(current_user.sub)
    match_score = await skill_matcher.calculate_match_score(user_skills, gig.required_skills)
    
    # Create bid
    db_bid = BidModel(
        gig_id=gig_id,
        bidder_id=current_user.sub,
        skill_match_score=match_score,
        **bid.dict()
    )
    
    db.add(db_bid)
    await db.commit()
    
    # Notify gig creator
    background_tasks.add_task(
        notify_gig_creator_about_bid,
        gig.created_by,
        gig_id,
        current_user.sub,
        bid.proposed_amount
    )
    
    return {"message": "Bid submitted successfully", "bid_id": db_bid.id, "match_score": match_score}

@router.post("/{gig_id}/assign/{bid_id}")
async def assign_gig(
    gig_id: str,
    bid_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: UserClaims = Depends(get_current_user)
):
    """Assign a gig to a bidder"""
    # Verify gig ownership
    result = await db.execute(select(GigModel).where(GigModel.id == gig_id))
    gig = result.scalar_one_or_none()
    
    if not gig:
        raise HTTPException(status_code=404, detail="Gig not found")
    
    if gig.created_by != current_user.sub:
        raise HTTPException(status_code=403, detail="Only gig creator can assign the gig")
    
    if gig.status != GigStatus.OPEN:
        raise HTTPException(status_code=400, detail="Gig is not open for assignment")
    
    # Get the bid
    bid_result = await db.execute(
        select(BidModel).where(
            and_(BidModel.id == bid_id, BidModel.gig_id == gig_id)
        )
    )
    bid = bid_result.scalar_one_or_none()
    
    if not bid:
        raise HTTPException(status_code=404, detail="Bid not found")
    
    # Create escrow payment
    escrow_result = await payment_processor.create_escrow(gig_id, bid.proposed_amount)
    
    # Update gig
    gig.status = GigStatus.ASSIGNED
    gig.assigned_to = bid.bidder_id
    gig.assigned_at = datetime.utcnow()
    gig.payment_status = PaymentStatus.ESCROW
    gig.payment_transaction_id = escrow_result["transaction_id"]
    
    # Mark bid as selected
    bid.is_selected = True
    
    await db.commit()
    
    # Notify the selected bidder
    background_tasks.add_task(
        notify_bidder_about_assignment,
        bid.bidder_id,
        gig_id,
        gig.title
    )
    
    return {
        "message": "Gig assigned successfully",
        "assigned_to": bid.bidder_id,
        "transaction_id": escrow_result["transaction_id"]
    }

@router.post("/{gig_id}/submit")
async def submit_work(
    gig_id: str,
    submission: SubmitWorkRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserClaims = Depends(get_current_user)
):
    """Submit completed work for a gig"""
    # Verify assignment
    result = await db.execute(select(GigModel).where(GigModel.id == gig_id))
    gig = result.scalar_one_or_none()
    
    if not gig:
        raise HTTPException(status_code=404, detail="Gig not found")
    
    if gig.assigned_to != current_user.sub:
        raise HTTPException(status_code=403, detail="You are not assigned to this gig")
    
    if gig.status not in [GigStatus.ASSIGNED, GigStatus.IN_PROGRESS]:
        raise HTTPException(status_code=400, detail="Cannot submit work for this gig")
    
    # Create submission
    db_submission = SubmissionModel(
        gig_id=gig_id,
        submitter_id=current_user.sub,
        **submission.dict()
    )
    
    # Update gig status
    gig.status = GigStatus.SUBMITTED
    
    db.add(db_submission)
    await db.commit()
    
    # Notify gig creator
    await shared_pulsar_client.send_message(
        "persistent://public/default/gig-notifications",
        {
            "type": "WORK_SUBMITTED",
            "recipient": gig.created_by,
            "gig_id": gig_id,
            "submission_id": db_submission.id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return {"message": "Work submitted successfully", "submission_id": db_submission.id}

@router.post("/{gig_id}/review/{submission_id}")
async def review_submission(
    gig_id: str,
    submission_id: str,
    review: ReviewSubmissionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: UserClaims = Depends(get_current_user)
):
    """Review submitted work"""
    # Verify gig ownership
    result = await db.execute(select(GigModel).where(GigModel.id == gig_id))
    gig = result.scalar_one_or_none()
    
    if not gig:
        raise HTTPException(status_code=404, detail="Gig not found")
    
    if gig.created_by != current_user.sub:
        raise HTTPException(status_code=403, detail="Only gig creator can review submissions")
    
    # Get submission
    submission_result = await db.execute(
        select(SubmissionModel).where(
            and_(SubmissionModel.id == submission_id, SubmissionModel.gig_id == gig_id)
        )
    )
    submission = submission_result.scalar_one_or_none()
    
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    # Update submission
    submission.is_approved = review.is_approved
    submission.review_feedback = review.feedback
    submission.reviewed_by = current_user.sub
    submission.reviewed_at = datetime.utcnow()
    
    if review.is_approved:
        # Mark gig as completed
        gig.status = GigStatus.COMPLETED
        gig.completed_at = datetime.utcnow()
        
        # Release payment
        if gig.payment_transaction_id:
            background_tasks.add_task(
                payment_processor.release_payment,
                gig.payment_transaction_id,
                gig.assigned_to
            )
            gig.payment_status = PaymentStatus.RELEASED
    else:
        # Return to in progress for revision
        gig.status = GigStatus.IN_PROGRESS
    
    await db.commit()
    
    # Notify submitter
    await shared_pulsar_client.send_message(
        "persistent://public/default/gig-notifications",
        {
            "type": "SUBMISSION_REVIEWED",
            "recipient": submission.submitter_id,
            "gig_id": gig_id,
            "is_approved": review.is_approved,
            "feedback": review.feedback,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return {
        "message": f"Submission {'approved' if review.is_approved else 'rejected'}",
        "payment_status": gig.payment_status if review.is_approved else None
    }

# --- Background Tasks ---
async def notify_agents_about_gig(gig_id: str, required_skills: List[str], reward: float):
    """Notify relevant agents about new gig opportunity"""
    await shared_pulsar_client.send_message(
        "persistent://public/default/agent-opportunities",
        {
            "type": "NEW_GIG",
            "gig_id": gig_id,
            "required_skills": required_skills,
            "reward_usd": reward,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def notify_gig_creator_about_bid(creator_id: str, gig_id: str, bidder_id: str, amount: float):
    """Notify gig creator about new bid"""
    await shared_pulsar_client.send_message(
        "persistent://public/default/user-notifications",
        {
            "type": "NEW_BID",
            "recipient": creator_id,
            "gig_id": gig_id,
            "bidder_id": bidder_id,
            "amount": amount,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def notify_bidder_about_assignment(bidder_id: str, gig_id: str, gig_title: str):
    """Notify bidder about gig assignment"""
    await shared_pulsar_client.send_message(
        "persistent://public/default/user-notifications",
        {
            "type": "GIG_ASSIGNED",
            "recipient": bidder_id,
            "gig_id": gig_id,
            "gig_title": gig_title,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# --- Database initialization ---
async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Run initialization on module import
asyncio.create_task(init_db()) 