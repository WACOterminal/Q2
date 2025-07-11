"""
Human-Agent Teaming API for Q Platform

This API provides endpoints for sophisticated human-agent collaboration:
- Team formation and management
- Task delegation and coordination
- Real-time collaboration interfaces
- Performance tracking and optimization
- Trust and transparency mechanisms
- Adaptive interaction modes
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
import logging
import json
import uuid
from enum import Enum

# Internal imports
from shared.pulsar_client import shared_pulsar_client
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from agentQ.multi_agent_coordinator import multi_agent_coordinator
from agentQ.adaptive_persona_service import adaptive_persona_service

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

class TeamRole(str, Enum):
    HUMAN_LEAD = "human_lead"
    AI_LEAD = "ai_lead"
    COLLABORATIVE = "collaborative"
    HUMAN_SUPERVISOR = "human_supervisor"
    AI_ASSISTANT = "ai_assistant"

class InteractionMode(str, Enum):
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    HYBRID = "hybrid"
    AUTONOMOUS = "autonomous"

class TrustLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

class CollaborationStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

# Request/Response Models

class TeamFormationRequest(BaseModel):
    team_name: str = Field(..., description="Name of the team")
    objective: str = Field(..., description="Team objective")
    human_members: List[str] = Field(..., description="List of human member IDs")
    ai_agents: List[str] = Field(..., description="List of AI agent IDs")
    team_role: TeamRole = Field(..., description="Team role structure")
    interaction_mode: InteractionMode = Field(..., description="Interaction mode")
    expected_duration: int = Field(..., description="Expected duration in seconds")
    trust_requirements: Dict[str, TrustLevel] = Field(..., description="Trust level requirements")
    skill_requirements: Dict[str, float] = Field(..., description="Required skills and levels")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class TaskDelegationRequest(BaseModel):
    team_id: str = Field(..., description="Team ID")
    task_name: str = Field(..., description="Task name")
    task_description: str = Field(..., description="Task description")
    task_complexity: TaskComplexity = Field(..., description="Task complexity")
    assigned_to: str = Field(..., description="Assigned member ID")
    human_oversight: bool = Field(False, description="Requires human oversight")
    ai_assistance: bool = Field(False, description="Requires AI assistance")
    deadline: Optional[datetime] = Field(None, description="Task deadline")
    dependencies: List[str] = Field([], description="Task dependencies")
    success_criteria: Dict[str, Any] = Field(..., description="Success criteria")

class CollaborationMessage(BaseModel):
    team_id: str = Field(..., description="Team ID")
    sender_id: str = Field(..., description="Sender ID")
    sender_type: str = Field(..., description="Sender type (human/ai)")
    message_type: str = Field(..., description="Message type")
    content: Dict[str, Any] = Field(..., description="Message content")
    recipients: List[str] = Field([], description="Specific recipients")
    priority: str = Field("normal", description="Message priority")
    requires_response: bool = Field(False, description="Requires response")

class FeedbackRequest(BaseModel):
    team_id: str = Field(..., description="Team ID")
    target_id: str = Field(..., description="Target member ID")
    feedback_type: str = Field(..., description="Feedback type")
    rating: float = Field(..., description="Rating score")
    comments: str = Field(..., description="Feedback comments")
    suggestions: List[str] = Field([], description="Improvement suggestions")

class TeamConfigUpdate(BaseModel):
    interaction_mode: Optional[InteractionMode] = Field(None, description="New interaction mode")
    trust_levels: Optional[Dict[str, TrustLevel]] = Field(None, description="Updated trust levels")
    skill_requirements: Optional[Dict[str, float]] = Field(None, description="Updated skill requirements")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

# Response Models

class TeamInfo(BaseModel):
    team_id: str
    team_name: str
    objective: str
    human_members: List[Dict[str, Any]]
    ai_agents: List[Dict[str, Any]]
    team_role: TeamRole
    interaction_mode: InteractionMode
    status: CollaborationStatus
    performance_metrics: Dict[str, float]
    trust_metrics: Dict[str, float]
    created_at: datetime
    last_active: datetime

class TaskStatus(BaseModel):
    task_id: str
    task_name: str
    assigned_to: str
    status: str
    progress: float
    estimated_completion: Optional[datetime]
    human_oversight_required: bool
    ai_assistance_available: bool
    blockers: List[str]
    recent_updates: List[Dict[str, Any]]

class CollaborationMetrics(BaseModel):
    team_id: str
    performance_score: float
    efficiency_rating: float
    communication_quality: float
    trust_level: float
    human_satisfaction: float
    ai_performance: float
    task_completion_rate: float
    collaboration_frequency: float
    measurement_period: Dict[str, datetime]

# Global state management
active_teams: Dict[str, Dict[str, Any]] = {}
team_tasks: Dict[str, List[Dict[str, Any]]] = {}
collaboration_history: Dict[str, List[Dict[str, Any]]] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}

# ===== TEAM FORMATION ENDPOINTS =====

@router.post("/teams/create", response_model=Dict[str, Any])
async def create_team(
    request: TeamFormationRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Create a new human-agent team"""
    
    try:
        team_id = f"team_{uuid.uuid4().hex[:12]}"
        
        # Validate human members and AI agents
        human_members_info = []
        for human_id in request.human_members:
            # In a real system, this would validate against user database
            human_members_info.append({
                "id": human_id,
                "type": "human",
                "status": "active",
                "skills": {},  # Would be populated from user profile
                "trust_level": request.trust_requirements.get(human_id, TrustLevel.MEDIUM)
            })
        
        ai_agents_info = []
        for agent_id in request.ai_agents:
            # Get agent profile from multi-agent coordinator
            agent_profile = await multi_agent_coordinator.get_agent_profile(agent_id)
            if agent_profile:
                ai_agents_info.append({
                    "id": agent_id,
                    "type": "ai",
                    "status": agent_profile.current_state.value,
                    "skills": agent_profile.capabilities,
                    "trust_level": request.trust_requirements.get(agent_id, TrustLevel.MEDIUM),
                    "performance_metrics": agent_profile.performance_metrics
                })
        
        # Create team structure
        team = {
            "team_id": team_id,
            "team_name": request.team_name,
            "objective": request.objective,
            "human_members": human_members_info,
            "ai_agents": ai_agents_info,
            "team_role": request.team_role,
            "interaction_mode": request.interaction_mode,
            "expected_duration": request.expected_duration,
            "trust_requirements": request.trust_requirements,
            "skill_requirements": request.skill_requirements,
            "status": CollaborationStatus.ACTIVE,
            "performance_metrics": {
                "tasks_completed": 0,
                "success_rate": 0.0,
                "efficiency_score": 0.0,
                "communication_quality": 0.0
            },
            "trust_metrics": {
                "overall_trust": 0.5,
                "human_to_ai_trust": 0.5,
                "ai_to_human_trust": 0.5,
                "inter_human_trust": 0.5
            },
            "created_at": datetime.utcnow(),
            "last_active": datetime.utcnow(),
            "metadata": request.metadata or {}
        }
        
        # Store team
        active_teams[team_id] = team
        team_tasks[team_id] = []
        collaboration_history[team_id] = []
        
        # Initialize AI agent personas for team context
        for agent_info in ai_agents_info:
            agent_id = agent_info["id"]
            
            # Analyze team context for persona selection
            context_id = await adaptive_persona_service.analyze_context(
                agent_id=agent_id,
                context_type="team_collaboration",
                features={
                    "team_size": len(request.human_members) + len(request.ai_agents),
                    "team_role": request.team_role.value,
                    "interaction_mode": request.interaction_mode.value,
                    "task_complexity": "moderate",  # Default
                    "human_oversight": True,
                    "collaboration_intensity": 0.7
                }
            )
            
            # Select appropriate persona
            persona_instance_id = await adaptive_persona_service.select_persona(
                agent_id=agent_id,
                context_id=context_id,
                selection_criteria={
                    "persona_type": "collaborative" if request.team_role == TeamRole.COLLABORATIVE else "assistant"
                }
            )
        
        # Publish team formation event
        await shared_pulsar_client.publish(
            "q.human_agent_teaming.team.created",
            {
                "team_id": team_id,
                "team_name": request.team_name,
                "human_members": request.human_members,
                "ai_agents": request.ai_agents,
                "team_role": request.team_role.value,
                "interaction_mode": request.interaction_mode.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        background_tasks.add_task(initialize_team_coordination, team_id)
        
        return {
            "team_id": team_id,
            "status": "created",
            "message": f"Team '{request.team_name}' created successfully",
            "team_info": team
        }
        
    except Exception as e:
        logger.error(f"Error creating team: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/teams/{team_id}", response_model=TeamInfo)
async def get_team(team_id: str, token: str = Depends(security)):
    """Get team information"""
    
    if team_id not in active_teams:
        raise HTTPException(status_code=404, detail="Team not found")
    
    team = active_teams[team_id]
    
    return TeamInfo(
        team_id=team["team_id"],
        team_name=team["team_name"],
        objective=team["objective"],
        human_members=team["human_members"],
        ai_agents=team["ai_agents"],
        team_role=team["team_role"],
        interaction_mode=team["interaction_mode"],
        status=team["status"],
        performance_metrics=team["performance_metrics"],
        trust_metrics=team["trust_metrics"],
        created_at=team["created_at"],
        last_active=team["last_active"]
    )

@router.get("/teams", response_model=List[TeamInfo])
async def list_teams(
    status: Optional[CollaborationStatus] = None,
    human_member: Optional[str] = None,
    ai_agent: Optional[str] = None,
    limit: int = 20,
    token: str = Depends(security)
):
    """List teams with optional filtering"""
    
    teams = []
    
    for team in active_teams.values():
        # Apply filters
        if status and team["status"] != status:
            continue
        
        if human_member and human_member not in [h["id"] for h in team["human_members"]]:
            continue
        
        if ai_agent and ai_agent not in [a["id"] for a in team["ai_agents"]]:
            continue
        
        teams.append(TeamInfo(
            team_id=team["team_id"],
            team_name=team["team_name"],
            objective=team["objective"],
            human_members=team["human_members"],
            ai_agents=team["ai_agents"],
            team_role=team["team_role"],
            interaction_mode=team["interaction_mode"],
            status=team["status"],
            performance_metrics=team["performance_metrics"],
            trust_metrics=team["trust_metrics"],
            created_at=team["created_at"],
            last_active=team["last_active"]
        ))
        
        if len(teams) >= limit:
            break
    
    return teams

@router.put("/teams/{team_id}/config", response_model=Dict[str, Any])
async def update_team_config(
    team_id: str,
    request: TeamConfigUpdate,
    token: str = Depends(security)
):
    """Update team configuration"""
    
    if team_id not in active_teams:
        raise HTTPException(status_code=404, detail="Team not found")
    
    team = active_teams[team_id]
    
    # Update configuration
    if request.interaction_mode:
        team["interaction_mode"] = request.interaction_mode
    
    if request.trust_levels:
        team["trust_requirements"].update(request.trust_levels)
    
    if request.skill_requirements:
        team["skill_requirements"].update(request.skill_requirements)
    
    if request.metadata:
        team["metadata"].update(request.metadata)
    
    team["last_active"] = datetime.utcnow()
    
    # Publish configuration update
    await shared_pulsar_client.publish(
        "q.human_agent_teaming.team.config_updated",
        {
            "team_id": team_id,
            "updates": request.dict(exclude_none=True),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return {"status": "updated", "team_id": team_id}

# ===== TASK DELEGATION ENDPOINTS =====

@router.post("/teams/{team_id}/tasks/delegate", response_model=Dict[str, Any])
async def delegate_task(
    team_id: str,
    request: TaskDelegationRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Delegate a task to a team member"""
    
    if team_id not in active_teams:
        raise HTTPException(status_code=404, detail="Team not found")
    
    team = active_teams[team_id]
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    
    # Validate assignee
    all_members = [h["id"] for h in team["human_members"]] + [a["id"] for a in team["ai_agents"]]
    if request.assigned_to not in all_members:
        raise HTTPException(status_code=400, detail="Invalid assignee")
    
    # Create task
    task = {
        "task_id": task_id,
        "task_name": request.task_name,
        "task_description": request.task_description,
        "task_complexity": request.task_complexity,
        "assigned_to": request.assigned_to,
        "human_oversight": request.human_oversight,
        "ai_assistance": request.ai_assistance,
        "deadline": request.deadline,
        "dependencies": request.dependencies,
        "success_criteria": request.success_criteria,
        "status": "assigned",
        "progress": 0.0,
        "created_at": datetime.utcnow(),
        "last_updated": datetime.utcnow(),
        "updates": [],
        "blockers": []
    }
    
    # Store task
    team_tasks[team_id].append(task)
    
    # If assigned to AI agent, create coordination task
    if request.assigned_to in [a["id"] for a in team["ai_agents"]]:
        await multi_agent_coordinator.create_task(
            task_name=request.task_name,
            task_type="human_agent_collaboration",
            description=request.task_description,
            required_capabilities={"collaboration": 0.7, "task_execution": 0.8},
            created_by=team_id
        )
    
    # Publish task delegation event
    await shared_pulsar_client.publish(
        "q.human_agent_teaming.task.delegated",
        {
            "team_id": team_id,
            "task_id": task_id,
            "task_name": request.task_name,
            "assigned_to": request.assigned_to,
            "complexity": request.task_complexity.value,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    background_tasks.add_task(monitor_task_progress, team_id, task_id)
    
    return {
        "task_id": task_id,
        "status": "delegated",
        "message": f"Task '{request.task_name}' delegated to {request.assigned_to}",
        "task_info": task
    }

@router.get("/teams/{team_id}/tasks", response_model=List[TaskStatus])
async def get_team_tasks(
    team_id: str,
    status: Optional[str] = None,
    assignee: Optional[str] = None,
    token: str = Depends(security)
):
    """Get tasks for a team"""
    
    if team_id not in team_tasks:
        raise HTTPException(status_code=404, detail="Team not found")
    
    tasks = team_tasks[team_id]
    
    # Apply filters
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    
    if assignee:
        tasks = [t for t in tasks if t["assigned_to"] == assignee]
    
    return [
        TaskStatus(
            task_id=task["task_id"],
            task_name=task["task_name"],
            assigned_to=task["assigned_to"],
            status=task["status"],
            progress=task["progress"],
            estimated_completion=task.get("estimated_completion"),
            human_oversight_required=task["human_oversight"],
            ai_assistance_available=task["ai_assistance"],
            blockers=task["blockers"],
            recent_updates=task["updates"][-5:]  # Last 5 updates
        )
        for task in tasks
    ]

@router.put("/teams/{team_id}/tasks/{task_id}/update", response_model=Dict[str, Any])
async def update_task_progress(
    team_id: str,
    task_id: str,
    progress: float = Field(..., description="Progress percentage"),
    status: Optional[str] = None,
    notes: Optional[str] = None,
    blockers: Optional[List[str]] = None,
    token: str = Depends(security)
):
    """Update task progress"""
    
    if team_id not in team_tasks:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # Find task
    task = None
    for t in team_tasks[team_id]:
        if t["task_id"] == task_id:
            task = t
            break
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update task
    task["progress"] = progress
    if status:
        task["status"] = status
    if blockers:
        task["blockers"] = blockers
    
    task["last_updated"] = datetime.utcnow()
    
    # Add update log
    update_log = {
        "timestamp": datetime.utcnow().isoformat(),
        "progress": progress,
        "status": status,
        "notes": notes,
        "blockers": blockers
    }
    task["updates"].append(update_log)
    
    # Update team performance metrics
    team = active_teams[team_id]
    team["last_active"] = datetime.utcnow()
    
    # Publish task update
    await shared_pulsar_client.publish(
        "q.human_agent_teaming.task.updated",
        {
            "team_id": team_id,
            "task_id": task_id,
            "progress": progress,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return {"status": "updated", "task_id": task_id, "progress": progress}

# ===== COMMUNICATION ENDPOINTS =====

@router.post("/teams/{team_id}/messages/send", response_model=Dict[str, Any])
async def send_message(
    team_id: str,
    message: CollaborationMessage,
    token: str = Depends(security)
):
    """Send a message to team members"""
    
    if team_id not in active_teams:
        raise HTTPException(status_code=404, detail="Team not found")
    
    message_id = f"msg_{uuid.uuid4().hex[:12]}"
    
    # Create message record
    message_record = {
        "message_id": message_id,
        "team_id": team_id,
        "sender_id": message.sender_id,
        "sender_type": message.sender_type,
        "message_type": message.message_type,
        "content": message.content,
        "recipients": message.recipients,
        "priority": message.priority,
        "requires_response": message.requires_response,
        "timestamp": datetime.utcnow(),
        "responses": []
    }
    
    # Store message
    collaboration_history[team_id].append(message_record)
    
    # Send to WebSocket connections
    await broadcast_to_team(team_id, message_record)
    
    # If message is for AI agent, process it
    if message.sender_type == "human":
        for recipient in message.recipients:
            team = active_teams[team_id]
            ai_agents = [a["id"] for a in team["ai_agents"]]
            if recipient in ai_agents:
                await process_ai_message(team_id, recipient, message_record)
    
    # Publish message event
    await shared_pulsar_client.publish(
        "q.human_agent_teaming.message.sent",
        {
            "team_id": team_id,
            "message_id": message_id,
            "sender_id": message.sender_id,
            "sender_type": message.sender_type,
            "message_type": message.message_type,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return {"message_id": message_id, "status": "sent"}

@router.get("/teams/{team_id}/messages", response_model=List[Dict[str, Any]])
async def get_team_messages(
    team_id: str,
    limit: int = 50,
    since: Optional[datetime] = None,
    token: str = Depends(security)
):
    """Get team messages"""
    
    if team_id not in collaboration_history:
        raise HTTPException(status_code=404, detail="Team not found")
    
    messages = collaboration_history[team_id]
    
    # Apply time filter
    if since:
        messages = [m for m in messages if m["timestamp"] >= since]
    
    # Sort by timestamp (newest first) and limit
    messages.sort(key=lambda x: x["timestamp"], reverse=True)
    return messages[:limit]

@router.websocket("/teams/{team_id}/collaborate")
async def collaborate_websocket(websocket: WebSocket, team_id: str):
    """WebSocket endpoint for real-time collaboration"""
    
    await websocket.accept()
    
    # Add to connections
    if team_id not in websocket_connections:
        websocket_connections[team_id] = []
    websocket_connections[team_id].append(websocket)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message
            await handle_websocket_message(team_id, websocket, message_data)
            
    except WebSocketDisconnect:
        # Remove from connections
        if team_id in websocket_connections:
            websocket_connections[team_id].remove(websocket)

# ===== FEEDBACK AND TRUST ENDPOINTS =====

@router.post("/teams/{team_id}/feedback", response_model=Dict[str, Any])
async def submit_feedback(
    team_id: str,
    feedback: FeedbackRequest,
    token: str = Depends(security)
):
    """Submit feedback for a team member"""
    
    if team_id not in active_teams:
        raise HTTPException(status_code=404, detail="Team not found")
    
    team = active_teams[team_id]
    
    # Store feedback
    feedback_record = {
        "feedback_id": f"fb_{uuid.uuid4().hex[:12]}",
        "team_id": team_id,
        "target_id": feedback.target_id,
        "feedback_type": feedback.feedback_type,
        "rating": feedback.rating,
        "comments": feedback.comments,
        "suggestions": feedback.suggestions,
        "timestamp": datetime.utcnow()
    }
    
    # Update trust metrics
    await update_trust_metrics(team_id, feedback.target_id, feedback.rating)
    
    # If feedback is for AI agent, record for persona adaptation
    ai_agents = [a["id"] for a in team["ai_agents"]]
    if feedback.target_id in ai_agents:
        await adaptive_persona_service.record_interaction(
            agent_id=feedback.target_id,
            interaction_data={
                "type": "human_feedback",
                "success": feedback.rating > 3.0,
                "user_satisfaction": feedback.rating / 5.0,
                "response_time": 0.0,
                "context": {"team_id": team_id, "feedback_type": feedback.feedback_type},
                "metadata": {"comments": feedback.comments, "suggestions": feedback.suggestions}
            }
        )
    
    # Publish feedback event
    await shared_pulsar_client.publish(
        "q.human_agent_teaming.feedback.submitted",
        {
            "team_id": team_id,
            "target_id": feedback.target_id,
            "feedback_type": feedback.feedback_type,
            "rating": feedback.rating,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return {"status": "submitted", "feedback_id": feedback_record["feedback_id"]}

@router.get("/teams/{team_id}/metrics", response_model=CollaborationMetrics)
async def get_collaboration_metrics(
    team_id: str,
    period_hours: int = 24,
    token: str = Depends(security)
):
    """Get collaboration metrics for a team"""
    
    if team_id not in active_teams:
        raise HTTPException(status_code=404, detail="Team not found")
    
    team = active_teams[team_id]
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=period_hours)
    
    # Calculate metrics
    metrics = await calculate_collaboration_metrics(team_id, start_time, end_time)
    
    return CollaborationMetrics(
        team_id=team_id,
        performance_score=metrics["performance_score"],
        efficiency_rating=metrics["efficiency_rating"],
        communication_quality=metrics["communication_quality"],
        trust_level=metrics["trust_level"],
        human_satisfaction=metrics["human_satisfaction"],
        ai_performance=metrics["ai_performance"],
        task_completion_rate=metrics["task_completion_rate"],
        collaboration_frequency=metrics["collaboration_frequency"],
        measurement_period={"start": start_time, "end": end_time}
    )

# ===== HELPER FUNCTIONS =====

async def initialize_team_coordination(team_id: str):
    """Initialize team coordination processes"""
    
    try:
        team = active_teams[team_id]
        
        # Set up agent coordination
        for agent_info in team["ai_agents"]:
            agent_id = agent_info["id"]
            
            # Update agent state to active
            await multi_agent_coordinator.update_agent_state(agent_id, "active")
        
        # Initialize trust metrics
        await initialize_trust_metrics(team_id)
        
        logger.info(f"Initialized coordination for team {team_id}")
        
    except Exception as e:
        logger.error(f"Error initializing team coordination: {e}")

async def monitor_task_progress(team_id: str, task_id: str):
    """Monitor task progress and provide assistance"""
    
    try:
        # This would implement task monitoring logic
        # For now, just log the monitoring start
        logger.info(f"Started monitoring task {task_id} for team {team_id}")
        
    except Exception as e:
        logger.error(f"Error monitoring task progress: {e}")

async def broadcast_to_team(team_id: str, message: Dict[str, Any]):
    """Broadcast message to all team WebSocket connections"""
    
    if team_id not in websocket_connections:
        return
    
    for websocket in websocket_connections[team_id]:
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")

async def process_ai_message(team_id: str, agent_id: str, message: Dict[str, Any]):
    """Process message directed to AI agent"""
    
    try:
        # This would implement AI message processing
        # For now, just log the message
        logger.info(f"Processing message for AI agent {agent_id} in team {team_id}")
        
    except Exception as e:
        logger.error(f"Error processing AI message: {e}")

async def handle_websocket_message(team_id: str, websocket: WebSocket, message_data: Dict[str, Any]):
    """Handle WebSocket message"""
    
    try:
        message_type = message_data.get("type")
        
        if message_type == "chat":
            # Broadcast chat message
            await broadcast_to_team(team_id, message_data)
        elif message_type == "task_update":
            # Handle task update
            pass
        elif message_type == "status_update":
            # Handle status update
            pass
        
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")

async def update_trust_metrics(team_id: str, target_id: str, rating: float):
    """Update trust metrics based on feedback"""
    
    try:
        team = active_teams[team_id]
        
        # Simple trust metric update
        current_trust = team["trust_metrics"]["overall_trust"]
        new_trust = (current_trust + rating / 5.0) / 2  # Simple averaging
        team["trust_metrics"]["overall_trust"] = new_trust
        
        # Update specific trust metrics
        ai_agents = [a["id"] for a in team["ai_agents"]]
        if target_id in ai_agents:
            team["trust_metrics"]["human_to_ai_trust"] = new_trust
        else:
            team["trust_metrics"]["inter_human_trust"] = new_trust
        
    except Exception as e:
        logger.error(f"Error updating trust metrics: {e}")

async def initialize_trust_metrics(team_id: str):
    """Initialize trust metrics for a team"""
    
    try:
        team = active_teams[team_id]
        
        # Initialize trust metrics based on team composition
        team["trust_metrics"] = {
            "overall_trust": 0.5,
            "human_to_ai_trust": 0.4,  # Start lower for AI
            "ai_to_human_trust": 0.6,  # AI starts with higher trust
            "inter_human_trust": 0.7   # Humans typically trust each other more initially
        }
        
    except Exception as e:
        logger.error(f"Error initializing trust metrics: {e}")

async def calculate_collaboration_metrics(team_id: str, start_time: datetime, end_time: datetime) -> Dict[str, float]:
    """Calculate collaboration metrics for a team"""
    
    try:
        team = active_teams[team_id]
        tasks = team_tasks.get(team_id, [])
        messages = collaboration_history.get(team_id, [])
        
        # Filter by time period
        period_tasks = [t for t in tasks if start_time <= t["created_at"] <= end_time]
        period_messages = [m for m in messages if start_time <= m["timestamp"] <= end_time]
        
        # Calculate metrics
        metrics = {
            "performance_score": team["performance_metrics"]["success_rate"],
            "efficiency_rating": team["performance_metrics"]["efficiency_score"],
            "communication_quality": len(period_messages) / max(1, (end_time - start_time).total_seconds() / 3600),
            "trust_level": team["trust_metrics"]["overall_trust"],
            "human_satisfaction": 0.7,  # Would be calculated from feedback
            "ai_performance": 0.8,  # Would be calculated from AI agent metrics
            "task_completion_rate": len([t for t in period_tasks if t["status"] == "completed"]) / max(1, len(period_tasks)),
            "collaboration_frequency": len(period_messages) / max(1, len(team["human_members"]) + len(team["ai_agents"]))
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating collaboration metrics: {e}")
        return {
            "performance_score": 0.0,
            "efficiency_rating": 0.0,
            "communication_quality": 0.0,
            "trust_level": 0.0,
            "human_satisfaction": 0.0,
            "ai_performance": 0.0,
            "task_completion_rate": 0.0,
            "collaboration_frequency": 0.0
        } 