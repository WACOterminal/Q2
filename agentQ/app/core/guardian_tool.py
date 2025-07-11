import structlog
import json
from agentQ.app.core.toolbox import Tool

logger = structlog.get_logger(__name__)

def submit_ethical_review_decision(decision: str, reasoning: str, violated_principle_id: str = None, config: dict = None) -> str:
    """
    Submits the final decision of an ethical review.

    Args:
        decision (str): The final decision, must be either 'APPROVE' or 'VETO'.
        reasoning (str): A concise explanation for the decision.
        violated_principle_id (str, optional): The ID of the principle that was violated, if the decision is a VETO.

    Returns:
        str: A JSON string confirming the submission of the decision.
    """
    logger.info("Submitting ethical review decision", decision=decision, reasoning=reasoning)
    
    if decision.upper() not in ["APPROVE", "VETO"]:
        return "Error: Decision must be either 'APPROVE' or 'VETO'."
        
    if decision.upper() == "VETO" and not violated_principle_id:
        return "Error: A 'VETO' decision must specify the 'violated_principle_id'."

    result = {
        "status": "submitted",
        "decision": decision.upper(),
        "reasoning": reasoning,
        "violated_principle_id": violated_principle_id
    }
    return json.dumps(result)

guardian_tool = Tool(
    name="submit_ethical_review_decision",
    description="Formats and submits the final judgment (APPROVE or VETO) of an ethical review.",
    func=submit_ethical_review_decision
) 