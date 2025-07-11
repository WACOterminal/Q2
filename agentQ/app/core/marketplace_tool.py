import httpx
import structlog
import json
from agentQ.app.core.toolbox import Tool

logger = structlog.get_logger(__name__)

def find_available_gigs(config: dict = None) -> str:
    """
    Finds available gigs on the external marketplace.

    Returns:
        str: A JSON string list of available gigs.
    """
    integrationhub_url = config.get("integrationhub_url")
    if not integrationhub_url:
        return "Error: integrationhub_url is not configured."

    request_url = f"{integrationhub_url}/api/v1/marketplace/gigs"
    logger.info("Finding available gigs", url=request_url)
    
    try:
        with httpx.Client() as client:
            response = client.get(request_url, timeout=30.0)
            response.raise_for_status()
            return response.text
    except Exception as e:
        logger.error("Failed to find available gigs", exc_info=True)
        return f"Error: An unexpected error occurred while finding gigs: {e}"

def bid_on_gig(gig_id: str, agent_squad_id: str, config: dict = None) -> str:
    """
    Bids on a specific gig on behalf of an agent squad.

    Args:
        gig_id (str): The ID of the gig to bid on.
        agent_squad_id (str): The ID of the agent squad that will perform the work.

    Returns:
        str: A JSON string of the bid response.
    """
    integrationhub_url = config.get("integrationhub_url")
    if not integrationhub_url:
        return "Error: integrationhub_url is not configured."

    request_url = f"{integrationhub_url}/api/v1/marketplace/gigs/{gig_id}/bid"
    logger.info("Bidding on gig", url=request_url, gig_id=gig_id, squad_id=agent_squad_id)
    
    try:
        with httpx.Client() as client:
            response = client.post(request_url, json={"agent_squad_id": agent_squad_id}, timeout=30.0)
            response.raise_for_status()
            return response.text
    except Exception as e:
        logger.error("Failed to bid on gig", exc_info=True)
        return f"Error: An unexpected error occurred while bidding on gig: {e}"

# --- Tool Registration ---
find_gigs_tool = Tool(
    name="find_available_gigs",
    description="Scans the external marketplace for available tasks (gigs) that can be completed for a reward.",
    func=find_available_gigs
)

bid_on_gig_tool = Tool(
    name="bid_on_gig",
    description="Places a bid on a specific gig from the marketplace, assigning it to an agent squad.",
    func=bid_on_gig
)

marketplace_tools = [find_gigs_tool, bid_on_gig_tool] 