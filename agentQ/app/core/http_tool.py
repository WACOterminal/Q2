import httpx
import structlog
from .toolbox import Tool

logger = structlog.get_logger(__name__)

def http_get(url: str, config: dict = None) -> str:
    """
    Performs an HTTP GET request to the given URL and returns the response body.
    
    Args:
        url (str): The URL to fetch.
    
    Returns:
        str: The content of the response, or an error message.
    """
    logger.info("Performing HTTP GET", url=url)
    try:
        with httpx.Client(follow_redirects=True) as client:
            response = client.get(url, timeout=30.0)
            response.raise_for_status()
            # We assume the content is text-based (HTML, JSON, etc.)
            return response.text
    except httpx.RequestError as e:
        logger.error("HTTP GET request failed", url=url, error=str(e))
        return f"Error: Failed to fetch URL {url}. Reason: {e}"
    except Exception as e:
        logger.error("An unexpected error occurred during HTTP GET", url=url, exc_info=True)
        return f"Error: An unexpected error occurred: {e}"

http_get_tool = Tool(
    name="http_get",
    description="Makes an HTTP GET request to a URL and returns the raw content of the response.",
    func=http_get
) 