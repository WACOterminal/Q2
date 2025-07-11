import httpx
import structlog
from datetime import datetime, timedelta
from .toolbox import Tool

logger = structlog.get_logger(__name__)

def prometheus_query_range(query: str, duration_minutes: int, config: dict = None) -> str:
    """
    Queries Prometheus for a given PromQL query over a specified time range.

    Args:
        query (str): The PromQL query to execute.
        duration_minutes (int): The duration in the past (in minutes) to query over.
    
    Returns:
        str: The query result from Prometheus, or an error message.
    """
    prometheus_url = config.get("prometheus_url")
    if not prometheus_url:
        return "Error: prometheus_url is not configured."

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=duration_minutes)
    
    params = {
        'query': query,
        'start': start_time.isoformat() + "Z",
        'end': end_time.isoformat() + "Z",
        'step': '60s'  # 1-minute steps
    }

    logger.info("Querying Prometheus", query=query, start=params['start'], end=params['end'])

    try:
        with httpx.Client() as client:
            response = client.get(f"{prometheus_url}/api/v1/query_range", params=params)
            response.raise_for_status()
            return response.text
    except httpx.RequestError as e:
        logger.error("Prometheus query failed", query=query, error=str(e))
        return f"Error: Failed to query Prometheus. Reason: {e}"
    except Exception as e:
        logger.error("An unexpected error occurred during Prometheus query", query=query, exc_info=True)
        return f"Error: An unexpected error occurred: {e}"

prometheus_tool = Tool(
    name="prometheus_query_range",
    description="Executes a PromQL query over a time range (e.g., last 15 minutes) and returns the results.",
    func=prometheus_query_range,
    config={"prometheus_url": "http://prometheus.q-platform.svc.cluster.local:9090"} # Example URL
) 