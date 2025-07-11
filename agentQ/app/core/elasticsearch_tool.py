import httpx
import structlog
from datetime import datetime, timedelta
import json
from .toolbox import Tool

logger = structlog.get_logger(__name__)

def elasticsearch_query(query_string: str, index_pattern: str, duration_minutes: int, config: dict = None) -> str:
    """
    Queries Elasticsearch with a given query string over a specified time range.

    Args:
        query_string (str): The Lucene query string (e.g., 'level:ERROR AND "OOMKilled"').
        index_pattern (str): The index pattern to search against (e.g., 'logs-my-service-*').
        duration_minutes (int): The duration in the past (in minutes) to query over.
    
    Returns:
        str: The search results from Elasticsearch, or an error message.
    """
    elasticsearch_url = config.get("elasticsearch_url")
    if not elasticsearch_url:
        return "Error: elasticsearch_url is not configured."

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=duration_minutes)

    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "query": query_string,
                            "analyze_wildcard": True
                        }
                    },
                    {
                        "range": {
                            "@timestamp": {
                                "gte": start_time.isoformat() + "Z",
                                "lte": end_time.isoformat() + "Z",
                                "format": "strict_date_optional_time"
                            }
                        }
                    }
                ]
            }
        },
        "size": 100,  # Limit the number of results
        "sort": [{"@timestamp": {"order": "desc"}}]
    }

    headers = {'Content-Type': 'application/json'}
    url = f"{elasticsearch_url}/{index_pattern}/_search"
    logger.info("Querying Elasticsearch", url=url, query=query)

    try:
        with httpx.Client() as client:
            response = client.post(url, json=query, headers=headers)
            response.raise_for_status()
            return response.text
    except httpx.RequestError as e:
        logger.error("Elasticsearch query failed", query=query_string, error=str(e))
        return f"Error: Failed to query Elasticsearch. Reason: {e}"
    except Exception as e:
        logger.error("An unexpected error occurred during Elasticsearch query", query=query_string, exc_info=True)
        return f"Error: An unexpected error occurred: {e}"

elasticsearch_tool = Tool(
    name="elasticsearch_query",
    description="Executes a Lucene query string against an Elasticsearch index over a time range.",
    func=elasticsearch_query,
    config={"elasticsearch_url": "http://elasticsearch.q-platform.svc.cluster.local:9200"} # Example URL
) 