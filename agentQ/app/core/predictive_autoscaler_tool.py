import structlog
from pyignite import Client
import json
from agentQ.app.core.toolbox import Tool

logger = structlog.get_logger(__name__)

def get_load_forecast(service_name: str, config: dict = None) -> str:
    """
    Retrieves the CPU utilization forecast for a specific service from the AIOps cache.

    Args:
        service_name (str): The name of the service to get the forecast for.

    Returns:
        str: A JSON string of the forecast data, or an error message.
    """
    if not config or "ignite_host" not in config or "ignite_port" not in config:
        return "Error: Ignite connection details are not configured for the tool."

    ignite_host = config["ignite_host"]
    ignite_port = config["ignite_port"]
    forecast_cache_name = "aiops_forecasts"

    logger.info("Fetching load forecast for service", service=service_name)

    try:
        client = Client()
        client.connect(ignite_host, ignite_port)
        cache = client.get_or_create_cache(forecast_cache_name)
        
        forecast_data = cache.get(service_name)
        
        if forecast_data:
            # The data in cache is a dict of {timestamp_iso: forecast_value}.
            # We can return it directly as a JSON string.
            logger.info("Successfully fetched forecast", service=service_name)
            return json.dumps(forecast_data)
        else:
            logger.warning("No forecast data found for service", service=service_name)
            return f"Error: No forecast data found for service '{service_name}'."

    except Exception as e:
        logger.error("Failed to fetch forecast from Ignite", service=service_name, exc_info=True)
        return f"Error: An unexpected error occurred while connecting to the forecast cache: {e}"
    finally:
        if 'client' in locals() and client.is_connected():
            client.close()


predictive_autoscaler_tool = Tool(
    name="get_load_forecast",
    description="Retrieves the 24-hour CPU utilization forecast for a specific service.",
    func=get_load_forecast,
    # The agent setup will need to inject the correct Ignite config here
    config={"ignite_host": "ignite", "ignite_port": 10800}
) 