import httpx
import structlog
import json
from agentQ.app.core.toolbox import Tool

logger = structlog.get_logger(__name__)

def solve_llm_routing_problem(providers_json: str, config: dict = None) -> str:
    """
    Invokes the QuantumPulse optimization service to solve the LLM routing problem.

    Args:
        providers_json (str): A JSON string representing a list of provider dictionaries.
                              Each dict must have 'name', 'cost_per_1k_tokens', and 'p90_latency_ms'.
                              Example: '[{"name": "provider-a", "cost_per_1k_tokens": 0.5, "p90_latency_ms": 500}]'

    Returns:
        str: A JSON string of the optimization result.
    """
    quantumpulse_url = config.get("quantumpulse_url")
    if not quantumpulse_url:
        return "Error: quantumpulse_url is not configured."

    try:
        # The agent will pass a JSON string, so we need to parse it
        providers = json.loads(providers_json)
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for providers_json. It must be a valid JSON array."

    request_url = f"{quantumpulse_url}/v1/optimize/llm-routing"
    logger.info("Calling QuantumPulse optimization service", url=request_url)

    try:
        with httpx.Client() as client:
            response = client.post(request_url, json={"providers": providers}, timeout=120.0)
            response.raise_for_status()
            return response.text
    except httpx.RequestError as e:
        logger.error("Failed to call QuantumPulse service", error=str(e))
        return f"Error: Request to QuantumPulse failed. Reason: {e}"
    except Exception as e:
        logger.error("An unexpected error occurred during quantum tool execution", exc_info=True)
        return f"Error: An unexpected error occurred: {e}"

quantum_routing_tool = Tool(
    name="solve_llm_routing_problem",
    description="Solves the LLM provider routing problem by finding the optimal balance between cost and latency using a quantum algorithm.",
    func=solve_llm_routing_problem,
    config={"quantumpulse_url": "http://quantumpulse-api.q-platform.svc.cluster.local:8000"}
) 