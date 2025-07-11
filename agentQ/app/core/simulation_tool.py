import httpx
import structlog
import json
import time
from agentQ.app.core.toolbox import Tool

logger = structlog.get_logger(__name__)

def run_simulation_scenario(scenario_name: str, config: dict = None) -> str:
    """
    Triggers a simulation scenario in the AgentSandbox and waits for the result.

    Args:
        scenario_name (str): The name of the scenario YAML file (without the .yaml extension).

    Returns:
        str: A JSON string of the final simulation result, or an error message.
    """
    sandbox_url = config.get("agentsandbox_url")
    if not sandbox_url:
        return "Error: agentsandbox_url is not configured."

    run_url = f"{sandbox_url}/simulations/run/{scenario_name}"
    logger.info("Triggering simulation scenario", scenario=scenario_name, url=run_url)

    try:
        with httpx.Client(timeout=30.0) as client:
            # Start the simulation
            response = client.post(run_url)
            response.raise_for_status()
            sim_id = response.json()["simulation_id"]
            logger.info("Simulation started", simulation_id=sim_id)

            # Poll for the result
            status_url = f"{sandbox_url}/simulations/status/{sim_id}"
            for _ in range(60): # Poll for up to 5 minutes (60 * 5s)
                time.sleep(5)
                status_response = client.get(status_url)
                status_data = status_response.json()
                
                if status_data["status"].startswith("COMPLETED"):
                    logger.info("Simulation finished", status=status_data["status"])
                    return json.dumps(status_data["details"], indent=2)
            
            return json.dumps({"status": "TIMEOUT", "details": "Simulation did not complete within the time limit."})

    except httpx.RequestError as e:
        logger.error("Failed to call AgentSandbox service", error=str(e))
        return f"Error: Request to AgentSandbox failed. Reason: {e}"
    except Exception as e:
        logger.error("An unexpected error occurred during simulation", exc_info=True)
        return f"Error: An unexpected error occurred: {e}"

simulation_tool = Tool(
    name="run_simulation_scenario",
    description="Runs a full, end-to-end simulation scenario in a secure sandbox and returns the detailed results.",
    func=run_simulation_scenario,
    config={"agentsandbox_url": "http://localhost:8004/api/v1"}
) 