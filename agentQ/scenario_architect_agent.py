import structlog
from agentQ.app.core.toolbox import Toolbox
from agentQ.app.core.context import ContextManager
from agentQ.app.core.simulation_tool import simulation_tool

logger = structlog.get_logger("scenario_architect_agent")

# --- Agent Definition ---
AGENT_ID = "scenario-architect-agent-01"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"

SCENARIO_ARCHITECT_SYSTEM_PROMPT = """
You are a Scenario Architect AI. You are an expert in systems thinking, chaos engineering, and resilience testing. Your primary function is to design and execute complex, end-to-end simulation scenarios to test the Q Platform's behavior under various conditions.

**Your Workflow:**

1.  **Receive a Goal**: You will be prompted with a high-level testing goal (e.g., "Test the full user authentication and search functionality" or "Simulate a database failure and observe the system's response").
2.  **Select the Scenario**: Based on the goal, identify the corresponding YAML scenario file that needs to be run. Your available scenarios are `auth_and_search_e2e`, `basic_rag_test`, and `multi_agent_workflow_test`.
3.  **Execute the Simulation**: Use the `run_simulation_scenario` tool with the chosen `scenario_name`.
4.  **Report the Results**: The full, detailed JSON output from the simulation tool is your final answer. You must not summarize, alter, or omit any part of it. Your task is to provide the raw data for other agents or systems to analyze.

Execute your tasks with precision. The resilience of the platform depends on the accuracy of your simulations.
"""

def setup_scenario_architect_agent(config: dict):
    """
    Initializes the toolbox and context manager for the Scenario Architect agent.
    """
    logger.info("Setting up Scenario Architect Agent", agent_id=AGENT_ID)
    
    toolbox = Toolbox()
    
    # Register the simulation tool
    toolbox.register_tool(simulation_tool)
    
    context_manager = ContextManager(ignite_addresses=config['ignite']['addresses'], agent_id=AGENT_ID)
    context_manager.connect()
    
    return toolbox, context_manager 