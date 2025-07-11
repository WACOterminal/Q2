import structlog
from agentQ.app.core.toolbox import Toolbox
from agentQ.app.core.context import ContextManager
from agentQ.app.core.neuromorphic_tool import neuromorphic_tools

logger = structlog.get_logger("neuromorphic_analyst_agent")

# --- Agent Definition ---
AGENT_ID = "neuromorphic-analyst-agent-01"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"

NEUROMORPHIC_ANALYST_SYSTEM_PROMPT = """
You are a Neuromorphic Analyst AI, a highly specialized agent responsible for operating the platform's Spiking Neural Network (SNN) systems for advanced pattern detection.

**Your Primary Workflow (Continuous Anomaly Detection):**

1.  **Configuration**: Your first action is to configure the SNN to monitor the correct data stream. Use the `configure_snn_for_anomaly_detection` tool. You will be given the `pulsar_topic` in the prompt.
2.  **Monitoring Loop**: Once configured, you will enter a continuous monitoring loop.
    a. Use the `get_snn_anomalies` tool to check for any detected anomalies. The `network_id` will be available from the output of the configuration step.
    b. Wait for a short period (e.g., 5 seconds) before checking again.
3.  **Reporting**: If the `get_snn_anomalies` tool returns any findings, your task is to immediately report them as your final answer. Format the output clearly.
4.  **No Deviation**: You must not deviate from this loop. Your sole focus is on configuring and then persistently querying the SNN for its results. If no anomalies are found, continue the loop.

This is a critical, real-time monitoring task.
"""

def setup_neuromorphic_analyst_agent(config: dict):
    """
    Initializes the toolbox and context manager for the Neuromorphic Analyst agent.
    """
    logger.info("Setting up Neuromorphic Analyst Agent", agent_id=AGENT_ID)
    
    toolbox = Toolbox()
    
    # Register all neuromorphic tools
    for tool in neuromorphic_tools:
        toolbox.register_tool(tool)
    
    context_manager = ContextManager(ignite_addresses=config['ignite']['addresses'], agent_id=AGENT_ID)
    context_manager.connect()
    
    return toolbox, context_manager 