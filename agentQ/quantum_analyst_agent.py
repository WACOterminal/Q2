import structlog
from agentQ.app.core.toolbox import Toolbox
from agentQ.app.core.context import ContextManager
from agentQ.app.core.quantum_tool import quantum_routing_tool

logger = structlog.get_logger("quantum_analyst_agent")

# --- Agent Definition ---
AGENT_ID = "quantum-analyst-agent-01"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"

QUANTUM_ANALYST_SYSTEM_PROMPT = """
You are a Quantum Analyst AI. You specialize in solving complex optimization problems by leveraging quantum computing algorithms.

**Your Primary Workflow (LLM Routing Optimization):**

1.  **Receive Problem**: You will be given a prompt containing a list of LLM providers, each with its own cost and latency metrics.
2.  **Formulate Input**: Your task is to correctly format this information into the JSON string required by the `solve_llm_routing_problem` tool.
3.  **Execute Tool**: Call the `solve_llm_routing_problem` tool with the prepared JSON input.
4.  **Return Result**: The JSON output from the tool, containing the optimal provider and the reasons for the choice, will be your final answer. You must not alter or summarize it.

Execute this process with precision. The integrity of the optimization depends on it.
"""

def setup_quantum_analyst_agent(config: dict):
    """
    Initializes the toolbox and context manager for the Quantum Analyst agent.
    """
    logger.info("Setting up Quantum Analyst Agent", agent_id=AGENT_ID)
    
    toolbox = Toolbox()
    
    # Register the quantum tool
    toolbox.register_tool(quantum_routing_tool)
    
    context_manager = ContextManager(ignite_addresses=config['ignite']['addresses'], agent_id=AGENT_ID)
    context_manager.connect()
    
    return toolbox, context_manager 