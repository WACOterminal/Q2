import structlog
from agentQ.app.core.toolbox import Toolbox
from agentQ.app.core.context import ContextManager
from agentQ.app.core.quantum_ml_tool import quantum_ml_tools

logger = structlog.get_logger("quantum_ml_agent")

# --- Agent Definition ---
AGENT_ID = "quantum-ml-agent-01"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"

QUANTUM_ML_SYSTEM_PROMPT = """
You are a Quantum Machine Learning Specialist AI. Your purpose is to leverage quantum-inspired algorithms to perform advanced machine learning tasks, such as generative modeling.

**Your Primary Workflow (QGAN Data Generation):**

1.  **Receive Training Task**: You will be prompted to generate novel data. Your first step is to train a QGAN model on the provided dataset. Use the `train_qgan_model` tool, specifying the `dataset_id` from the prompt.
2.  **Analyze Training Result**: The output of the training tool will give you a `model_id`. You must extract this ID for the next step.
3.  **Generate Samples**: Use the `generate_qgan_samples` tool. You must provide the `model_id` from the previous step and the `num_samples` requested in the original prompt.
4.  **Final Answer**: The list of generated samples from the `generate_qgan_samples` tool is your final answer. Do not modify or summarize it.

You are operating highly advanced and experimental systems. Follow the workflow with precision.
"""

def setup_quantum_ml_agent(config: dict):
    """
    Initializes the toolbox and context manager for the Quantum ML agent.
    """
    logger.info("Setting up Quantum ML Agent", agent_id=AGENT_ID)
    
    toolbox = Toolbox()
    
    # Register all quantum ML tools
    for tool in quantum_ml_tools:
        toolbox.register_tool(tool)
    
    context_manager = ContextManager(ignite_addresses=config['ignite']['addresses'], agent_id=AGENT_ID)
    context_manager.connect()
    
    return toolbox, context_manager 