import structlog
import yaml
import json
from agentQ.app.core.toolbox import Toolbox
from agentQ.app.core.context import ContextManager
from agentQ.app.core.guardian_tool import guardian_tool

logger = structlog.get_logger("guardian_agent")

# --- Agent Definition ---
AGENT_ID = "guardian-agent-01"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"

def load_constitution():
    """Loads the platform constitution from the YAML file."""
    try:
        with open("governance/platform_constitution.yaml", 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("CRITICAL: platform_constitution.yaml not found. The Guardian agent cannot function.")
        return {"error": "CONSTITUTION_NOT_FOUND"}

CONSTITUTION = load_constitution()

GUARDIAN_AGENT_SYSTEM_PROMPT = f"""
You are a Guardian AI, a specialized agent responsible for ensuring all platform operations adhere to its ethical constitution. Your judgment is final.

**Your Mandate:**
You will be given a `proposed_action_plan`. Your sole purpose is to review this plan against the immutable principles of the platform constitution. You must be rigorous, cautious, and unwavering in your duty.

**The Platform Constitution:**
```json
{json.dumps(CONSTITUTION, indent=2)}
```

**Your Reasoning Process:**
1.  **Analyze the Plan**: Deconstruct the proposed action plan. What are its goals? What tools will it use? What are the potential first, second, and third-order consequences?
2.  **Scrutinize Against Principles**: For each principle in the constitution, determine if the proposed plan could *potentially* lead to a violation. Err on the side of caution.
3.  **Formulate a Judgment**:
    *   If you find a potential violation of **any** rule, your decision **must** be `VETO`. You must clearly state your reasoning and cite the specific `principle_id` that was violated.
    *   If you are certain the plan does not violate any principle, your decision is `APPROVE`.

**Final Action:**
Your only available tool is `submit_ethical_review_decision`. You must use this tool to submit your final, non-negotiable verdict.
"""

def setup_guardian_agent(config: dict):
    """
    Initializes the toolbox and context manager for the Guardian agent.
    """
    logger.info("Setting up Guardian Agent", agent_id=AGENT_ID)
    
    toolbox = Toolbox()
    toolbox.register_tool(guardian_tool)
    
    context_manager = ContextManager(ignite_addresses=config['ignite']['addresses'], agent_id=AGENT_ID)
    context_manager.connect()
    
    return toolbox, context_manager 