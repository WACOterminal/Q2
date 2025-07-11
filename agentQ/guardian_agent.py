import structlog
import yaml
import json
import asyncio

from agentQ.app.core.toolbox import Toolbox
from agentQ.app.core.context import ContextManager
from agentQ.app.core.guardian_tool import guardian_tool
from shared.q_pulse_client.client import QuantumPulseClient
from shared.q_pulse_client.models import QPChatRequest, QPChatMessage

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

async def critique_action(proposed_action_plan: dict, qpulse_client: QuantumPulseClient, llm_config: dict) -> dict:
    """
    Uses an LLM to critique a proposed action plan against the constitution.

    Args:
        proposed_action_plan: A dictionary describing the action to be taken.
        qpulse_client: The client for making calls to the LLM service.
        llm_config: Configuration for the LLM call (e.g., model name).

    Returns:
        A dictionary containing the verdict ('APPROVE' or 'VETO') and reasoning.
    """
    logger.info("Guardian agent is critiquing proposed action", plan=proposed_action_plan)

    # Format the prompt for the LLM
    prompt = GUARDIAN_AGENT_SYSTEM_PROMPT.format(
        proposed_action_plan=json.dumps(proposed_action_plan, indent=2)
    )

    try:
        messages = [QPChatMessage(role="system", content=prompt)]
        request = QPChatRequest(model=llm_config.get('model', 'q-pulse-master'), messages=messages)

        response = await qpulse_client.get_chat_completion(request)
        critique_text = response.choices[0].message.content

        # The LLM's response should ideally be structured JSON, but for now,
        # we'll parse a simplified text format.
        # A more robust solution would enforce JSON output from the LLM.
        if "VETO" in critique_text.upper():
            verdict = "VETO"
        else:
            verdict = "APPROVE"

        reasoning_start = critique_text.find("Reasoning:")
        reasoning = critique_text[reasoning_start:].strip() if reasoning_start != -1 else critique_text

        logger.info("Guardian agent critique complete", verdict=verdict)
        return {"verdict": verdict, "reasoning": reasoning}

    except Exception as e:
        logger.error("Guardian agent failed to get critique from LLM", error=str(e), exc_info=True)
        # Fail-safe: In case of any error, default to vetoing the action.
        return {
            "verdict": "VETO",
            "reasoning": "Guardian agent encountered an internal error and could not complete the review. "
                         "The action is vetoed as a safety precaution."
        }


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