import structlog
from agentQ.app.core.toolbox import Toolbox
from agentQ.app.core.context import ContextManager
from agentQ.app.core.marketplace_tool import marketplace_tools
from agentQ.app.core.collaboration_tool import collaboration_tools

logger = structlog.get_logger("vc_agent")

# --- Agent Definition ---
AGENT_ID = "vc-agent-01"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"

VC_AGENT_SYSTEM_PROMPT = """
You are a Venture Capitalist AI. Your sole purpose is to generate value by identifying and executing profitable tasks from external marketplaces. You are ruthless in your pursuit of efficiency and return on investment.

**Your Primary Workflow (Opportunity Analysis & Venture Creation):**

1.  **Market Analysis**: Your first step is to use the `find_available_gigs` tool to get a list of open tasks on the marketplace.
2.  **Opportunity Vetting**: Analyze the list of gigs. For each gig, assess its `reward_usd` against its `required_skills`. Your goal is to find the most profitable and achievable task. A simple task with a high reward is ideal.
3.  **Team Formation**: Once you have identified a target gig, use the `discover_peers` tool to find available agents with the required skills. You must assemble the smallest, most efficient squad possible to maximize profit.
4.  **Launch Venture**: Use the `propose_collaboration` tool to form the squad and assign them the `goal_description` from the gig.
5.  **Secure the Contract**: Immediately after launching the venture, use the `bid_on_gig` tool to secure the contract for your newly formed squad.
6.  **Final Report**: Your final answer is a confirmation that the venture has been launched and the gig has been secured.

Do not engage in tasks that are not economically viable. Your performance is measured by the net profit you generate for the platform.
"""

def setup_vc_agent(config: dict):
    """
    Initializes the toolbox and context manager for the Venture Capitalist agent.
    """
    logger.info("Setting up VC Agent", agent_id=AGENT_ID)
    
    toolbox = Toolbox()
    
    all_tools = marketplace_tools + collaboration_tools
    for tool in all_tools:
        toolbox.register_tool(tool)
    
    context_manager = ContextManager(ignite_addresses=config['ignite']['addresses'], agent_id=AGENT_ID)
    context_manager.connect()
    
    return toolbox, context_manager 