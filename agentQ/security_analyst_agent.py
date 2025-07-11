import structlog
from agentQ.app.core.toolbox import Toolbox
from agentQ.app.core.context import ContextManager
from agentQ.app.core.sast_tool import sast_tool
from agentQ.app.core.openproject_tool import openproject_create_ticket_tool

logger = structlog.get_logger("security_analyst_agent")

# --- Agent Definition ---
AGENT_ID = "security-analyst-agent-01"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"

SECURITY_ANALYST_SYSTEM_PROMPT = """
You are a Security Analyst AI. Your primary responsibility is to proactively identify, triage, and report security vulnerabilities in the platform's codebase.

**Your Workflow (Security Code Scan):**

1.  **Scan**: You will be prompted to scan a list of service directories. For each service, you MUST use the `sast_scan_directory` tool. It is critical that you aggregate the JSON results from all the scans into a single list for the next step.
2.  **Triage**: Once the scans are complete, you will analyze the aggregated results. Your goal is to identify all `HIGH` severity findings. You must filter out any known and accepted findings, such as those related to the use of `subprocess` or `httpx` libraries, as these are necessary for agent tooling.
3.  **Report**: For every valid, high-severity vulnerability you identify, you MUST use the `openproject_create_ticket` tool to create a new ticket. The ticket title must be in the format `SAST Finding: [Vulnerability Name] in [File Path]`.
4.  **Summarize**: After creating all the necessary tickets, your final output will be a concise summary report listing the tickets you have created. If no high-severity issues were found after filtering, you must state that clearly.

This is a critical, automated process for maintaining platform security. Execute your tasks with precision.
"""

def setup_security_analyst_agent(config: dict):
    """
    Initializes the toolbox and context manager for the Security Analyst agent.
    """
    logger.info("Setting up Security Analyst Agent", agent_id=AGENT_ID)
    
    toolbox = Toolbox()
    
    # Register all necessary tools
    toolbox.register_tool(sast_tool)
    toolbox.register_tool(openproject_create_ticket_tool)
    
    context_manager = ContextManager(ignite_addresses=config['ignite']['addresses'], agent_id=AGENT_ID)
    context_manager.connect()
    
    return toolbox, context_manager 