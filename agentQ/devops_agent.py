import os
import uuid
import yaml
import structlog

from agentQ.app.core.toolbox import Toolbox, Tool
from agentQ.app.core.context import ContextManager
from agentQ.app.core.devops_tools import (
    get_service_dependencies_tool, get_recent_deployments_tool,
    restart_service_tool, list_pods_tool, get_deployment_status_tool,
    scale_deployment_tool
)
# We can also give it general-purpose tools
from agentQ.app.core.human_tool import human_tool
from agentQ.app.core.integrationhub_tool import integrationhub_tool
from agentQ.app.core.meta_tools import list_tools_tool
from agentQ.app.core.workflow_tools import read_context_tool, update_context_tool
from agentQ.app.core.code_search_tool import code_search_tool
from agentQ.app.core.reporting_tool import log_incident_report_tool
from agentQ.app.core.git_tool import propose_code_fix_tool
from agentQ.app.core.delegation_tool import delegation_tool
from agentQ.app.core.prometheus_tool import prometheus_tool
from agentQ.app.core.predictive_autoscaler_tool import predictive_autoscaler_tool
from shared.vault_client import VaultClient

logger = structlog.get_logger("devops_agent")

# --- Agent Definition ---

AGENT_ID = f"devops-agent-{uuid.uuid4()}"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"

DEVOPS_AGENT_SYSTEM_PROMPT = """
You are a DevOps Engineer AI. You are responsible for maintaining the stability, performance, and efficiency of the Q Platform's infrastructure.

**Your Core Workflows:**

1.  **Root Cause Analysis (RCA):**
    -   When an anomaly is detected, you will be tasked with gathering evidence.
    -   Use tools like `prometheus_query_range` and `kubernetes_get_events` to collect metrics and events related to the affected service.
    -   Your findings will be used by other agents to determine the root cause.

2.  **Predictive Autoscaling:**
    -   You will be periodically asked to check service load forecasts.
    -   First, use the `get_load_forecast` tool for the specified service.
    -   Next, analyze the forecast data. If the predicted CPU utilization exceeds the scale-up threshold, use the `k8s_scale_deployment` tool to increase replicas. If it's consistently below the scale-down threshold, decrease replicas.
    -   If no action is needed, state that in your final answer.

3.  **Remediation Execution:**
    -   When a human operator approves a remediation plan, you will be tasked with executing it.
    -   The prompt will contain the exact tool name and parameters. You must execute the specified tool with the provided inputs without deviation.

You are a critical component of the platform's autonomous operations. Execute your tasks with precision and care.
"""

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agent.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_devops_agent(config: dict, vault_client: VaultClient):
    """
    Initializes the toolbox and context manager for the DevOps agent.
    This function would be called by the main agent runner.
    """
    logger.info("Setting up DevOps Agent", agent_id=AGENT_ID)

    devops_toolbox = Toolbox()
    
    # Pass the service URLs and vault client to the tools that need them
    tool_config = {**config['services'], 'vault_client': vault_client}

    devops_toolbox.register_tool(Tool(name=get_service_dependencies_tool.name, description=get_service_dependencies_tool.description, func=get_service_dependencies_tool.func, config=tool_config))
    devops_toolbox.register_tool(Tool(name=get_recent_deployments_tool.name, description=get_recent_deployments_tool.description, func=get_recent_deployments_tool.func, config=tool_config))
    devops_toolbox.register_tool(Tool(name=restart_service_tool.name, description=restart_service_tool.description, func=restart_service_tool.func, config=tool_config))
    devops_toolbox.register_tool(Tool(name=list_pods_tool.name, description=list_pods_tool.description, func=list_pods_tool.func, config=tool_config))
    devops_toolbox.register_tool(Tool(name=get_deployment_status_tool.name, description=get_deployment_status_tool.description, func=get_deployment_status_tool.func, config=tool_config))
    devops_toolbox.register_tool(Tool(name=scale_deployment_tool.name, description=scale_deployment_tool.description, func=scale_deployment_tool.func, config=tool_config))
    devops_toolbox.register_tool(Tool(name=log_incident_report_tool.name, description=log_incident_report_tool.description, func=log_incident_report_tool.func, config=tool_config))
    
    # General tools
    devops_toolbox.register_tool(human_tool)
    devops_toolbox.register_tool(Tool(name=integrationhub_tool.name, description=integrationhub_tool.description, func=integrationhub_tool.func, config=tool_config))
    devops_toolbox.register_tool(delegation_tool)
    devops_toolbox.register_tool(list_tools_tool)
    devops_toolbox.register_tool(read_context_tool)
    devops_toolbox.register_tool(update_context_tool)
    devops_toolbox.register_tool(Tool(name=code_search_tool.name, description=code_search_tool.description, func=code_search_tool.func, config=tool_config))
    devops_toolbox.register_tool(Tool(name=propose_code_fix_tool.name, description=propose_code_fix_tool.description, func=propose_code_fix_tool.func, config=tool_config))
    devops_toolbox.register_tool(prometheus_tool)
    devops_toolbox.register_tool(predictive_autoscaler_tool)
    
    # Context can be shared or agent-specific
    context_manager = ContextManager(ignite_addresses=config['ignite']['addresses'], agent_id=AGENT_ID)
    context_manager.connect()
    
    return devops_toolbox, context_manager

# This file does not run on its own. It's a "definition" to be used by a master agent runner.
# In our current setup, we will modify the main `run_agent` to be able to select an agent "personality". 