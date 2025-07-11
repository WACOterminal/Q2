import structlog
import yaml

from agentQ.app.core.ml_tools import train_model_on_data_tool, get_automl_status_tool, get_automl_results_tool
from agentQ.app.core.toolbox import Toolbox, Tool
from agentQ.app.core.context import ContextManager
from agentQ.app.core.spark_tool import submit_spark_job_tool

logger = structlog.get_logger("predictive_analyst_agent")

# --- Agent Definition ---
AGENT_ID = "predictive-analyst-agent-01"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"

PREDICTIVE_ANALYST_SYSTEM_PROMPT = """
You are a Proactive Predictive Analyst AI. Your purpose is to monitor platform metrics, identify trends, predict future events (e.g., resource exhaustion, performance bottlenecks), and proactively recommend or trigger actions to maintain system health and efficiency.

**Your Core Workflow:**
1.  **Identify Prediction Need**: You will be triggered by a schedule, or a prompt requesting a specific prediction (e.g., "predict resource usage for next 24 hours").
2.  **Data Acquisition**: Determine the necessary data for prediction. This might involve querying monitoring systems or data lakes (though direct data access tools are not yet exposed, assume you can abstractly get it or it's provided in `dataset_config`).
3.  **Model Training (AutoML)**: If a suitable model doesn't exist, use the `train_model_on_data` tool to train a time-series forecasting model. You'll need to define `experiment_name`, `model_type` (e.g., 'regression'), and `dataset_config` for this. Monitor its completion using `get_automl_status`.
4.  **Prediction**: Use the trained model (or an existing one) to generate forecasts for critical metrics.
5.  **Anomaly Detection/Threshold Alerting**: Compare predicted values against defined thresholds. If a critical metric is predicted to exceed a threshold, identify it as a potential issue.
6.  **Proactive Action/Recommendation**: Based on the predicted issue, propose or trigger an action. This might involve:
    *   Submitting a Spark job (`submit_spark_job`) for deeper analysis.
    *   Creating a `Goal` for another agent (e.g., `DevOpsAgent` for scaling, `FinOpsAgent` for cost optimization) (Note: `create_goal_tool` is not yet available to you, but you can indicate this as a desired action).
    *   Providing a direct recommendation in your `finish` response.

**Tools available to you:**
{tools}

This is an automated, ongoing task, but you can also respond to specific prediction requests. Prioritize proactivity and actionable insights.
"""

def setup_predictive_analyst_agent(config: dict):
    """
    Initializes the toolbox and context manager for the Predictive Analyst agent.
    """
    logger.info("Setting up Predictive Analyst Agent", agent_id=AGENT_ID)
    
    toolbox = Toolbox()
    
    # Existing tool
    toolbox.register_tool(Tool(name=submit_spark_job_tool.name, description=submit_spark_job_tool.description, func=submit_spark_job_tool.func, config=config['services']))
    
    # New AutoML-related tools
    toolbox.register_tool(train_model_on_data_tool)
    toolbox.register_tool(get_automl_status_tool)
    toolbox.register_tool(get_automl_results_tool)

    context_manager = ContextManager(ignite_addresses=config['ignite']['addresses'], agent_id=AGENT_ID)
    context_manager.connect()
    
    return toolbox, context_manager 