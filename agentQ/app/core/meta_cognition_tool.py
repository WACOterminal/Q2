import structlog
import json
from agentQ.app.core.toolbox import Tool
from typing import List, Dict, Any, Optional

logger = structlog.get_logger(__name__)

def analyze_performance(task_history: List[Dict[str, Any]]) -> str:
    """
    Analyzes a given task history to identify patterns, inefficiencies, or errors.
    Returns a text-based analysis.

    Args:
        task_history (List[Dict[str, Any]]): A list of task records, where each record
                                             is a dictionary containing details like
                                             'thought', 'action', and 'timestamp'.

    Returns:
        A string containing a summary of performance insights.
    """
    logger.info("Analyzing task performance history.")
    if not task_history:
        return "No task history provided to analyze."

    # --- Placeholder Analysis Logic ---
    insights = []
    action_counts = {}
    
    for task in task_history:
        action = task.get("action")
        if action:
            action_counts[action] = action_counts.get(action, 0) + 1

    # Insight 1: Find repeated actions
    for action, count in action_counts.items():
        if count > 1:
            insights.append(f"The action '{action}' was repeated {count} times. This might indicate an inefficient loop or a recurring problem.")

    # Insight 2: Check for error patterns (simple keyword search)
    error_keywords = ["fail", "error", "unable", "could not"]
    for task in task_history:
        thought = task.get("thought", "").lower()
        for keyword in error_keywords:
            if keyword in thought:
                insights.append(f"A thought process contained the keyword '{keyword}', suggesting a potential failure or struggle. Reviewing this step is advised.")
                break # Move to the next task

    if not insights:
        return "Performance analysis complete. No obvious inefficiencies or errors were detected in the provided history."

    return "Performance analysis complete. The following insights were found:\n- " + "\n- ".join(insights)


def get_agent_thought_history(agent_personality: str, limit: int = 50, config: Optional[Dict] = None) -> str:
    """
    Retrieves the recent thought process history for a specific agent personality.
    This includes the final thought for each of the last N completed tasks.
    (This is a mock tool and returns generated data).

    Args:
        agent_personality (str): The personality of the agent to query (e.g., 'devops_agent').
        limit (int): The maximum number of recent thoughts to retrieve.

    Returns:
        str: A JSON string list of thought records.
    """
    logger.info("Fetching thought history for agent personality", agent_personality=agent_personality)
    
    # In a real system, this would query a database (e.g., Elasticsearch or Ignite)
    # where the full task results (including the 'thought' JSON blob) are stored.
    mock_history = [
        {
            "thought": "The user wants to restart a service. The `k8s_restart_deployment` tool is the most direct way to achieve this. I have the service name and namespace, so I can proceed.",
            "action": "k8s_restart_deployment",
            "timestamp": "2024-08-03T10:00:00Z"
        },
        {
            "thought": "The user asked for logs. The `elasticsearch_query` tool is appropriate. I will search for errors in the specified service.",
            "action": "elasticsearch_query",
            "timestamp": "2024-08-03T09:30:00Z"
        },
        {
            "thought": "The user wants to restart a service, but I don't have the namespace. I must first ask the user for the missing information.",
            "action": "ask_human_for_clarification",
            "timestamp": "2024-08-03T09:00:00Z"
        }
    ]
    
    return json.dumps(mock_history[:limit], indent=2)

def critique_and_refine_prompt(original_prompt: str, context: Dict[str, Any]) -> str:
    """
    Analyzes an agent's own prompt for ambiguity and refines it for clarity.
    In a real implementation, this would use an LLM.

    Args:
        original_prompt (str): The initial prompt for the task.
        context (Dict[str, Any]): The agent's current context, which might
                                   influence the refinement.

    Returns:
        A refined, more detailed prompt string.
    """
    logger.info("Critiquing and refining prompt", original_prompt=original_prompt)

    # --- Placeholder Refinement Logic ---
    # This would be a call to an LLM with a specialized "prompt critique" meta-prompt.
    refined_prompt = (
        "Refined Directive: The original goal was to '{original_prompt}'. "
        "To ensure success, focus on breaking the problem down into smaller, verifiable steps. "
        "First, validate all required parameters are present before calling any tool. "
        "Proceed with the original goal, keeping these instructions in mind."
    ).format(original_prompt=original_prompt)
    
    return refined_prompt

meta_cognition_tool = Tool(
    name="get_agent_thought_history",
    description="Retrieves the recent reasoning history (thought processes) for a specific type of agent.",
    func=get_agent_thought_history
)

analyze_performance_tool = Tool(
    name="analyze_task_performance",
    description="Analyzes a list of an agent's past actions and thoughts to find inefficiencies or patterns.",
    func=analyze_performance
)

critique_prompt_tool = Tool(
    name="critique_and_refine_prompt",
    description="Analyzes an agent's own prompt and refines it for clarity and to prevent common errors.",
    func=critique_and_refine_prompt
) 