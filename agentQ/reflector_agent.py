import logging
import json

from agentQ.app.core.base_agent import BaseAgent
from agentQ.app.core.insight_tool import save_insight_tool
from shared.q_pulse_client.models import QPChatRequest, QPChatMessage

logger = logging.getLogger(__name__)

KNOWLEDGE_ENGINEER_SYSTEM_PROMPT = """
You are a Reflector AI. Your purpose is to analyze the results of other agents' work to synthesize findings, identify root causes, or suggest improvements.

**Your Core Tasks:**

1.  **Root Cause Analysis (RCA) Synthesis:**
    -   You will be given a prompt containing the collected `Metrics`, `Logs`, and `Kubernetes Events` related to a service anomaly.
    -   **Your Goal**: Correlate the information from all three sources to determine the most probable root cause.
    -   **Output Structure**: You MUST structure your response as a markdown document with the following sections:
        -   `## Root Cause Analysis Report`
        -   `**Summary**: A one-sentence summary of your conclusion.`
        -   `**Evidence**: A bulleted list of the specific findings from metrics, logs, and events that led you to your conclusion. Be precise.`
        -   `**Recommendation**: A clear, actionable next step for remediation (e.g., "Restart the deployment", "Increase memory limits").`

2.  **Meta-Analysis and Improvement (Default Task):**
    -   If not performing a specific RCA task, your goal is to analyze completed workflows to find patterns and suggest improvements.
    -   Use the `get_completed_workflows` tool to fetch data.
    -   Analyze the steps, outcomes, and any failures.
    -   Use the `propose_workflow_improvement` tool to submit your suggestions.

Prioritize the RCA task if the prompt contains the required data.
"""

class ReflectorAgent(BaseAgent):
    def __init__(self, qpulse_url: str):
        super().__init__(
            qpulse_url=qpulse_url,
            system_prompt=KNOWLEDGE_ENGINEER_SYSTEM_PROMPT,
            tools=[save_insight_tool]
        )

    async def run(self, workflow_json: str) -> str:
        # For this specialized agent, we'll keep the one-shot call for simplicity,
        # as it doesn't need a complex ReAct loop. It formulates a thought and then acts.
        workflow_data = json.loads(workflow_json)
        
        # The agent's "thought" process is to create the prompt for the LLM.
        prompt_for_llm = f"""
        Analyze the following workflow and generate the parameters for the `save_insight` tool.
        
        Workflow Record:
        {json.dumps(workflow_data, indent=2)}

        Respond with ONLY a single `call_tool` action JSON object.
        """
        
        full_prompt = f"{self.system_prompt.format(tools=self.toolbox.get_tool_descriptions())}\n\n{prompt_for_llm}"

        messages = [QPChatMessage(role="system", content=full_prompt)]
        request = QPChatRequest(model="gpt-4-turbo", messages=messages, max_tokens=1024)
        
        response = await self.qpulse_client.get_chat_completion(request)
        action_json_str = response.choices[0].message.content
        
        try:
            action_json = json.loads(action_json_str)
            if action_json.get("action") == "call_tool" and action_json.get("tool_name") == "save_insight":
                params = action_json.get("parameters", {})
                # Execute the tool and return its result
                return self.toolbox.execute_tool("save_insight", **params)
            else:
                raise ValueError("LLM did not return the expected 'save_insight' tool call.")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to decode or execute reflector agent action: {e}", exc_info=True)
            return f"Error: Failed to generate and save insight. Reason: {e}" 