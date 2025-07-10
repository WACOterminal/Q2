import logging
import json

from agentQ.app.core.base_agent import BaseAgent
from agentQ.app.core.insight_tool import save_insight_tool
from shared.q_pulse_client.models import QPChatRequest, QPChatMessage

logger = logging.getLogger(__name__)

REFLECTOR_SYSTEM_PROMPT = """
You are a Reflector Agent. Your purpose is to analyze a completed workflow to find insights and lessons.
You will be given a detailed JSON record of a completed workflow.

Your process is:
1.  **Analyze the record:** Carefully read the workflow to understand the original goal and how the agents attempted to achieve it.
2.  **Formulate a lesson:** Create a single, concise, and actionable "lesson learned" that could help an agent make a better decision in a similar situation in the future.
3.  **Store the lesson:** Use the `save_insight` tool to save your lesson to the knowledge graph. You must extract the `original_prompt`, `final_status`, and your formulated `lesson_learned` from the prompt and pass them to the tool.
4.  **Finish:** Once the insight is stored, your job is complete. The result of the `save_insight` tool will be your final answer.

You have one tool available:
{tools}

Begin!
"""

class ReflectorAgent(BaseAgent):
    def __init__(self, qpulse_url: str):
        super().__init__(
            qpulse_url=qpulse_url,
            system_prompt=REFLECTOR_SYSTEM_PROMPT,
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