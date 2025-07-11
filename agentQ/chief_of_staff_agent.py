import structlog
import json
import pulsar
from collections import defaultdict, deque
from agentQ.app.core.toolbox import Toolbox
from agentQ.app.core.context import ContextManager
# We will create this tool in the next step
# from agentQ.app.core.proactive_assistance_tool import propose_assistance_tool

logger = structlog.get_logger("chief_of_staff_agent")

# --- Agent Definition ---
AGENT_ID = "chief-of-staff-agent-01" # Singleton agent for a user
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"
USER_ACTION_TOPIC = "persistent://public/default/q.h2m.user_actions"

CHIEF_OF_STAFF_SYSTEM_PROMPT = """
You are a "Chief of Staff" AI. Your sole purpose is to proactively assist your assigned human user by observing their actions and identifying opportunities for automation. You are a silent partner, only speaking when you have a valuable, concrete suggestion.

**Your Cognitive Loop:**
1.  **Observe**: You will receive a continuous stream of your user's actions. Maintain a short-term memory of their recent behavior.
2.  **Identify Patterns**: Analyze the action history to find patterns of inefficiency or repetition. For example:
    -   Does the user frequently navigate between the same two pages?
    -   Does the user manually execute the same sequence of three button clicks multiple times?
    -   Does the user repeatedly view a dashboard and then perform the same action?
3.  **Formulate a Suggestion**: If you identify such a pattern, formulate a clear, concise, and actionable suggestion. Your suggestion should be to create a new workflow to automate the repetitive task.
4.  **Propose Assistance**: Use the `propose_assistance` tool to present your suggestion to the user.

You do not respond to direct prompts. Your work is entirely proactive, based on observation.
"""

class ChiefOfStaffAgent:
    def __init__(self, pulsar_client: pulsar.Client, config: dict):
        self.pulsar_client = pulsar_client
        self.toolbox = Toolbox()
        # self.toolbox.register_tool(propose_assistance_tool) # Will be uncommented later
        self.user_histories = defaultdict(lambda: deque(maxlen=50))
        self._running = False

    def start(self):
        self._running = True
        self._consumer_thread = asyncio.create_task(self._consumer_loop())
        logger.info("Chief of Staff Agent started and listening for user actions.")

    def stop(self):
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.cancel()

    async def _consumer_loop(self):
        consumer = self.pulsar_client.subscribe(
            USER_ACTION_TOPIC,
            subscription_name='chief-of-staff-agent-sub'
        )
        while self._running:
            try:
                msg = consumer.receive(timeout_millis=1000)
                await self._analyze_action(msg)
                consumer.acknowledge(msg)
            except pulsar.Timeout:
                continue
            except Exception as e:
                logger.error("Error in ChiefOfStaffAgent consumer loop", exc_info=True)
                if 'msg' in locals(): consumer.negative_acknowledge(msg)

    async def _analyze_action(self, msg: pulsar.Message):
        try:
            action = json.loads(msg.data().decode('utf-8'))
            user_id = action.get("user_id")
            if not user_id: return

            history = self.user_histories[user_id]
            history.append(action['type']) # Just store the type for simple pattern matching

            # --- Simple Pattern Matching Logic ---
            # Look for 3 identical actions in a row
            if len(history) > 3 and len(set(list(history)[-3:])) == 1:
                last_action = history[-1]
                logger.info(f"Detected repetitive action pattern for user {user_id}", action=last_action)
                
                # Clear history to avoid re-triggering
                history.clear()
                
                # Propose a workflow to automate it
                suggestion_text = f"I've noticed you've performed the action '{last_action}' multiple times. I can create a workflow to automate this for you."
                # This would call the real tool in the full implementation
                # self.toolbox.execute_tool(
                #     "propose_assistance", 
                #     user_id=user_id, 
                #     suggestion_text=suggestion_text,
                #     action_type="CREATE_WORKFLOW",
                #     action_payload={"name": f"Automated '{last_action}'"}
                # )
                
        except Exception as e:
            logger.error("Failed to analyze user action", exc_info=True)

def setup_chief_of_staff_agent(config: dict):
    # This agent is long-running and not personality-based like the others,
    # so its setup and execution will be handled differently in main.py.
    pass 