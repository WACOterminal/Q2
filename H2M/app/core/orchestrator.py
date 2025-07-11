import logging
from typing import Dict, List, AsyncGenerator, Any
from jinja2 import Environment, FileSystemLoader
import json

from app.core.context import context_manager
from app.core.rag import rag_module
from app.core.config import get_config
from shared.q_pulse_client.client import QuantumPulseClient
from shared.q_pulse_client.models import InferenceRequest, QPChatRequest, QPChatMessage
from app.services.pulsar_client import h2m_pulsar_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Prompt Templating Setup ---
# A more robust approach would be to have a dedicated templates directory
PROMPT_TEMPLATE = """
System: You are a helpful and professional AI assistant. Answer the user's question based on the provided context. If the context is not relevant or does not contain the answer, say that you do not have enough information to answer. Do not make up information.

{% if rag_context %}
--- CONTEXT ---
{{ rag_context }}
--- END CONTEXT ---
{% endif %}

{% for message in history %}
{{ message.role | title }}: {{ message.content }}
{% endfor %}
User: {{ user_query }}
Assistant:
"""
jinja_env = Environment()
prompt_template = jinja_env.from_string(PROMPT_TEMPLATE)


class ConversationOrchestrator:
    """
    Orchestrates the entire process of handling a user's message.
    """

    def __init__(self):
        services_config = get_config().services
        self.qp_client = QuantumPulseClient(base_url=services_config.quantumpulse_url)

    async def handle_message_stream(self, user_id: str, text: str, conversation_id: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        The main streaming method to process a user's message. It yields
        message chunks (tokens) as they are generated by the LLM.

        Args:
            user_id: The unique ID of the authenticated user.
            text: The user's input message.
            conversation_id: The existing conversation ID, if any.

        Yields:
            A dictionary for each chunk of the response.
        """
        logger.info(f"Orchestrator: Handling message stream for user '{user_id}' in conversation '{conversation_id}'")

        # 1. Get conversation history
        conv_id, history = await context_manager.get_or_create_conversation_history(user_id, conversation_id)

        # 2. Get RAG context
        rag_context = await rag_module.retrieve_context(text)

        # 3. Construct the final prompt
        final_prompt = self._build_prompt(text, history, rag_context)

        # 4. Submit to QuantumPulse for streaming inference
        inference_request = QPChatRequest(
            model="gpt-4-turbo", # This should be configurable
            messages=[QPChatMessage(role="user", content=final_prompt)],
            stream=True
        )
        
        full_response_text = ""
        try:
            stream = self.qp_client.get_chat_completion_stream(inference_request)
            async for chunk in stream:
                # The raw chunk is an SSE message, e.g., "data: {...}\n\n"
                if chunk.strip():
                    sse_data = chunk.replace("data: ", "").strip()
                    try:
                        chunk_json = json.loads(sse_data)
                        delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            full_response_text += content
                            yield {"type": "token", "text": content, "conversation_id": conv_id}
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode SSE chunk: {sse_data}")
                        continue
            
            yield {"type": "final", "text": "End of stream."}

        except Exception as e:
            logger.error(f"Error during inference stream: {e}", exc_info=True)
            yield {"type": "error", "text": "An error occurred during streaming."}
        finally:
            # 5. Save the new turn to the conversation history
            await context_manager.add_message_to_history(user_id, conv_id, text, full_response_text)
            logger.info(f"Orchestrator: Successfully handled and saved message stream for conversation {conv_id}")

    def _build_prompt(self, user_query: str, history: List[Dict], rag_context: str) -> str:
        """
        Builds the final prompt to be sent to the language model using Jinja2.
        """
        prompt = prompt_template.render(
            rag_context=rag_context,
            history=history,
            user_query=user_query
        )
        logger.debug(f"Constructed final prompt:\n{prompt}")
        return prompt

# Global instance for the application
orchestrator = ConversationOrchestrator() 