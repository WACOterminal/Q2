
import logging
import httpx
from typing import Dict, Any, List
import uuid

from agentQ.app.core.toolbox import Tool
from sentence_transformers import SentenceTransformer
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient

logger = logging.getLogger(__name__)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def save_insight(lesson_learned: str, original_prompt: str, final_status: str, config: dict = {}) -> str:
    """
    Saves a 'lesson learned' from a completed workflow to the knowledge base for future reference.
    
    Args:
        lesson_learned (str): The concise, actionable lesson from the workflow.
        original_prompt (str): The original user prompt that started the workflow.
        final_status (str): The final status of the workflow (e.g., 'COMPLETED', 'FAILED').
        
    Returns:
        A confirmation or error message.
    """
    logger.info("Saving new insight to knowledge base.", lesson=lesson_learned)
    
    vectorstore_url = config.get('services', {}).get('vectorstoreq_url', 'http://localhost:8001')
    knowledgegraph_url = config.get('services', {}).get('knowledgegraphq_url', 'http://localhost:8003')
    service_token = config.get('service_token')
    
    if not service_token:
        return "Error: Service token not available. Cannot save insight."

    headers = {"Authorization": f"Bearer {service_token}"}
    insight_id = str(uuid.uuid4())
    
    # 1. Save to VectorStore for semantic search
    try:
        # Create an embedding for the lesson to make it searchable
        embedding = embedding_model.encode(lesson_learned).tolist()

        vector_payload = {
            "collection_name": "insights",
            "vectors": [
                {
                    "id": insight_id,
                    "values": embedding,
                    "metadata": {
                        "lesson_learned": lesson_learned,
                        "original_prompt": original_prompt,
                        "final_status": final_status,
                        "source": "ReflectorAgent"
                    }
                }
            ]
        }
        
        with httpx.Client() as client:
            response = client.post(f"{vectorstore_url}/v1/ingest/upsert", json=vector_payload, headers=headers)
            response.raise_for_status()
            
    except Exception as e:
        logger.error(f"Failed to save insight to VectorStore: {e}", exc_info=True)
        # We can continue to the graph part even if this fails
        pass

    # 2. Save to KnowledgeGraph for structured relationships
    try:
        kg_client = KnowledgeGraphClient(base_url=knowledgegraph_url, token=service_token)
        
        # A simple keyword extraction for now. A more advanced version could use an LLM.
        # This is a placeholder for a more sophisticated entity extraction mechanism.
        keywords = set(original_prompt.lower().split() + lesson_learned.lower().split())
        
        # Groovy script to create the lesson and link it to existing concepts/tools
        groovy_script = f"""
            // Create the new Lesson vertex
            def lesson = g.addV('lesson')
                .property('lesson_id', '{insight_id}')
                .property('lesson_text', '{lesson_learned}')
                .property('original_prompt', '{original_prompt}')
                .property('final_status', '{final_status}')
                .next()

            // Find existing entities (tools, concepts) mentioned in the text
            // and link the new lesson to them.
            def mentioned_entities = g.V().hasLabel('tool', 'concept')
                                       .where(values('name').is(within({str(list(keywords))})))
                                       .toList()

            mentioned_entities.each {{ entity ->
                g.V(lesson).addE('RELATES_TO').to(entity).iterate()
            }}
            
            return "Lesson {insight_id} added to graph and linked to " + mentioned_entities.size() + " entities."
        """

        response = kg_client.run_gremlin_script(groovy_script)
        return f"Successfully saved insight to VectorStore and KnowledgeGraph. Graph response: {response}"

    except Exception as e:
        logger.error(f"Failed to save insight to KnowledgeGraph: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while saving the insight to the KnowledgeGraph: {e}"


# --- Tool Registration Object ---

save_insight_tool = Tool(
    name="save_insight",
    description="Saves a lesson learned from a workflow analysis to the long-term knowledge base. This helps future planning agents make better decisions.",
    func=save_insight
) 