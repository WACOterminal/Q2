import structlog
import yaml
import httpx
import asyncio

from agentQ.app.core.toolbox import Toolbox, Tool
from agentQ.app.core.context import ContextManager
from agentQ.app.core.knowledgegraph_tool import knowledgegraph_tool
from agentQ.app.core.vectorstore_tool import vectorstore_tool
from agentQ.app.core.http_tool import http_get_tool

logger = structlog.get_logger("knowledge_engineer_agent")

# --- Agent Definition ---
AGENT_ID = "knowledge-engineer-agent-01"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"

KNOWLEDGE_ENGINEER_SYSTEM_PROMPT = """
You are a Knowledge Engineer AI. Your purpose is to enrich the platform's knowledge base by processing and ingesting information from various sources.

**Primary Workflows:**

1.  **Live Web Ingestion (Manual Trigger):**
    -   This workflow is triggered when a user provides a URL.
    -   You will be given a prompt containing the `source_url`.
    -   **Step 1 (Fetch):** Use the `http_get` tool to retrieve the content from the `source_url`.
    -   **Step 2 (Process & Chunk):** The result will be HTML. Extract the main textual content (paragraphs, headings) and split it into meaningful chunks.
    -   **Step 3 (Ingest):** For each chunk, use both `knowledgegraph_add_chunk` and `vectorstore_upsert` tools to save it to the knowledge graph and vector store respectively.

2.  **Batch Document Processing (Automated Task):**
    -   If not given a specific ingestion prompt, your default task is to process documents from the backlog.
    -   Use `get_unprocessed_documents` to fetch a batch.
    -   For each document, identify key entities and relationships.
    -   Use `upsert_vertex` and `upsert_edge` operations to add them to the knowledge graph.

Always prioritize the manually triggered workflow if a `source_url` is present in the context.
"""

# --- Tools for the Knowledge Engineer ---
# (get_unprocessed_documents tool definition remains the same)
def get_unprocessed_documents(config: dict) -> str:
    """
    Retrieves a batch of unprocessed documents from the vector store.
    (This is a placeholder for a more robust implementation).
    """
    # In a real system, we would have a mechanism to track which documents
    # have been processed. For now, we'll just grab the latest N chunks.
    vector_store_url = config.get("vector_store_url")
    if not vector_store_url:
        return "Error: vector_store_url not configured."
    
    # This is a simplified search query to get some recent documents.
    search_query = {
        "collection_name": "rag_document_chunks",
        "queries": [{"values": [0.0] * 384, "top_k": 10}] # Placeholder query
    }
    
    try:
        with httpx.Client() as client:
            response = client.post(f"{vector_store_url}/v1/search", json=search_query)
            response.raise_for_status()
        return str(response.json())
    except Exception as e:
        return f"Error fetching documents: {e}"

unprocessed_docs_tool = Tool(
    name="get_unprocessed_documents",
    description="Retrieves a batch of recently added documents that need to be processed and added to the Knowledge Graph.",
    func=get_unprocessed_documents
)


def setup_knowledge_engineer_agent(config: dict):
    """
    Initializes the toolbox and context manager for the Knowledge Engineer agent.
    """
    logger.info("Setting up Knowledge Engineer Agent", agent_id=AGENT_ID)
    
    toolbox = Toolbox()
    
    # Register all necessary tools
    toolbox.register_tool(unprocessed_docs_tool)
    toolbox.register_tool(knowledgegraph_tool)
    toolbox.register_tool(vectorstore_tool)
    toolbox.register_tool(http_get_tool)
    
    context_manager = ContextManager(ignite_addresses=config['ignite']['addresses'], agent_id=AGENT_ID)
    context_manager.connect()
    
    return toolbox, context_manager 