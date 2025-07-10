
import os
import json
import logging
from pyflink.common import SimpleStringSchema, WatermarkStrategy
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.pulsar import PulsarSource, PulsarSourceBuilder, SubscriptionType
from pyflink.datastream.functions import RuntimeContext, MapFunction
import httpx
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

# --- Configuration ---
LOG = logging.getLogger(__name__)
PULSAR_SERVICE_URL = os.getenv("PULSAR_SERVICE_URL", "pulsar://pulsar:6650")
PULSAR_ADMIN_URL = os.getenv("PULSAR_ADMIN_URL", "http://pulsar:8080")
PULSAR_TOPIC = os.getenv("PULSAR_TOPIC", "persistent://public/default/q.ingestion.unstructured")
VECTORSTORE_URL = os.getenv("VECTORSTORE_URL", "http://vectorstore-q:8001")
COLLECTION_NAME = "rag_document_chunks" # Or could be dynamic based on event
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Flink Sink & Transformation Logic ---

class DocumentEmbeddingProcessor(MapFunction):
    """
    A MapFunction that receives a document, splits it, embeds it, and upserts it to VectorStoreQ.
    """
    def __init__(self, batch_size=20):
        self.batch_size = batch_size
        self.vectors_to_upsert = []
        self.embedding_model = None
        self.text_splitter = None

    def open(self, runtime_context: RuntimeContext):
        """Load models and clients on task manager startup."""
        LOG.info("Initializing DocumentEmbeddingProcessor...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.client = httpx.Client(base_url=VECTORSTORE_URL, timeout=30.0)
        LOG.info("DocumentEmbeddingProcessor initialized successfully.")

    def close(self):
        """Flush any remaining items and close clients on shutdown."""
        self.flush()
        if self.client:
            self.client.close()

    def map(self, raw_json: str):
        """Process one document event."""
        try:
            event = json.loads(raw_json)
            doc_content = event.get("content")
            doc_metadata = event.get("metadata", {})
            doc_source = doc_metadata.get("source_uri", "unknown")

            if not doc_content:
                LOG.warning("Received event with no content.")
                return

            chunks = self.text_splitter.split_text(doc_content)
            LOG.info(f"Split document from '{doc_source}' into {len(chunks)} chunks.")

            if not chunks:
                return
            
            embeddings = self.embedding_model.encode(chunks)

            for i, chunk_text in enumerate(chunks):
                vector_data = {
                    "id": str(uuid.uuid4()),
                    "values": embeddings[i].tolist(),
                    "metadata": {
                        "text_chunk": chunk_text,
                        "source_uri": doc_source,
                        **doc_metadata # Pass through original metadata
                    }
                }
                self.vectors_to_upsert.append(vector_data)

                if len(self.vectors_to_upsert) >= self.batch_size:
                    self.flush()
        
        except Exception as e:
            LOG.error(f"Error processing document: {e}", exc_info=True)
        
        return raw_json # Pass through the original event

    def flush(self):
        """Sends the current batch of vectors to the VectorStoreQ API."""
        if not self.vectors_to_upsert:
            return

        LOG.info(f"Upserting batch of {len(self.vectors_to_upsert)} vectors to VectorStoreQ.")
        try:
            # Note: The vectorstore client library would be better here, but for simplicity
            # in a Flink job, a direct HTTP call avoids dependency complexities.
            response = self.client.post(
                f"/v1/ingest/upsert",
                json={
                    "collection_name": COLLECTION_NAME,
                    "vectors": self.vectors_to_upsert
                }
            )
            response.raise_for_status()
            LOG.info(f"Successfully upserted batch. Status: {response.status_code}")
        except httpx.HTTPError as e:
            LOG.error(f"Failed to send batch to VectorStoreQ: {e}", exc_info=True)
            # In a production system, add retries or a dead-letter queue here.
        finally:
            self.vectors_to_upsert.clear()

# --- Flink Job Definition ---

def run_job():
    env = StreamExecutionEnvironment.get_execution_environment()
    
    pulsar_source = PulsarSource.builder() \
        .set_service_url(PULSAR_SERVICE_URL) \
        .set_admin_url(PULSAR_ADMIN_URL) \
        .set_start_cursor_from_earliest() \
        .set_topics(PULSAR_TOPIC) \
        .set_deserialization_schema(SimpleStringSchema()) \
        .set_subscription_name("flink-unstructured-processor-sub") \
        .set_subscription_type(SubscriptionType.Shared) \
        .build()

    stream = env.from_source(
        pulsar_source,
        WatermarkStrategy.for_monotonous_timestamps(),
        "PulsarUnstructuredDataSource"
    )

    stream.map(DocumentEmbeddingProcessor()).name("ChunkEmbedAndStore")

    env.execute("RealtimeUnstructuredDocumentProcessor")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_job() 