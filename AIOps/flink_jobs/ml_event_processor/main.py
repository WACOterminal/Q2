"""
Flink ML Event Processor Job

This Flink job consumes ML events from Apache Pulsar, processes them,
and writes them to Elasticsearch for real-time monitoring and analytics.
"""

import os
import json
import logging
import yaml
from datetime import datetime

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.pulsar import PulsarSource, StartCursor
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.functions import MapFunction

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from avro import schema, io
import io as avro_io

logger = logging.getLogger(__name__)

class DeserializeMLEvent(MapFunction):
    """
    Deserializes Avro-encoded ML event messages from Pulsar.
    """
    def __init__(self, avro_schema_str):
        self.avro_schema = schema.Parse(avro_schema_str)
        self.reader = io.DatumReader(self.avro_schema)

    def map(self, value):
        bytes_reader = avro_io.BytesIO(value)
        decoder = io.BinaryDecoder(bytes_reader)
        event = self.reader.read(decoder)
        return event

class PrepareForElasticsearch(MapFunction):
    """
    Prepares ML event data for indexing in Elasticsearch.
    """
    def __init__(self, index_prefix="ml_events_"):
        self.index_prefix = index_prefix

    def map(self, event):
        # Create a dynamic index name based on month
        index_name = f"{self.index_prefix}{datetime.fromtimestamp(event['timestamp'] / 1000).strftime('%Y-%m')}"
        
        # Prepare document for Elasticsearch
        doc = {
            "_index": index_name,
            "_id": event['event_id'],
            "_source": event
        }
        return doc

class ElasticsearchSink:
    """
    A simple sink to write data to Elasticsearch.
    """
    def __init__(self, es_hosts, es_user, es_password):
        self.es_client = Elasticsearch(
            es_hosts,
            basic_auth=(es_user, es_password),
            verify_certs=False  # Consider True for production with proper CAs
        )

    def write(self, records):
        # records is an iterable of documents prepared by PrepareForElasticsearch
        try:
            success, failed = bulk(self.es_client, records, raise_on_error=False)
            if failed:
                logger.warning(f"Failed to index {len(failed)} documents: {failed}")
            if success:
                logger.info(f"Successfully indexed {success} documents to Elasticsearch.")
        except Exception as e:
            logger.error(f"Error during Elasticsearch bulk insert: {e}", exc_info=True)

def main():
    # Load configuration from environment variables or a config file
    pulsar_service_url = os.environ.get("PULSAR_SERVICE_URL", "pulsar://localhost:6650")
    pulsar_topic = os.environ.get("PULSAR_ML_EVENTS_TOPIC", "persistent://public/default/q.ml.events")
    elasticsearch_hosts = os.environ.get("ELASTICSEARCH_HOSTS", "http://localhost:9200").split(',')
    elasticsearch_user = os.environ.get("ELASTICSEARCH_USER", "elastic")
    elasticsearch_password = os.environ.get("ELASTICSEARCH_PASSWORD", "changeme")
    
    # Load Avro schema
    script_dir = os.path.dirname(os.path.realpath(__file__))
    schema_path = os.path.join(script_dir, "..", "..", "schemas", "ml_event.avsc")
    with open(schema_path, 'r') as f:
        ml_event_avro_schema = f.read()

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1) # Adjust parallelism as needed

    # Configure Pulsar source
    pulsar_source = (
        PulsarSource.builder()
        .set_service_url(pulsar_service_url)
        .set_topic(pulsar_topic)
        .set_starting_offsets(StartCursor.latest())
        .set_subscription_name("ml-event-processor-sub")
        .set_subscription_type("Shared") # Or Exclusive, Failover
        .set_deserialization_schema(SimpleStringSchema()) # Raw bytes as string
        .build()
    )

    # Create a DataStream from Pulsar
    data_stream = env.from_source(pulsar_source, "Pulsar ML Events Source", Types.STRING())

    # Deserialize Avro bytes to Python dict
    deserialized_stream = data_stream.map(DeserializeMLEvent(ml_event_avro_schema), Types.MAP(Types.STRING(), Types.GENERIC_ARRAY()))

    # Prepare for Elasticsearch indexing
    es_ready_stream = deserialized_stream.map(PrepareForElasticsearch(), Types.MAP(Types.STRING(), Types.OBJECT()))
    
    # Sink to Elasticsearch (using a collection-based sink for simplicity; Flink has built-in ES sinks for production)
    # For real production, you'd use FlinkKafkaConnector with ES sink, or Flink's native ES sink.
    # This is a simplified custom sink for demonstration.
    es_ready_stream.add_sink(lambda records: ElasticsearchSink(elasticsearch_hosts, elasticsearch_user, elasticsearch_password).write([records]))

    logger.info("Starting Flink ML Event Processor Job...")
    env.execute("ml_event_processor")

if __name__ == '__main__':
    # Basic logging setup for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main() 