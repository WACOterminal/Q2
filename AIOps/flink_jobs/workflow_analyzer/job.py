# AIOps/flink_jobs/workflow_analyzer/job.py
import logging
import os

from pyflink.common import WatermarkStrategy, Time
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.window import TumblingProcessingTimeWindows
from pyflink.datastream.connectors.pulsar import PulsarSource, PulsarSourceBuilder, SubscriptionType
from pyflink.datastream.connectors.elasticsearch import ElasticsearchSink, Elasticsearch7SinkBuilder
from pyflink.common.typeinfo import Types
from org.apache.flink.connector.elasticsearch.sink import FlushOnCheckpointListener

# Import the shared schema classes
from shared.q_messaging_schemas.schemas import ResultMessage, LogMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def define_workflow(env: StreamExecutionEnvironment):
    """
    Defines the Flink data pipeline for analyzing workflow events.
    """
    pulsar_service_url = os.getenv("PULSAR_SERVICE_URL", "pulsar://localhost:6650")
    results_topic = os.getenv("PULSAR_RESULTS_TOPIC", "persistent://public/default/q.agentq.results")
    logs_topic = os.getenv("PULSAR_LOGS_TOPIC", "persistent://public/default/q.agentq.logs")
    es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
    es_port = int(os.getenv("ELASTICSEARCH_PORT", "9200"))

    # 1. Pulsar Source for Results
    results_source = PulsarSource.builder() \
        .set_service_url(pulsar_service_url) \
        .set_admin_url(pulsar_service_url.replace("pulsar://", "http://").replace("6650", "8080")) \
        .set_start_cursor_message_id_initial() \
        .set_subscription_name("flink-workflow-analyzer-results") \
        .set_subscription_type(SubscriptionType.SHARED) \
        .set_topics(results_topic) \
        .set_value_schema(ResultMessage) \
        .build()

    # 2. Pulsar Source for Logs (to catch failures)
    logs_source = PulsarSource.builder() \
        .set_service_url(pulsar_service_url) \
        .set_admin_url(pulsar_service_url.replace("pulsar://", "http://").replace("6650", "8080")) \
        .set_start_cursor_message_id_initial() \
        .set_subscription_name("flink-workflow-analyzer-logs") \
        .set_subscription_type(SubscriptionType.SHARED) \
        .set_topics(logs_topic) \
        .set_value_schema(LogMessage) \
        .build()
        
    # 3. Create DataStreams
    results_stream = env.from_source(results_source, WatermarkStrategy.for_monotonous_timestamps(), "PulsarResultsSource")
    logs_stream = env.from_source(logs_source, WatermarkStrategy.for_monotonous_timestamps(), "PulsarLogsSource")

    # 4. Process and Union Streams
    # We only care about failure logs for this analysis
    failure_stream = logs_stream \
        .filter(lambda log: log.level.upper() == 'ERROR' and log.agent_id is not None) \
        .map(lambda log: (log.agent_id, 0, 1, 0), output_type=Types.TUPLE([Types.STRING(), Types.INT(), Types.INT(), Types.INT()]))

    # Extract relevant fields from results stream
    processed_results_stream = results_stream \
        .filter(lambda res: res.agent_personality is not None) \
        .map(lambda res: (res.agent_personality, 1, 0, res.timestamp), output_type=Types.TUPLE([Types.STRING(), Types.INT(), Types.INT(), Types.LONG()]))

    # For now, we will just analyze successes and failures. Duration can be added later.
    analytics_stream = failure_stream.union(processed_results_stream)

    # 5. Key by agent_id and apply a tumbling window
    windowed_stream = analytics_stream \
        .key_by(lambda x: x[0]) \
        .window(TumblingProcessingTimeWindows.of(Time.minutes(1)))

    # 6. Aggregate metrics within the window
    aggregated_stream = windowed_stream.reduce(
        lambda a, b: (a[0], a[1] + b[1], a[2] + b[2], 0), # (agent_id, success_count, failure_count, placeholder)
        output_type=Types.TUPLE([Types.STRING(), Types.INT(), Types.INT(), Types.INT()])
    )

    # 7. Sink to Elasticsearch
    es_sink = Elasticsearch7SinkBuilder() \
        .set_hosts(es_host, es_port, "http") \
        .set_emitter(
            lambda context, element, writer: writer.add_index_request(
                index="workflow_analytics",
                source={
                    "agent_id": element[0],
                    "success_count": element[1],
                    "failure_count": element[2],
                    "timestamp": context.current_processing_time()
                }
            )
        ) \
        .set_bulk_flush_max_actions(1) \
        .set_connection_request_timeout(30000) \
        .build()

    aggregated_stream.sink_to(es_sink).name("ElasticsearchAnalyticsSink")


def run():
    """
    Entry point for running the Flink job.
    """
    env = StreamExecutionEnvironment.get_execution_environment()
    define_workflow(env)
    env.execute("Workflow_Analytics_Job")

if __name__ == "__main__":
    run() 