import logging
import os
import uuid

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.pulsar import PulsarSource, PulsarSink
from pyflink.common import WatermarkStrategy, SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.formats.avro import AvroRowSerializationSchema, AvroRowDeserializationSchema
from pyflink.datastream.functions import MapFunction
from pyflink.table.types import Row

# Schema paths
SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "schemas")
PERSONA_PERFORMANCE_SCHEMA_PATH = os.path.join(SCHEMA_DIR, "persona_performance_metrics.avsc")
ANOMALY_EVENT_SCHEMA_PATH = os.path.join(SCHEMA_DIR, "anomaly_event.avsc")

# Pulsar configuration
PULSAR_SERVICE_URL = os.getenv("PULSAR_SERVICE_URL", "pulsar://localhost:6650")
INPUT_TOPIC = "q.agents.persona.performance.updated"
OUTPUT_TOPIC = "q.aiops.agent.performance.anomaly.detected"

# Anomaly detection thresholds (simplified for demonstration)
TASK_COMPLETION_THRESHOLD = 0.5 # If task_completion_rate drops below this, it's a warning
USER_SATISFACTION_THRESHOLD = 0.6 # If user_satisfaction drops below this, it's a warning

class AnomalyDetector(MapFunction):
    def open(self, runtime_context):
        self.LOG = logging.getLogger(self.__class__.__name__)

    def map(self, value):
        # Assuming value is a Flink Row corresponding to PersonaPerformanceMetrics
        persona_id = value['persona_id']
        agent_id = value['agent_id']
        task_completion_rate = value['task_completion_rate']
        user_satisfaction = value['user_satisfaction']
        computed_at = value['computed_at'] # Timestamp in milliseconds

        severity = "NORMAL"
        description = []

        if task_completion_rate < TASK_COMPLETION_THRESHOLD:
            severity = "WARNING"
            description.append(f"Task completion rate ({task_completion_rate}) is below threshold ({TASK_COMPLETION_THRESHOLD}).")
        
        if user_satisfaction < USER_SATISFACTION_THRESHOLD:
            severity = "WARNING"
            description.append(f"User satisfaction ({user_satisfaction}) is below threshold ({USER_SATISFACTION_THRESHOLD}).")

        if severity != "NORMAL":
            # Create an AnomalyEvent
            anomaly_event = {
                "event_id": str(uuid.uuid4()),
                "service_name": "AdaptivePersonaService",
                "metric_name": "agent_performance",
                "anomalous_value": task_completion_rate, # Could be more specific based on which metric triggered
                "expected_value": (TASK_COMPLETION_THRESHOLD + USER_SATISFACTION_THRESHOLD) / 2, # Placeholder
                "severity": severity,
                "timestamp": computed_at
            }
            self.LOG.warning(f"Anomaly detected for persona {persona_id}, agent {agent_id}: {'. '.join(description)}")
            return Row(anomaly_event['event_id'], anomaly_event['service_name'], anomaly_event['metric_name'],
                       anomaly_event['anomalous_value'], anomaly_event['expected_value'], anomaly_event['severity'],
                       anomaly_event['timestamp'])
        else:
            return None # No anomaly, so filter out

def define_anomaly_detection_job():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.add_jars("file:///path/to/flink-sql-connector-pulsar-1.17.0.jar", "file:///path/to/flink-avro-1.17.0.jar") # Placeholder, replace with actual paths
    env.set_parallelism(1) # For simplicity

    # Load Avro schemas
    with open(PERSONA_PERFORMANCE_SCHEMA_PATH, 'r') as f:
        persona_performance_schema = f.read()
    with open(ANOMALY_EVENT_SCHEMA_PATH, 'r') as f:
        anomaly_event_schema = f.read()

    # Source: Persona Performance Metrics from Pulsar
    pulsar_source = (
        PulsarSource.builder()
        .set_service_url(PULSAR_SERVICE_URL)
        .set_admin_url(PULSAR_SERVICE_URL.replace("pulsar:", "http:").replace("6650", "8080")) # Admin URL for schema
        .set_topic(INPUT_TOPIC)
        .set_starting_offsets("earliest")
        .set_value_deserialization_schema(AvroRowDeserializationSchema.for_spec_and_class(persona_performance_schema, Types.ROW_NAMED(["persona_id", "agent_id", "context_type", "task_completion_rate", "user_satisfaction", "response_quality", "collaboration_effectiveness", "learning_progress", "adaptability_score", "consistency_score", "innovation_score", "sample_size", "computed_at"], [Types.STRING(), Types.STRING(), Types.STRING(), Types.DOUBLE(), Types.DOUBLE(), Types.DOUBLE(), Types.DOUBLE(), Types.DOUBLE(), Types.DOUBLE(), Types.DOUBLE(), Types.DOUBLE(), Types.INT(), Types.SQL_TIMESTAMP()]))) # Need to manually define row type based on Avro schema
        .build()
    )

    # Sink: Anomaly Events to Pulsar
    pulsar_sink = (
        PulsarSink.builder()
        .set_service_url(PULSAR_SERVICE_URL)
        .set_admin_url(PULSAR_SERVICE_URL.replace("pulsar:", "http:").replace("6650", "8080")) # Admin URL for schema
        .set_topic(OUTPUT_TOPIC)
        .set_value_serialization_schema(AvroRowSerializationSchema.for_spec_and_class(anomaly_event_schema, Types.ROW_NAMED(["event_id", "service_name", "metric_name", "anomalous_value", "expected_value", "severity", "timestamp"], [Types.STRING(), Types.STRING(), Types.STRING(), Types.DOUBLE(), Types.DOUBLE(), Types.STRING(), Types.SQL_TIMESTAMP()]))) # Need to manually define row type based on Avro schema
        .build()
    )

    # Data Stream
    data_stream = env.from_source(pulsar_source)

    # Apply anomaly detection logic
    anomaly_stream = data_stream.map(AnomalyDetector(), output_type=Types.ROW_NAMED(["event_id", "service_name", "metric_name", "anomalous_value", "expected_value", "severity", "timestamp"], [Types.STRING(), Types.STRING(), Types.STRING(), Types.DOUBLE(), Types.DOUBLE(), Types.STRING(), Types.SQL_TIMESTAMP()])) \
                               .filter(lambda x: x is not None)

    # Add sink
    anomaly_stream.sink_to(pulsar_sink)

    # Execute job
    env.execute("AgentPerformanceAnomalyDetection")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    define_anomaly_detection_job() 