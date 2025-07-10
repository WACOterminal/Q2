from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.table.api.config import Configuration

def log_ingestion_job():
    """
    A Flink job that ingests structured logs from a Pulsar topic
    and writes them to an Elasticsearch index.
    """
    # Set up the stream table environment
    settings = EnvironmentSettings.new_instance().in_streaming_mode().build()
    t_env = StreamTableEnvironment.create(environment_settings=settings)
    
    # Add the Pulsar and Elasticsearch connectors to the classpath
    # In a real deployment, these JARs would be included in the Docker image.
    config = Configuration()
    config.set_string("pipeline.jars", "file:///opt/flink/connectors/flink-sql-connector-pulsar-1.15.4.jar;file:///opt/flink/connectors/flink-sql-connector-elasticsearch7-1.15.4.jar")
    t_env.get_config().add_configuration(config)

    # Define the source table (Pulsar)
    t_env.execute_sql("""
        CREATE TABLE pulsar_logs (
            `log_level` STRING,
            `logger_name` STRING,
            `otel_service_name` STRING,
            `otel_trace_id` STRING,
            `otel_span_id` STRING,
            `timestamp` TIMESTAMP(3),
            `message` STRING,
            `stack_trace` STRING,
            `exception` STRING
        ) WITH (
            'connector' = 'pulsar',
            'topic' = 'persistent://public/default/platform-logs',
            'value.format' = 'json',
            'service-url' = 'pulsar://pulsar-broker.q-platform.svc.cluster.local:6650',
            'admin-url' = 'http://pulsar-broker.q-platform.svc.cluster.local:8080'
        )
    """)

    # Define the sink table (Elasticsearch)
    t_env.execute_sql("""
        CREATE TABLE elasticsearch_logs (
            `log_level` STRING,
            `logger_name` STRING,
            `otel_service_name` STRING,
            `otel_trace_id` STRING,
            `otel_span_id` STRING,
            `@timestamp` TIMESTAMP(3),
            `message` STRING,
            `stack_trace` STRING,
            `exception` STRING
        ) WITH (
            'connector' = 'elasticsearch-7',
            'hosts' = 'http://elasticsearch.database.svc.cluster.local:9200',
            'index' = 'platform-logs-{otel_service_name}-{now():yyyy-MM-dd}'
        )
    """)

    # Create and execute the insert statement
    t_env.from_path("pulsar_logs").execute_insert("elasticsearch_logs").wait()

if __name__ == "__main__":
    log_ingestion_job() 