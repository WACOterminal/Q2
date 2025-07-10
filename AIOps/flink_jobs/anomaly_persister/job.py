from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.table.api.config import Configuration

def anomaly_persister_job():
    """
    A Flink job that ingests anomaly events from a Pulsar topic
    and writes them to an Elasticsearch index.
    """
    settings = EnvironmentSettings.new_instance().in_streaming_mode().build()
    t_env = StreamTableEnvironment.create(environment_settings=settings)

    # Add connectors to classpath
    jars = [
        "file:///opt/flink/connectors/flink-sql-connector-pulsar-1.15.4.jar",
        "file:///opt/flink/connectors/flink-sql-connector-elasticsearch7-1.15.4.jar",
        "file:///opt/flink/connectors/flink-avro-1.15.4.jar"
    ]
    t_env.get_config().get_configuration().set_string("pipeline.jars", ";".join(jars))

    # Define the source table (Pulsar)
    t_env.execute_sql("""
        CREATE TABLE pulsar_anomalies (
            `event_id` STRING,
            `service_name` STRING,
            `metric_name` STRING,
            `anomalous_value` DOUBLE,
            `expected_value` DOUBLE,
            `severity` STRING,
            `timestamp` TIMESTAMP(3)
        ) WITH (
            'connector' = 'pulsar',
            'topic' = 'persistent://public/default/platform-anomalies',
            'value.format' = 'avro',
            'service-url' = 'pulsar://pulsar-broker.q-platform.svc.cluster.local:6650',
            'admin-url' = 'http://pulsar-broker.q-platform.svc.cluster.local:8080'
        )
    """)

    # Define the sink table (Elasticsearch)
    t_env.execute_sql("""
        CREATE TABLE elasticsearch_anomalies (
            `event_id` STRING,
            `service_name` STRING,
            `metric_name` STRING,
            `anomalous_value` DOUBLE,
            `expected_value` DOUBLE,
            `severity` STRING,
            `@timestamp` TIMESTAMP(3),
            PRIMARY KEY (`event_id`) NOT ENFORCED
        ) WITH (
            'connector' = 'elasticsearch-7',
            'hosts' = 'http://elasticsearch.database.svc.cluster.local:9200',
            'index' = 'platform-anomalies-{now():yyyy-MM-dd}'
        )
    """)

    # Create and execute the insert statement
    t_env.from_path("pulsar_anomalies").execute_insert("elasticsearch_anomalies").wait()

if __name__ == "__main__":
    anomaly_persister_job() 