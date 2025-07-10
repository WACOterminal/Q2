import os
from pyflink.table import StreamTableEnvironment, EnvironmentSettings, Table, DataTypes
from pyflink.table.window import Tumble
from pyflink.table.expressions import lit, col
from pyflink.table.udf import udtf

def anomaly_detection_job():
    """
    A Flink job that reads metrics from Prometheus, detects anomalies,
    and writes them to a Pulsar topic.
    """
    settings = EnvironmentSettings.new_instance().in_streaming_mode().build()
    t_env = StreamTableEnvironment.create(environment_settings=settings)

    # Add connectors to classpath
    # In a real deployment, these JARs would be included in the Docker image.
    jars = [
        "file:///opt/flink/connectors/flink-sql-connector-prometheus-1.15.4.jar",
        "file:///opt/flink/connectors/flink-sql-connector-pulsar-1.15.4.jar",
        "file:///opt/flink/connectors/flink-avro-1.15.4.jar"
    ]
    t_env.get_config().get_configuration().set_string("pipeline.jars", ";".join(jars))

    # Define the source table (Prometheus)
    prometheus_server_endpoint = os.environ.get("PROMETHEUS_ENDPOINT", "http://prometheus-server.q-platform.svc.cluster.local:9090")
    t_env.execute_sql(f"""
        CREATE TABLE prometheus_metrics (
            `name` STRING,
            `labels` MAP<STRING, STRING>,
            `value` DOUBLE,
            `timestamp` TIMESTAMP(3)
        ) WITH (
            'connector' = 'prometheus',
            'endpoint' = '{prometheus_server_endpoint}',
            'query' = '{{__name__=\\"starlette_request_duration_seconds_bucket\\"}}',
            'scrape.interval' = '15s'
        )
    """)

    # Define the sink table (Pulsar)
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

    metrics_table = t_env.from_path("prometheus_metrics")

    # Simple anomaly detection logic
    # Calculate moving average over a 5-minute window
    windowed_avg = metrics_table.window(Tumble.over(lit(5).minutes).on(col("timestamp")).alias("w")) \
        .group_by(col("w"), col("labels")['app_name']) \
        .select(
            col("labels")['app_name'].alias("service_name"),
            col("value").avg.alias("avg_value"),
            col("w").end.alias("window_end")
        )

    # Join original stream with moving average
    joined_stream = metrics_table.join(windowed_avg, col("labels")['app_name'] == col("service_name"))

    # Detect anomalies
    @udtf(result_types=[DataTypes.STRING(), DataTypes.STRING(), DataTypes.STRING(), DataTypes.DOUBLE(), DataTypes.DOUBLE(), DataTypes.STRING(), DataTypes.TIMESTAMP(3)])
    def detect_anomaly(service, metric, current_value, avg_value, ts):
        if current_value > avg_value * 2: # Anomaly if value is > 2x the moving average
            yield "event_id_placeholder", service, metric, current_value, avg_value, "CRITICAL", ts

    anomalies = joined_stream.flat_map(detect_anomaly(col("service_name"), col("name"), col("value"), col("avg_value"), col("timestamp"))) \
                             .select(
                                 col("f0").alias("event_id"),
                                 col("f1").alias("service_name"),
                                 col("f2").alias("metric_name"),
                                 col("f3").alias("anomalous_value"),
                                 col("f4").alias("expected_value"),
                                 col("f5").alias("severity"),
                                 col("f6").alias("timestamp")
                             )

    anomalies.execute_insert("pulsar_anomalies").wait()

if __name__ == "__main__":
    anomaly_detection_job() 