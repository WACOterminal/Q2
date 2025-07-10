
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
import json

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'daily_meta_analysis',
    default_args=default_args,
    description='A DAG to periodically trigger the meta-analysis goal to find inefficient workflows.',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    trigger_meta_analysis_goal = SimpleHttpOperator(
        task_id='trigger_meta_analysis_goal',
        http_conn_id='managerq_api', # This connection must be configured in Airflow UI
        endpoint='/api/v1/goals',
        method='POST',
        data=json.dumps({
            "prompt": "Analyze all completed workflows from the past week and identify any with a high rate of failure or an excessive number of loops. Propose a workflow improvement.",
        }),
        headers={"Content-Type": "application/json"},
        log_response=True,
    ) 