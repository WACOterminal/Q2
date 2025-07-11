from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import requests
import json
import logging
import time

logger = logging.getLogger(__name__)

def trigger_automl_experiment(experiment_name, model_type, dataset_config, optimization_objective="accuracy", n_trials=50, timeout_hours=2, manager_url="http://managerq:8000"):
    """
    Triggers an AutoML experiment in the managerQ service.
    """
    ml_api_url = f"{manager_url}/v1/ml/automl/start"
    payload = {
        "experiment_name": experiment_name,
        "model_type": model_type,
        "dataset_config": dataset_config,
        "optimization_objective": optimization_objective,
        "n_trials": n_trials,
        "timeout_hours": timeout_hours
    }
    
    try:
        response = requests.post(ml_api_url, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors
        result = response.json()
        logger.info(f"AutoML experiment triggered: {result}")
        return result["experiment_id"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error triggering AutoML experiment: {e}")
        raise

def monitor_automl_experiment(experiment_id, manager_url="http://managerq:8000", timeout_seconds=3600, check_interval_seconds=30):
    """
    Monitors the status of an AutoML experiment until completion or timeout.
    """
    ml_api_url = f"{manager_url}/v1/ml/automl/status/{experiment_id}"
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        try:
            response = requests.get(ml_api_url)
            response.raise_for_status()
            status = response.json()
            logger.info(f"AutoML Experiment {experiment_id} Status: {status.get('status')}, Progress: {status.get('trials_completed', 0)}/{status.get('total_trials', 0)}")

            if status.get('status') == 'completed':
                logger.info(f"AutoML Experiment {experiment_id} completed successfully.")
                return status
            elif status.get('status') == 'failed' or status.get('status') == 'cancelled':
                raise Exception(f"AutoML Experiment {experiment_id} failed or was cancelled.")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error monitoring AutoML experiment: {e}")
            raise

        time.sleep(check_interval_seconds)
    
    raise TimeoutError(f"AutoML Experiment {experiment_id} timed out after {timeout_seconds} seconds.")

def get_automl_results(experiment_id, manager_url="http://managerq:8000"):
    """
    Fetches the results of a completed AutoML experiment.
    """
    ml_api_url = f"{manager_url}/v1/ml/automl/results/{experiment_id}"
    try:
        response = requests.get(ml_api_url)
        response.raise_for_status()
        results = response.json()
        logger.info(f"AutoML Experiment {experiment_id} results fetched.")
        # In a real scenario, you'd process these results, evaluate the best model,
        # and potentially trigger model deployment.
        return results
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching AutoML results: {e}")
        raise

# Define the DAG
with DAG(
    dag_id='ml_model_retraining_pipeline',
    start_date=days_ago(1),
    schedule_interval=None, # Triggered manually or by external event
    catchup=False,
    tags=['mlops', 'automl', 'retraining'],
    params={
        "experiment_name": {"type": "string", "default": "scheduled_retraining", "title": "AutoML Experiment Name"},
        "model_type": {"type": "string", "default": "classification", "title": "Model Type (e.g., classification, regression)"},
        "dataset_config": {"type": "string", "default": "{\"dataset_path\":\"/data/training_data.csv\", \"target_column\":\"label\"}", "title": "Dataset Configuration (JSON string)"},
        "optimization_objective": {"type": "string", "default": "accuracy", "title": "Optimization Objective"},
        "n_trials": {"type": "integer", "default": 50, "title": "Number of AutoML Trials"}
    }
) as dag:
    
    trigger_task = PythonOperator(
        task_id='trigger_automl_experiment',
        python_callable=trigger_automl_experiment,
        op_kwargs={
            "experiment_name": "{{ dag_run.conf.get('experiment_name', params.experiment_name) }}",
            "model_type": "{{ dag_run.conf.get('model_type', params.model_type) }}",
            "dataset_config": json.loads("{{ dag_run.conf.get('dataset_config', params.dataset_config) }}"),
            "optimization_objective": "{{ dag_run.conf.get('optimization_objective', params.optimization_objective) }}",
            "n_trials": "{{ dag_run.conf.get('n_trials', params.n_trials) }}"
        },
    )

    monitor_task = PythonOperator(
        task_id='monitor_automl_experiment',
        python_callable=monitor_automl_experiment,
        op_kwargs={
            "experiment_id": "{{ task_instance.xcom_pull(task_ids='trigger_automl_experiment') }}"
        },
    )

    get_results_task = PythonOperator(
        task_id='get_automl_results',
        python_callable=get_automl_results,
        op_kwargs={
            "experiment_id": "{{ task_instance.xcom_pull(task_ids='trigger_automl_experiment') }}"
        },
    )

    # Define task dependencies
    trigger_task >> monitor_task >> get_results_task 