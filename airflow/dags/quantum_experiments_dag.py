"""
Quantum Computing Experiments DAG

This DAG automates quantum computing experiments:
- Daily quantum optimization benchmarks
- Weekly quantum ML experiments  
- Quantum advantage evaluation
- Performance monitoring and reporting
- Integration with Q Platform services
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import asyncio
import logging
import json
import numpy as np

# Q Platform imports (would be available in Airflow environment)
import sys
import os
sys.path.append('/opt/airflow/dags/repo/agentQ')

logger = logging.getLogger(__name__)

# DAG Configuration
default_args = {
    'owner': 'q-platform',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email': ['admin@q-platform.com']
}

dag = DAG(
    'quantum_experiments',
    default_args=default_args,
    description='Automated quantum computing experiments',
    schedule_interval=timedelta(days=1),  # Daily execution
    catchup=False,
    tags=['quantum', 'experiments', 'optimization', 'ml']
)

# ===== QUANTUM OPTIMIZATION EXPERIMENTS =====

def run_quantum_optimization_benchmark(**context):
    """Run quantum optimization benchmark experiments"""
    logger.info("Starting quantum optimization benchmark")
    
    try:
        # Import services (async context)
        from app.services.quantum_optimization_service import (
            quantum_optimization_service, OptimizationProblem, QuantumAlgorithm, QuantumBackend
        )
        
        async def benchmark_experiments():
            await quantum_optimization_service.initialize()
            
            # Benchmark problems
            benchmark_problems = [
                {
                    "name": "Max-Cut Small",
                    "problem_type": OptimizationProblem.MAX_CUT,
                    "problem_data": {
                        "num_vertices": 6,
                        "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
                    },
                    "algorithm": QuantumAlgorithm.QAOA,
                    "num_qubits": 6,
                    "max_iterations": 50
                },
                {
                    "name": "TSP Small",
                    "problem_type": OptimizationProblem.TRAVELING_SALESMAN,
                    "problem_data": {"num_cities": 4},
                    "algorithm": QuantumAlgorithm.QAOA,
                    "num_qubits": 4,
                    "max_iterations": 100
                },
                {
                    "name": "Portfolio Optimization",
                    "problem_type": OptimizationProblem.PORTFOLIO_OPTIMIZATION,
                    "problem_data": {"num_assets": 5},
                    "algorithm": QuantumAlgorithm.VQE,
                    "num_qubits": 5,
                    "max_iterations": 75
                }
            ]
            
            results = []
            
            for problem in benchmark_problems:
                logger.info(f"Running benchmark: {problem['name']}")
                
                task_id = await quantum_optimization_service.solve_optimization_problem(
                    problem_type=problem["problem_type"],
                    problem_data=problem["problem_data"],
                    algorithm=problem["algorithm"],
                    backend=QuantumBackend.SIMULATOR,
                    num_qubits=problem["num_qubits"],
                    max_iterations=problem["max_iterations"]
                )
                
                # Wait for completion (simplified)
                await asyncio.sleep(30)  # Mock wait time
                
                result = await quantum_optimization_service.get_optimization_result(task_id)
                
                if result:
                    results.append({
                        "problem_name": problem["name"],
                        "task_id": task_id,
                        "algorithm": problem["algorithm"].value,
                        "quantum_advantage": result.get("quantum_advantage", 1.0),
                        "execution_time": result.get("execution_time", 0.0),
                        "objective_value": result.get("objective_value", 0.0),
                        "convergence_iterations": len(result.get("convergence_history", []))
                    })
            
            # Store results for downstream tasks
            context['task_instance'].xcom_push(key='benchmark_results', value=results)
            
            logger.info(f"Quantum optimization benchmark completed: {len(results)} problems solved")
            return results
        
        # Run async function
        return asyncio.run(benchmark_experiments())
        
    except Exception as e:
        logger.error(f"Quantum optimization benchmark failed: {e}")
        raise

def run_quantum_ml_experiments(**context):
    """Run quantum machine learning experiments"""
    logger.info("Starting quantum ML experiments")
    
    try:
        from app.services.quantum_ml_experiments import (
            quantum_ml_experiments, QuantumMLAlgorithm, MLTaskType
        )
        
        async def ml_experiments():
            await quantum_ml_experiments.initialize()
            
            # Create sample datasets
            np.random.seed(42)
            
            # Binary classification dataset
            features_binary = np.random.randn(100, 4)
            labels_binary = (features_binary[:, 0] + features_binary[:, 1] > 0).astype(int)
            
            binary_dataset_id = await quantum_ml_experiments.create_dataset(
                name="Daily Binary Classification",
                features=features_binary,
                labels=labels_binary,
                task_type=MLTaskType.BINARY_CLASSIFICATION
            )
            
            # Quantum ML experiments
            experiments = [
                {
                    "name": "Daily QNN Binary Classification",
                    "algorithm": QuantumMLAlgorithm.QUANTUM_NEURAL_NETWORK,
                    "dataset_id": binary_dataset_id,
                    "description": "Daily quantum neural network experiment"
                },
                {
                    "name": "Daily QSVM Binary Classification", 
                    "algorithm": QuantumMLAlgorithm.QUANTUM_SVM,
                    "dataset_id": binary_dataset_id,
                    "description": "Daily quantum SVM experiment"
                }
            ]
            
            results = []
            
            for exp in experiments:
                logger.info(f"Running ML experiment: {exp['name']}")
                
                experiment_id = await quantum_ml_experiments.create_ml_experiment(
                    name=exp["name"],
                    algorithm=exp["algorithm"],
                    dataset_id=exp["dataset_id"],
                    description=exp["description"]
                )
                
                # Wait for completion
                await asyncio.sleep(60)  # Mock wait time
                
                result = await quantum_ml_experiments.get_experiment_results(experiment_id)
                
                if result:
                    results.append({
                        "experiment_name": exp["name"],
                        "experiment_id": experiment_id,
                        "algorithm": exp["algorithm"].value,
                        "accuracy": result.get("best_model_details", {}).get("accuracy", 0.0),
                        "quantum_advantage": result.get("quantum_advantage", 1.0),
                        "models_trained": result.get("models_trained", 0)
                    })
            
            context['task_instance'].xcom_push(key='ml_results', value=results)
            
            logger.info(f"Quantum ML experiments completed: {len(results)} experiments")
            return results
        
        return asyncio.run(ml_experiments())
        
    except Exception as e:
        logger.error(f"Quantum ML experiments failed: {e}")
        raise

def evaluate_quantum_advantage(**context):
    """Evaluate quantum advantage across experiments"""
    logger.info("Evaluating quantum advantage")
    
    try:
        # Get results from previous tasks
        optimization_results = context['task_instance'].xcom_pull(key='benchmark_results')
        ml_results = context['task_instance'].xcom_pull(key='ml_results')
        
        # Calculate aggregate quantum advantage
        all_advantages = []
        
        if optimization_results:
            all_advantages.extend([r.get("quantum_advantage", 1.0) for r in optimization_results])
        
        if ml_results:
            all_advantages.extend([r.get("quantum_advantage", 1.0) for r in ml_results])
        
        if all_advantages:
            avg_advantage = sum(all_advantages) / len(all_advantages)
            max_advantage = max(all_advantages)
            min_advantage = min(all_advantages)
            
            # Count successful quantum advantages (>1.0)
            successful_advantages = [a for a in all_advantages if a > 1.0]
            success_rate = len(successful_advantages) / len(all_advantages)
            
            advantage_summary = {
                "date": datetime.now().isoformat(),
                "total_experiments": len(all_advantages),
                "average_quantum_advantage": avg_advantage,
                "max_quantum_advantage": max_advantage,
                "min_quantum_advantage": min_advantage,
                "success_rate": success_rate,
                "optimization_experiments": len(optimization_results) if optimization_results else 0,
                "ml_experiments": len(ml_results) if ml_results else 0
            }
            
            context['task_instance'].xcom_push(key='advantage_summary', value=advantage_summary)
            
            logger.info(f"Quantum advantage evaluation: avg={avg_advantage:.3f}, success_rate={success_rate:.2%}")
            
            return advantage_summary
        else:
            logger.warning("No experimental results found for quantum advantage evaluation")
            return {}
    
    except Exception as e:
        logger.error(f"Quantum advantage evaluation failed: {e}")
        raise

def publish_quantum_results(**context):
    """Publish quantum experiment results to Q Platform"""
    logger.info("Publishing quantum experiment results")
    
    try:
        from app.services.pulsar_service import PulsarService
        
        async def publish_results():
            pulsar_service = PulsarService()
            
            # Get all results
            optimization_results = context['task_instance'].xcom_pull(key='benchmark_results')
            ml_results = context['task_instance'].xcom_pull(key='ml_results')
            advantage_summary = context['task_instance'].xcom_pull(key='advantage_summary')
            
            # Publish daily quantum report
            daily_report = {
                "report_type": "daily_quantum_experiments",
                "date": datetime.now().isoformat(),
                "optimization_results": optimization_results,
                "ml_results": ml_results,
                "advantage_summary": advantage_summary,
                "metadata": {
                    "dag_id": context['dag'].dag_id,
                    "task_id": context['task'].task_id,
                    "execution_date": context['execution_date'].isoformat()
                }
            }
            
            await pulsar_service.publish("q.quantum.daily.report", daily_report)
            
            # Publish individual experiment results
            for result in (optimization_results or []):
                await pulsar_service.publish("q.quantum.optimization.completed", result)
            
            for result in (ml_results or []):
                await pulsar_service.publish("q.quantum.ml.completed", result)
            
            logger.info("Quantum experiment results published successfully")
        
        asyncio.run(publish_results())
        
    except Exception as e:
        logger.error(f"Failed to publish quantum results: {e}")
        raise

def store_experiment_memories(**context):
    """Store quantum experiment results as memories"""
    logger.info("Storing quantum experiment memories")
    
    try:
        from app.services.memory_service import MemoryService
        from shared.q_memory_schemas.memory_models import AgentMemory, MemoryType
        
        async def store_memories():
            memory_service = MemoryService()
            
            advantage_summary = context['task_instance'].xcom_pull(key='advantage_summary')
            
            if advantage_summary:
                # Create memory for quantum advantage trends
                memory = AgentMemory(
                    memory_id=f"quantum_daily_{datetime.now().strftime('%Y%m%d')}",
                    agent_id="quantum_experiments_dag",
                    memory_type=MemoryType.EXPERIENCE,
                    content=f"Daily quantum experiments achieved {advantage_summary['average_quantum_advantage']:.3f}x average quantum advantage",
                    context=advantage_summary,
                    importance=min(1.0, advantage_summary['average_quantum_advantage']),
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    access_count=1
                )
                
                await memory_service.store_memory(memory)
                
                logger.info(f"Stored quantum experiment memory: {memory.memory_id}")
        
        asyncio.run(store_memories())
        
    except Exception as e:
        logger.error(f"Failed to store experiment memories: {e}")
        raise

# ===== DAG TASKS =====

# Quantum optimization benchmark task
quantum_optimization_task = PythonOperator(
    task_id='quantum_optimization_benchmark',
    python_callable=run_quantum_optimization_benchmark,
    dag=dag,
    pool='quantum_pool',
    retries=2
)

# Quantum ML experiments task
quantum_ml_task = PythonOperator(
    task_id='quantum_ml_experiments', 
    python_callable=run_quantum_ml_experiments,
    dag=dag,
    pool='quantum_pool',
    retries=2
)

# Quantum advantage evaluation task
quantum_advantage_task = PythonOperator(
    task_id='evaluate_quantum_advantage',
    python_callable=evaluate_quantum_advantage,
    dag=dag,
    retries=1
)

# Publish results task
publish_results_task = PythonOperator(
    task_id='publish_quantum_results',
    python_callable=publish_quantum_results,
    dag=dag,
    retries=2
)

# Store memories task
store_memories_task = PythonOperator(
    task_id='store_experiment_memories',
    python_callable=store_experiment_memories,
    dag=dag,
    retries=1
)

# System health check
health_check_task = BashOperator(
    task_id='quantum_system_health_check',
    bash_command='curl -f http://agentq:8000/health || exit 1',
    dag=dag,
    retries=3
)

# ===== TASK DEPENDENCIES =====

# Parallel execution of optimization and ML experiments
[quantum_optimization_task, quantum_ml_task] >> quantum_advantage_task

# Sequential publishing and storage
quantum_advantage_task >> publish_results_task >> store_memories_task

# Health check runs independently
health_check_task

# Optional: Add task groups for better organization
from airflow.utils.task_group import TaskGroup

with TaskGroup("quantum_experiments", dag=dag) as quantum_group:
    quantum_optimization_task
    quantum_ml_task

with TaskGroup("results_processing", dag=dag) as results_group:
    quantum_advantage_task
    publish_results_task
    store_memories_task

quantum_group >> results_group 