"""
Agent Memory Consolidation DAG

This DAG runs periodically to consolidate agent memories, preventing unbounded growth
and maintaining memory relevance through:
- Daily consolidation of episodic memories
- Weekly consolidation for broader patterns
- Memory importance decay over time
- Cleanup of low-importance memories
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.providers.apache.pulsar.operators.pulsar import PulsarProducerOperator
import json
import logging
import httpx
from typing import List, Dict, Any

# DAG Configuration
default_args = {
    'owner': 'memory-service',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['platform-alerts@qagi.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'memory_consolidation',
    default_args=default_args,
    description='Consolidate and manage agent memories',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    catchup=False,
    max_active_runs=1,
    tags=['memory', 'agents', 'maintenance'],
)

# Service endpoints
MANAGERQ_URL = "http://managerq:8003"
MEMORY_SERVICE_URL = "http://agentq:8000/api/v1/memory"
KNOWLEDGEGRAPH_URL = "http://knowledgegraphq:8006"


def get_active_agents(**context) -> List[str]:
    """Get list of currently active agents from managerQ"""
    try:
        response = httpx.get(f"{MANAGERQ_URL}/api/v1/agents/active")
        response.raise_for_status()
        
        agents = response.json()
        agent_ids = [agent['agent_id'] for agent in agents]
        
        logging.info(f"Found {len(agent_ids)} active agents for memory consolidation")
        
        # Push to XCom for downstream tasks
        context['task_instance'].xcom_push(key='agent_ids', value=agent_ids)
        
        return agent_ids
        
    except Exception as e:
        logging.error(f"Failed to get active agents: {e}")
        raise


def consolidate_agent_memories(agent_id: str, consolidation_type: str, **context):
    """Consolidate memories for a specific agent"""
    try:
        # Call memory service consolidation endpoint
        response = httpx.post(
            f"{MEMORY_SERVICE_URL}/consolidate",
            json={
                "agent_id": agent_id,
                "consolidation_type": consolidation_type
            }
        )
        response.raise_for_status()
        
        result = response.json()
        
        if result.get('consolidation_id'):
            logging.info(
                f"Successfully consolidated {consolidation_type} memories for agent {agent_id}. "
                f"Consolidation ID: {result['consolidation_id']}"
            )
            
            # Store consolidation results in knowledge graph
            store_consolidation_insights(agent_id, result)
        else:
            logging.info(f"No memories to consolidate for agent {agent_id}")
            
    except Exception as e:
        logging.error(f"Failed to consolidate memories for agent {agent_id}: {e}")
        raise


def store_consolidation_insights(agent_id: str, consolidation_result: Dict[str, Any]):
    """Store insights from memory consolidation in knowledge graph"""
    try:
        operations = []
        
        # Create consolidation node
        operations.append({
            "operation": "upsert_vertex",
            "label": "MemoryConsolidation",
            "id_key": "consolidation_id",
            "properties": {
                "consolidation_id": consolidation_result['consolidation_id'],
                "agent_id": agent_id,
                "type": consolidation_result['consolidation_type'],
                "timestamp": consolidation_result['timestamp'],
                "memories_consolidated": len(consolidation_result.get('source_memory_ids', [])),
                "patterns_found": len(consolidation_result.get('patterns_identified', []))
            }
        })
        
        # Create pattern nodes
        for pattern in consolidation_result.get('patterns_identified', []):
            pattern_id = f"pattern_{pattern['type']}_{agent_id}_{consolidation_result['timestamp']}"
            operations.append({
                "operation": "upsert_vertex",
                "label": "MemoryPattern",
                "id_key": "pattern_id",
                "properties": {
                    "pattern_id": pattern_id,
                    "type": pattern['type'],
                    "description": pattern['description'],
                    "entities": pattern.get('entities', [])
                }
            })
            
            # Link pattern to consolidation
            operations.append({
                "operation": "upsert_edge",
                "label": "IDENTIFIED_IN",
                "from_vertex_id": pattern_id,
                "to_vertex_id": consolidation_result['consolidation_id'],
                "from_vertex_label": "MemoryPattern",
                "to_vertex_label": "MemoryConsolidation"
            })
        
        # Send to knowledge graph
        response = httpx.post(
            f"{KNOWLEDGEGRAPH_URL}/api/v1/ingest",
            json={"operations": operations}
        )
        response.raise_for_status()
        
        logging.info(f"Stored consolidation insights for agent {agent_id}")
        
    except Exception as e:
        logging.error(f"Failed to store consolidation insights: {e}")
        # Don't raise - this is not critical


def decay_memory_importance(**context):
    """Apply time-based decay to memory importance scores"""
    try:
        # Get all agent IDs from upstream task
        agent_ids = context['task_instance'].xcom_pull(key='agent_ids')
        
        for agent_id in agent_ids:
            response = httpx.post(
                f"{MEMORY_SERVICE_URL}/decay-importance",
                json={
                    "agent_id": agent_id,
                    "decay_factor": 0.95  # 5% decay per day
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                logging.info(
                    f"Applied importance decay to {result['memories_updated']} memories "
                    f"for agent {agent_id}"
                )
            else:
                logging.warning(
                    f"Failed to apply importance decay for agent {agent_id}: "
                    f"{response.status_code}"
                )
                
    except Exception as e:
        logging.error(f"Failed to apply memory importance decay: {e}")
        raise


def cleanup_old_memories(**context):
    """Remove memories below importance threshold"""
    try:
        agent_ids = context['task_instance'].xcom_pull(key='agent_ids')
        total_cleaned = 0
        
        for agent_id in agent_ids:
            response = httpx.delete(
                f"{MEMORY_SERVICE_URL}/cleanup",
                json={
                    "agent_id": agent_id,
                    "importance_threshold": 0.1,  # Remove memories below 10% importance
                    "preserve_recent_days": 7  # Always keep memories from last 7 days
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                cleaned_count = result.get('memories_removed', 0)
                total_cleaned += cleaned_count
                
                logging.info(
                    f"Cleaned up {cleaned_count} low-importance memories for agent {agent_id}"
                )
            else:
                logging.warning(
                    f"Failed to cleanup memories for agent {agent_id}: {response.status_code}"
                )
                
        logging.info(f"Total memories cleaned up: {total_cleaned}")
        
        # Push cleanup stats to XCom
        context['task_instance'].xcom_push(key='total_cleaned', value=total_cleaned)
        
    except Exception as e:
        logging.error(f"Failed to cleanup old memories: {e}")
        raise


def generate_memory_report(**context):
    """Generate and send a report about memory consolidation"""
    try:
        agent_ids = context['task_instance'].xcom_pull(key='agent_ids')
        total_cleaned = context['task_instance'].xcom_pull(key='total_cleaned') or 0
        
        # Gather memory statistics
        stats = {
            'agents_processed': len(agent_ids),
            'memories_cleaned': total_cleaned,
            'consolidation_time': datetime.now().isoformat(),
            'agent_memory_stats': []
        }
        
        for agent_id in agent_ids:
            try:
                response = httpx.get(f"{MEMORY_SERVICE_URL}/stats/{agent_id}")
                if response.status_code == 200:
                    agent_stats = response.json()
                    stats['agent_memory_stats'].append({
                        'agent_id': agent_id,
                        'total_memories': agent_stats.get('total_memories', 0),
                        'memory_types': agent_stats.get('memory_types', {}),
                        'avg_importance': agent_stats.get('avg_importance', 0)
                    })
            except Exception as e:
                logging.warning(f"Failed to get stats for agent {agent_id}: {e}")
                
        # Create a summary message
        summary = f"""
Memory Consolidation Report - {datetime.now().strftime('%Y-%m-%d')}

Agents Processed: {stats['agents_processed']}
Total Memories Cleaned: {stats['memories_cleaned']}

Agent Memory Distribution:
"""
        
        for agent_stat in stats['agent_memory_stats']:
            summary += f"\n- {agent_stat['agent_id']}: {agent_stat['total_memories']} memories"
            summary += f" (avg importance: {agent_stat['avg_importance']:.2f})"
            
        logging.info(summary)
        
        # Return stats for potential downstream use
        return stats
        
    except Exception as e:
        logging.error(f"Failed to generate memory report: {e}")
        raise


# DAG Task Definitions

start_task = DummyOperator(
    task_id='start',
    dag=dag
)

get_agents_task = PythonOperator(
    task_id='get_active_agents',
    python_callable=get_active_agents,
    provide_context=True,
    dag=dag
)

# Daily consolidation for each agent (dynamic task generation would be better)
daily_consolidation_task = PythonOperator(
    task_id='daily_memory_consolidation',
    python_callable=lambda **context: [
        consolidate_agent_memories(agent_id, 'daily', **context) 
        for agent_id in context['task_instance'].xcom_pull(key='agent_ids')
    ],
    provide_context=True,
    dag=dag
)

# Weekly consolidation (only on Sundays)
weekly_consolidation_task = PythonOperator(
    task_id='weekly_memory_consolidation',
    python_callable=lambda **context: [
        consolidate_agent_memories(agent_id, 'weekly', **context) 
        for agent_id in context['task_instance'].xcom_pull(key='agent_ids')
    ] if datetime.now().weekday() == 6 else logging.info("Skipping weekly consolidation"),
    provide_context=True,
    dag=dag
)

decay_importance_task = PythonOperator(
    task_id='decay_memory_importance',
    python_callable=decay_memory_importance,
    provide_context=True,
    dag=dag
)

cleanup_task = PythonOperator(
    task_id='cleanup_old_memories',
    python_callable=cleanup_old_memories,
    provide_context=True,
    dag=dag
)

report_task = PythonOperator(
    task_id='generate_memory_report',
    python_callable=generate_memory_report,
    provide_context=True,
    dag=dag
)

# Send summary to platform events
publish_summary_task = PulsarProducerOperator(
    task_id='publish_consolidation_summary',
    pulsar_service_url='pulsar://pulsar:6650',
    topic='persistent://public/default/q.platform.events',
    msg="{{ ti.xcom_pull(task_ids='generate_memory_report') | tojson }}",
    dag=dag
)

end_task = DummyOperator(
    task_id='end',
    dag=dag
)

# Task Dependencies
start_task >> get_agents_task
get_agents_task >> [daily_consolidation_task, weekly_consolidation_task]
[daily_consolidation_task, weekly_consolidation_task] >> decay_importance_task
decay_importance_task >> cleanup_task
cleanup_task >> report_task
report_task >> publish_summary_task
publish_summary_task >> end_task 