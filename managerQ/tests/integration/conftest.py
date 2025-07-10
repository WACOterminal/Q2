"""
Pytest configuration and shared fixtures for coordination framework integration tests
"""

import pytest
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, Optional
from unittest.mock import Mock, AsyncMock, patch
import os
import json
import time

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    'agent_count': 5,
    'test_timeout': 30,
    'mock_pulsar': True,
    'mock_kubernetes': True,
    'performance_thresholds': {
        'agent_selection_time_ms': 10,
        'metrics_collection_time_ms': 1000,
        'message_dispatch_time_ms': 100
    },
    'test_data_dir': 'test_data',
    'cleanup_after_tests': True
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return TEST_CONFIG

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="coordination_test_")
    yield Path(temp_dir)
    
    if TEST_CONFIG['cleanup_after_tests']:
        shutil.rmtree(temp_dir)

@pytest.fixture
def mock_kubernetes_client():
    """Mock Kubernetes client for testing"""
    with patch('managerQ.app.core.predictive_autoscaler.k8s_apps_v1') as mock_k8s:
        # Mock deployment object
        mock_deployment = Mock()
        mock_deployment.metadata.name = "agentq-test"
        mock_deployment.spec.replicas = 2
        
        # Mock list response
        mock_response = Mock()
        mock_response.items = [mock_deployment]
        mock_k8s.list_namespaced_deployment.return_value = mock_response
        
        # Mock patch response
        mock_k8s.patch_namespaced_deployment_scale.return_value = Mock()
        
        yield mock_k8s

@pytest.fixture
def mock_pulsar_client_extended():
    """Extended mock Pulsar client with more functionality"""
    from shared.pulsar_client import SharedPulsarClient
    
    client = Mock(spec=SharedPulsarClient)
    client._client = Mock()
    client._connect = Mock()
    client.publish_message = Mock()
    
    # Mock consumer and producer
    mock_consumer = Mock()
    mock_consumer.receive = Mock()
    mock_consumer.acknowledge = Mock()
    mock_consumer.close = Mock()
    
    client._client.subscribe = Mock(return_value=mock_consumer)
    client._client.create_producer = Mock()
    
    return client

@pytest.fixture
def performance_timer():
    """Performance timing fixture"""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.measurements = {}
        
        def start(self, name: str = "default"):
            self.start_time = time.time()
            self.measurements[name] = {'start': self.start_time}
        
        def stop(self, name: str = "default") -> float:
            if name not in self.measurements:
                raise ValueError(f"Timer '{name}' not started")
            
            end_time = time.time()
            duration = end_time - self.measurements[name]['start']
            self.measurements[name]['duration'] = duration
            self.measurements[name]['end'] = end_time
            
            return duration
        
        def get_duration(self, name: str = "default") -> float:
            if name not in self.measurements:
                raise ValueError(f"Timer '{name}' not found")
            return self.measurements[name].get('duration', 0)
        
        def assert_under_threshold(self, name: str, threshold_ms: float):
            duration_ms = self.get_duration(name) * 1000
            assert duration_ms < threshold_ms, f"Performance test '{name}' took {duration_ms:.2f}ms, threshold was {threshold_ms}ms"
    
    return PerformanceTimer()

@pytest.fixture
def test_metrics_data():
    """Generate test metrics data"""
    return {
        'agents_total': 5,
        'agents_healthy': 4,
        'agents_degraded': 1,
        'agents_unhealthy': 0,
        'system_health_percentage': 80.0,
        'dispatcher_queue_size': 10,
        'dispatcher_tasks_dispatched': 100,
        'communication_total_messages': 50,
        'coordination_active_locks': 2
    }

@pytest.fixture
def test_agent_data():
    """Generate test agent data"""
    agents_data = []
    for i in range(TEST_CONFIG['agent_count']):
        agents_data.append({
            'agent_id': f'test_agent_{i}',
            'personality': 'developer' if i % 2 == 0 else 'researcher',
            'capabilities': {
                'personalities': ['developer', 'analyst'] if i % 2 == 0 else ['researcher', 'writer'],
                'max_concurrent_tasks': 5 + i,
                'supported_tools': ['code_analysis'] if i % 2 == 0 else ['research'],
                'resource_requirements': {'cpu': 2.0 + i, 'memory': 4.0 + i}
            },
            'metrics': {
                'total_tasks_completed': 100 + i * 10,
                'current_load': i,
                'cpu_usage': 30.0 + i * 10,
                'memory_usage': 40.0 + i * 5,
                'success_rate': 95.0 - i,
                'error_rate': 5.0 + i
            }
        })
    return agents_data

@pytest.fixture
async def coordination_test_environment(mock_pulsar_client_extended, test_agent_data):
    """Complete coordination framework test environment"""
    from managerQ.app.core.agent_registry import AgentRegistry, Agent, AgentCapabilities, AgentMetrics, AgentStatus
    from managerQ.app.core.task_dispatcher import TaskDispatcher
    from managerQ.app.core.failure_handler import FailureHandler
    from managerQ.app.core.agent_communication import AgentCommunicationHub
    from managerQ.app.core.coordination_protocols import CoordinationProtocolManager
    from managerQ.app.core.performance_monitor import PerformanceMonitor
    from managerQ.app.core.predictive_autoscaler import PredictiveAutoScaler
    
    # Initialize components
    agent_registry = AgentRegistry(mock_pulsar_client_extended)
    task_dispatcher = TaskDispatcher(mock_pulsar_client_extended, agent_registry)
    failure_handler = FailureHandler(agent_registry, task_dispatcher, mock_pulsar_client_extended)
    communication_hub = AgentCommunicationHub(agent_registry, mock_pulsar_client_extended)
    coordination_manager = CoordinationProtocolManager("test_node", agent_registry, communication_hub)
    performance_monitor = PerformanceMonitor(
        agent_registry, task_dispatcher, failure_handler, 
        communication_hub, coordination_manager
    )
    predictive_autoscaler = PredictiveAutoScaler(
        agent_registry, task_dispatcher, performance_monitor
    )
    
    # Create test agents
    for agent_data in test_agent_data:
        capabilities = AgentCapabilities(
            personalities=set(agent_data['capabilities']['personalities']),
            max_concurrent_tasks=agent_data['capabilities']['max_concurrent_tasks'],
            supported_tools=set(agent_data['capabilities']['supported_tools']),
            resource_requirements=agent_data['capabilities']['resource_requirements']
        )
        
        agent = Agent(
            agent_id=agent_data['agent_id'],
            personality=agent_data['personality'],
            topic_name=f"persistent://public/default/{agent_data['agent_id']}",
            capabilities=capabilities
        )
        
        # Set metrics
        metrics_data = agent_data['metrics']
        for key, value in metrics_data.items():
            if hasattr(agent.metrics, key):
                setattr(agent.metrics, key, value)
        
        agent.status = AgentStatus.HEALTHY
        agent_registry.agents[agent.agent_id] = agent
    
    environment = {
        'agent_registry': agent_registry,
        'task_dispatcher': task_dispatcher,
        'failure_handler': failure_handler,
        'communication_hub': communication_hub,
        'coordination_manager': coordination_manager,
        'performance_monitor': performance_monitor,
        'predictive_autoscaler': predictive_autoscaler,
        'pulsar_client': mock_pulsar_client_extended
    }
    
    # Start necessary components
    await communication_hub.start()
    await coordination_manager.start()
    await failure_handler.start()
    await performance_monitor.start()
    
    yield environment
    
    # Cleanup
    await communication_hub.stop()
    await coordination_manager.stop()
    await failure_handler.stop()
    await performance_monitor.stop()

@pytest.fixture
def load_test_data():
    """Generate data for load testing"""
    return {
        'task_count': 100,
        'message_count': 50,
        'agent_variations': 10,
        'concurrent_operations': 20
    }

# Performance test markers
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "stress: mark test as stress test"
    )
    config.addinivalue_line(
        "markers", "failure: mark test as failure scenario test"
    )
    config.addinivalue_line(
        "markers", "smoke: mark test as smoke test"
    )
    config.addinivalue_line(
        "markers", "critical: mark test as critical functionality test"
    )

@pytest.fixture(autouse=True)
def test_environment_setup():
    """Automatically setup test environment"""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'INFO'
    
    yield
    
    # Cleanup
    if 'TESTING' in os.environ:
        del os.environ['TESTING']

# Test data generators
class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def generate_task_data(count: int = 10):
        """Generate test task data"""
        tasks = []
        for i in range(count):
            tasks.append({
                'task_id': f'test_task_{i}',
                'personality': 'developer' if i % 2 == 0 else 'researcher',
                'prompt': f'Test task {i} description',
                'priority': 'high' if i % 3 == 0 else 'normal',
                'tools_required': ['code_analysis'] if i % 2 == 0 else ['research']
            })
        return tasks
    
    @staticmethod
    def generate_failure_scenarios():
        """Generate failure test scenarios"""
        return [
            {
                'type': 'agent_timeout',
                'agent_id': 'test_agent_0',
                'description': 'Agent becomes unresponsive'
            },
            {
                'type': 'resource_exhaustion',
                'agent_id': 'test_agent_1',
                'description': 'Agent runs out of resources'
            },
            {
                'type': 'communication_error',
                'description': 'Message delivery failure'
            },
            {
                'type': 'coordination_failure',
                'description': 'Consensus protocol failure'
            }
        ]
    
    @staticmethod
    def generate_performance_scenarios():
        """Generate performance test scenarios"""
        return [
            {
                'name': 'high_load',
                'task_count': 100,
                'concurrent_agents': 10,
                'duration_seconds': 60
            },
            {
                'name': 'burst_traffic',
                'task_count': 50,
                'burst_interval': 5,
                'duration_seconds': 30
            },
            {
                'name': 'sustained_load',
                'task_count': 200,
                'concurrent_agents': 5,
                'duration_seconds': 300
            }
        ]

@pytest.fixture
def test_data_generator():
    """Test data generator fixture"""
    return TestDataGenerator()

# Helper functions for test assertions
def assert_agent_healthy(agent):
    """Assert that an agent is in healthy state"""
    assert agent.status.value in ['healthy', 'degraded']
    assert agent.metrics.last_heartbeat > 0
    assert agent.metrics.success_rate >= 0

def assert_system_stable(framework):
    """Assert that the coordination system is stable"""
    registry_stats = framework['agent_registry'].get_registry_stats()
    assert registry_stats['total_agents'] > 0
    assert registry_stats['healthy_agents'] > 0
    
    dispatcher_stats = framework['task_dispatcher'].get_queue_stats()
    assert 'routing_strategy' in dispatcher_stats

def assert_performance_acceptable(timer, operation_name, threshold_ms):
    """Assert that operation performance is acceptable"""
    duration_ms = timer.get_duration(operation_name) * 1000
    assert duration_ms < threshold_ms, \
        f"Operation '{operation_name}' took {duration_ms:.2f}ms, exceeds threshold of {threshold_ms}ms"

# Test timeout handler
@pytest.fixture
def test_timeout():
    """Provide test timeout functionality"""
    return TEST_CONFIG['test_timeout']

# Logging helpers
@pytest.fixture
def test_logger():
    """Provide test logger"""
    return logging.getLogger("coordination_test")

# Error collection
@pytest.fixture
def error_collector():
    """Collect errors during testing"""
    errors = []
    
    def add_error(error_msg: str, context: Optional[Dict[str, Any]] = None):
        errors.append({
            'message': error_msg,
            'context': context if context is not None else {},
            'timestamp': time.time()
        })
    
    def get_errors():
        return errors.copy()
    
    def clear_errors():
        errors.clear()
    
    collector = type('ErrorCollector', (), {
        'add_error': add_error,
        'get_errors': get_errors,
        'clear_errors': clear_errors,
        'has_errors': lambda: len(errors) > 0
    })()
    
    return collector 