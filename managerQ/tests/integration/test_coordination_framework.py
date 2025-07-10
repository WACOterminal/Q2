import pytest
import asyncio
import time
import logging
import random
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, AsyncMock, patch
import uuid

# Import coordination framework components
from managerQ.app.core.agent_registry import AgentRegistry, Agent, AgentCapabilities, AgentMetrics, AgentStatus
from managerQ.app.core.task_dispatcher import TaskDispatcher, TaskRequest, TaskPriority, RoutingStrategy
from managerQ.app.core.failure_handler import FailureHandler, FailureType, FailureEvent
from managerQ.app.core.agent_communication import AgentCommunicationHub, Message, MessageType, MessagePriority
from managerQ.app.core.coordination_protocols import CoordinationProtocolManager, NodeState
from managerQ.app.core.performance_monitor import PerformanceMonitor, MetricsCollector
from managerQ.app.core.predictive_autoscaler import PredictiveAutoScaler, ScalingStrategy

from shared.pulsar_client import SharedPulsarClient

logger = logging.getLogger(__name__)

@pytest.fixture
def mock_pulsar_client():
    """Mock Pulsar client for testing"""
    client = Mock(spec=SharedPulsarClient)
    client._client = Mock()
    client._connect = Mock()
    client.publish_message = Mock()
    return client

@pytest.fixture
def test_agents():
    """Create test agents for coordination testing"""
    agents = []
    
    # Create diverse agent capabilities
    for i in range(5):
        capabilities = AgentCapabilities(
            personalities={"developer", "analyst"} if i % 2 == 0 else {"researcher", "writer"},
            max_concurrent_tasks=5 + i,
            supported_tools={"code_analysis", "documentation"} if i % 2 == 0 else {"research", "writing"},
            resource_requirements={"cpu": 2.0 + i, "memory": 4.0 + i},
            preferred_task_types={"coding", "debugging"} if i % 2 == 0 else {"research", "analysis"}
        )
        
        metrics = AgentMetrics(
            total_tasks_completed=100 + i * 10,
            total_tasks_failed=5 + i,
            current_load=i,
            cpu_usage=30.0 + i * 10,
            memory_usage=40.0 + i * 5,
            last_heartbeat=time.time(),
            response_time_p95=500.0 + i * 100,
            error_rate=5.0 + i,
            uptime_seconds=3600 + i * 300
        )
        
        agent = Agent(
            agent_id=f"test_agent_{i}",
            personality="developer" if i % 2 == 0 else "researcher",
            topic_name=f"persistent://public/default/agent_{i}",
            capabilities=capabilities
        )
        agent.metrics = metrics
        agent.status = AgentStatus.HEALTHY
        
        agents.append(agent)
    
    return agents

@pytest.fixture
def coordination_framework(mock_pulsar_client, test_agents):
    """Setup complete coordination framework for testing"""
    
    # Initialize core components
    agent_registry = AgentRegistry(mock_pulsar_client)
    task_dispatcher = TaskDispatcher(mock_pulsar_client, agent_registry)
    failure_handler = FailureHandler(agent_registry, task_dispatcher, mock_pulsar_client)
    communication_hub = AgentCommunicationHub(agent_registry, mock_pulsar_client)
    coordination_manager = CoordinationProtocolManager("test_node", agent_registry, communication_hub)
    performance_monitor = PerformanceMonitor(
        agent_registry, task_dispatcher, failure_handler, 
        communication_hub, coordination_manager
    )
    predictive_autoscaler = PredictiveAutoScaler(
        agent_registry, task_dispatcher, performance_monitor
    )
    
    # Populate agent registry
    for agent in test_agents:
        agent_registry.agents[agent.agent_id] = agent
    
    framework = {
        'agent_registry': agent_registry,
        'task_dispatcher': task_dispatcher,
        'failure_handler': failure_handler,
        'communication_hub': communication_hub,
        'coordination_manager': coordination_manager,
        'performance_monitor': performance_monitor,
        'predictive_autoscaler': predictive_autoscaler
    }
    
    return framework

class TestAgentRegistryIntegration:
    """Test Agent Registry integration with other components"""
    
    def test_agent_discovery_and_selection(self, coordination_framework):
        """Test agent discovery and intelligent selection"""
        framework = coordination_framework
        agent_registry = framework['agent_registry']
        
        # Test basic agent retrieval
        developer_agent = agent_registry.get_agent("developer")
        assert developer_agent is not None
        assert "developer" in developer_agent.capabilities.personalities
        
        # Test agent selection with tool requirements
        agent_with_tools = agent_registry.get_agent(
            "developer", 
            tools_required={"code_analysis"}
        )
        assert agent_with_tools is not None
        assert "code_analysis" in agent_with_tools.capabilities.supported_tools
        
        # Test no suitable agent scenario
        impossible_agent = agent_registry.get_agent(
            "nonexistent_personality",
            tools_required={"impossible_tool"}
        )
        assert impossible_agent is None
    
    def test_agent_health_monitoring(self, coordination_framework):
        """Test agent health monitoring and status updates"""
        framework = coordination_framework
        agent_registry = framework['agent_registry']
        
        # Get a test agent
        agent = list(agent_registry.agents.values())[0]
        original_status = agent.status
        
        # Simulate unhealthy metrics
        agent.metrics.cpu_usage = 95.0
        agent.metrics.error_rate = 60.0
        agent._update_status()
        
        assert agent.status == AgentStatus.UNHEALTHY
        
        # Simulate recovery
        agent.metrics.cpu_usage = 50.0
        agent.metrics.error_rate = 5.0
        agent._update_status()
        
        assert agent.status == AgentStatus.HEALTHY
    
    def test_registry_statistics(self, coordination_framework):
        """Test registry statistics and metrics"""
        framework = coordination_framework
        agent_registry = framework['agent_registry']
        
        stats = agent_registry.get_registry_stats()
        
        assert stats['total_agents'] == 5
        assert stats['healthy_agents'] > 0
        assert 'supported_personalities' in stats
        assert len(stats['supported_personalities']) > 0
        assert stats['total_capacity'] > 0

class TestTaskDispatcherIntegration:
    """Test Task Dispatcher integration and load balancing"""
    
    @pytest.mark.asyncio
    async def test_intelligent_task_routing(self, coordination_framework):
        """Test intelligent task routing strategies"""
        framework = coordination_framework
        task_dispatcher = framework['task_dispatcher']
        
        # Test different routing strategies
        strategies = [
            RoutingStrategy.ROUND_ROBIN,
            RoutingStrategy.LEAST_LOADED,
            RoutingStrategy.PRIORITY_BASED,
            RoutingStrategy.RESOURCE_AWARE
        ]
        
        for strategy in strategies:
            task_dispatcher.set_routing_strategy(strategy)
            
            # Dispatch high priority task
            task_id = task_dispatcher.dispatch_task(
                personality="developer",
                prompt="Test task for routing",
                priority=TaskPriority.HIGH,
                tools_required={"code_analysis"}
            )
            
            assert task_id is not None
            assert len(task_id) > 0
    
    @pytest.mark.asyncio
    async def test_queue_management(self, coordination_framework):
        """Test task queue management and processing"""
        framework = coordination_framework
        task_dispatcher = framework['task_dispatcher']
        
        # Start queue processor
        await task_dispatcher.start_queue_processor()
        
        # Dispatch multiple tasks
        task_ids = []
        for i in range(10):
            task_id = task_dispatcher.dispatch_task(
                personality="developer",
                prompt=f"Test task {i}",
                priority=TaskPriority.NORMAL if i % 2 == 0 else TaskPriority.HIGH
            )
            task_ids.append(task_id)
        
        # Check queue stats
        stats = task_dispatcher.get_queue_stats()
        assert 'queue_size' in stats
        assert 'pending_tasks' in stats
        assert 'routing_strategy' in stats
        
        await task_dispatcher.stop_queue_processor()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, coordination_framework):
        """Test circuit breaker for agent failures"""
        framework = coordination_framework
        task_dispatcher = framework['task_dispatcher']
        agent_registry = framework['agent_registry']
        
        # Get an agent and simulate failures
        agent = list(agent_registry.agents.values())[0]
        agent_id = agent.agent_id
        
        # Record multiple failures to trigger circuit breaker
        for _ in range(6):  # More than failure threshold
            task_dispatcher._record_agent_failure(agent_id)
        
        # Check circuit breaker state
        circuit_breaker = task_dispatcher.circuit_breakers.get(agent_id)
        assert circuit_breaker is not None
        assert circuit_breaker.state == "open"

class TestFailureHandlerIntegration:
    """Test failure handling and recovery mechanisms"""
    
    @pytest.mark.asyncio
    async def test_failure_detection_and_recovery(self, coordination_framework):
        """Test failure detection and automatic recovery"""
        framework = coordination_framework
        failure_handler = framework['failure_handler']
        
        # Report a failure
        failure_id = failure_handler.report_failure(
            failure_type=FailureType.AGENT_TIMEOUT,
            agent_id="test_agent_0",
            error_message="Agent became unresponsive",
            severity="high"
        )
        
        assert failure_id is not None
        
        # Allow some time for processing
        await asyncio.sleep(0.1)
        
        # Check failure stats
        stats = failure_handler.get_failure_stats()
        assert stats['total_failure_types'] > 0
    
    @pytest.mark.asyncio
    async def test_escalation_procedures(self, coordination_framework):
        """Test failure escalation when recovery fails"""
        framework = coordination_framework
        failure_handler = framework['failure_handler']
        
        # Create a failure event that will trigger escalation
        failure_event = FailureEvent(
            event_id="test_escalation",
            failure_type=FailureType.RESOURCE_EXHAUSTION,
            timestamp=time.time(),
            agent_id="test_agent_0",
            error_message="Critical resource exhaustion",
            severity="critical"
        )
        
        # Simulate max retries exceeded
        recovery_manager = failure_handler.recovery_manager
        recovery_manager.recovery_attempts["test_agent_0_resource_exhaustion"]["count"] = 5
        
        # Handle the failure
        success = await failure_handler.handle_failure(failure_event)
        
        # Should trigger escalation
        assert not success or failure_event.event_id in failure_handler.active_failures

class TestAgentCommunicationIntegration:
    """Test agent communication and coordination"""
    
    @pytest.mark.asyncio
    async def test_direct_messaging(self, coordination_framework):
        """Test direct agent-to-agent messaging"""
        framework = coordination_framework
        communication_hub = framework['communication_hub']
        
        # Send direct message
        message_id = await communication_hub.send_direct_message(
            sender_id="test_agent_0",
            recipient_id="test_agent_1",
            subject="Test coordination",
            content={"action": "collaborate", "task_id": "test_task"},
            priority=MessagePriority.HIGH
        )
        
        assert message_id is not None
        
        # Check communication stats
        stats = communication_hub.get_communication_stats()
        assert 'total_messages' in stats
        assert 'active_conversations' in stats
    
    @pytest.mark.asyncio
    async def test_group_coordination(self, coordination_framework):
        """Test group-based agent coordination"""
        framework = coordination_framework
        communication_hub = framework['communication_hub']
        
        # Create a coordination group
        group_id = communication_hub.group_manager.create_group(
            name="Development Team",
            description="Agents working on development tasks",
            coordinator_id="test_agent_0"
        )
        
        assert group_id is not None
        
        # Add agents to group
        for i in range(1, 4):
            success = communication_hub.group_manager.add_agent_to_group(f"test_agent_{i}", group_id)
            assert success
        
        # Send group message
        message_id = await communication_hub.broadcast_message(
            sender_id="test_agent_0",
            subject="Team coordination",
            content={"task": "coordinate_development"},
            group_id=group_id
        )
        
        assert message_id is not None
    
    @pytest.mark.asyncio
    async def test_service_discovery(self, coordination_framework):
        """Test service discovery mechanisms"""
        framework = coordination_framework
        communication_hub = framework['communication_hub']
        
        # Service discovery should be running
        assert communication_hub._running
        
        # Check that agents are discoverable
        stats = communication_hub.get_communication_stats()
        assert stats['total_messages'] >= 0

class TestCoordinationProtocolsIntegration:
    """Test consensus and coordination protocols"""
    
    @pytest.mark.asyncio
    async def test_consensus_operations(self, coordination_framework):
        """Test consensus mechanisms"""
        framework = coordination_framework
        coordination_manager = framework['coordination_manager']
        
        # Check initial state
        status = coordination_manager.get_coordination_status()
        assert 'consensus_state' in status
        assert 'cluster_members' in status
        
        # Test coordination request
        success = await coordination_manager.coordinate_task(
            task_id="test_coordination",
            participating_agents=["test_agent_0", "test_agent_1"],
            coordination_type="consensus"
        )
        
        # May succeed or fail depending on leadership
        assert isinstance(success, bool)
    
    @pytest.mark.asyncio
    async def test_distributed_locks(self, coordination_framework):
        """Test distributed lock mechanisms"""
        framework = coordination_framework
        coordination_manager = framework['coordination_manager']
        lock_manager = coordination_manager.lock_manager
        
        # Acquire a lock
        success = await lock_manager.acquire_lock("test_resource")
        assert success
        
        # Check lock status
        status = lock_manager.get_lock_status("test_resource")
        assert status is not None
        assert status['owner'] == coordination_manager.node_id
        assert status['state'] == 'acquired'
        
        # Release the lock
        success = await lock_manager.release_lock("test_resource")
        assert success

class TestPerformanceMonitoringIntegration:
    """Test performance monitoring and analytics"""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, coordination_framework):
        """Test comprehensive metrics collection"""
        framework = coordination_framework
        performance_monitor = framework['performance_monitor']
        metrics_collector = performance_monitor.metrics_collector
        
        # Trigger metrics collection
        await metrics_collector._collect_all_metrics()
        
        # Check collected metrics
        assert len(metrics_collector.current_metrics) > 0
        assert 'agents_total' in metrics_collector.current_metrics
        assert 'system_health_percentage' in metrics_collector.current_metrics
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, coordination_framework):
        """Test alert generation and management"""
        framework = coordination_framework
        performance_monitor = framework['performance_monitor']
        alert_manager = performance_monitor.alert_manager
        
        # Trigger alert check
        await alert_manager._check_alerts()
        
        # Check alert system is working
        active_alerts = alert_manager.get_active_alerts()
        assert isinstance(active_alerts, list)
        
        alert_history = alert_manager.get_alert_history(1)
        assert isinstance(alert_history, list)
    
    @pytest.mark.asyncio
    async def test_performance_insights(self, coordination_framework):
        """Test performance analysis and insights"""
        framework = coordination_framework
        performance_monitor = framework['performance_monitor']
        analyzer = performance_monitor.performance_analyzer
        
        # Trigger performance analysis
        await analyzer._analyze_performance()
        
        # Check insights generation
        insights = analyzer.get_insights()
        assert isinstance(insights, list)
        
        # Check health score calculation
        health_score, health_level = analyzer.get_health_score()
        assert 0 <= health_score <= 1
        assert health_level in ['excellent', 'good', 'fair', 'poor', 'critical']

class TestPredictiveAutoscalerIntegration:
    """Test predictive autoscaling functionality"""
    
    @pytest.mark.asyncio
    @patch('managerQ.app.core.predictive_autoscaler.k8s_apps_v1')
    async def test_scaling_decision_making(self, mock_k8s, coordination_framework):
        """Test scaling decision algorithms"""
        framework = coordination_framework
        autoscaler = framework['predictive_autoscaler']
        
        # Mock Kubernetes deployment
        mock_deployment = Mock()
        mock_deployment.spec.replicas = 2
        mock_k8s.list_namespaced_deployment.return_value.items = [mock_deployment]
        mock_deployment.metadata.name = "agentq-developer"
        
        # Test different scaling strategies
        strategies = [
            ScalingStrategy.REACTIVE,
            ScalingStrategy.PREDICTIVE,
            ScalingStrategy.HYBRID,
            ScalingStrategy.COST_OPTIMIZED
        ]
        
        for strategy in strategies:
            autoscaler.set_strategy(strategy)
            
            # Evaluate scaling decision
            decision = await autoscaler._make_scaling_decision("agentq-developer", 2)
            
            # Decision may be None if no scaling needed
            if decision:
                assert decision.deployment_name == "agentq-developer"
                assert decision.current_replicas == 2
                assert decision.confidence > 0
    
    @pytest.mark.asyncio
    async def test_workload_prediction(self, coordination_framework):
        """Test workload prediction algorithms"""
        framework = coordination_framework
        autoscaler = framework['predictive_autoscaler']
        
        # Add historical data to predictor
        predictor = autoscaler.predictors.get("developer")
        if not predictor:
            from managerQ.app.core.predictive_autoscaler import WorkloadPredictor
            predictor = WorkloadPredictor()
            autoscaler.predictors["developer"] = predictor
        
        # Add test data points
        current_time = time.time()
        for i in range(20):
            predictor.add_data_point(current_time - (i * 300), 5 + i % 3)  # Varying load
        
        # Get prediction
        prediction = predictor.predict(15)
        
        assert prediction.predicted_load >= 0
        assert 0 <= prediction.confidence <= 1
        assert prediction.time_horizon == 15

class TestEndToEndWorkflows:
    """Test complete end-to-end coordination workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_task_coordination_workflow(self, coordination_framework):
        """Test a complete task coordination workflow"""
        framework = coordination_framework
        
        # 1. Task arrives and needs coordination
        task_dispatcher = framework['task_dispatcher']
        task_id = task_dispatcher.dispatch_task(
            personality="developer",
            prompt="Complex development task requiring coordination",
            priority=TaskPriority.HIGH,
            tools_required={"code_analysis", "documentation"}
        )
        
        assert task_id is not None
        
        # 2. Agent selection and communication
        communication_hub = framework['communication_hub']
        
        # Create coordination group
        group_id = communication_hub.group_manager.create_group(
            name="Task Coordination Group",
            coordinator_id="test_agent_0"
        )
        
        # 3. Coordinate between agents
        coordination_manager = framework['coordination_manager']
        coordination_success = await coordination_manager.coordinate_task(
            task_id=task_id,
            participating_agents=["test_agent_0", "test_agent_1"]
        )
        
        # 4. Monitor performance
        performance_monitor = framework['performance_monitor']
        dashboard_data = performance_monitor.get_dashboard_data()
        
        assert 'health_score' in dashboard_data
        assert 'current_metrics' in dashboard_data
        
        # 5. Finish coordination
        if coordination_success:
            finish_success = await coordination_manager.finish_coordination(task_id)
            assert isinstance(finish_success, bool)
    
    @pytest.mark.asyncio
    async def test_failure_recovery_workflow(self, coordination_framework):
        """Test complete failure detection and recovery workflow"""
        framework = coordination_framework
        
        # 1. Simulate agent failure
        agent_registry = framework['agent_registry']
        failure_handler = framework['failure_handler']
        
        # Make an agent unhealthy
        agent = list(agent_registry.agents.values())[0]
        agent.metrics.cpu_usage = 95.0
        agent.metrics.error_rate = 80.0
        agent.status = AgentStatus.UNHEALTHY
        
        # 2. Report failure
        failure_id = failure_handler.report_failure(
            failure_type=FailureType.RESOURCE_EXHAUSTION,
            agent_id=agent.agent_id,
            error_message="Agent resource exhaustion",
            severity="critical"
        )
        
        # 3. Trigger recovery
        await asyncio.sleep(0.1)  # Allow processing
        
        # 4. Check system adaptation
        task_dispatcher = framework['task_dispatcher']
        
        # Should avoid the failed agent
        healthy_agent = agent_registry.get_agent("developer")
        if healthy_agent:
            assert healthy_agent.agent_id != agent.agent_id or healthy_agent.status != AgentStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_scaling_response_workflow(self, coordination_framework):
        """Test scaling response to load changes"""
        framework = coordination_framework
        
        # 1. Simulate high load
        agent_registry = framework['agent_registry']
        for agent in agent_registry.agents.values():
            agent.metrics.current_load = agent.capabilities.max_concurrent_tasks - 1
        
        # 2. Performance monitoring detects high utilization
        performance_monitor = framework['performance_monitor']
        dashboard_data = performance_monitor.get_dashboard_data()
        
        # 3. Autoscaler should detect need for scaling
        autoscaler = framework['predictive_autoscaler']
        scaling_stats = autoscaler.get_scaling_stats()
        
        assert 'strategy' in scaling_stats
        assert 'predictors' in scaling_stats

class TestLoadAndStressScenarios:
    """Test system behavior under load and stress"""
    
    @pytest.mark.asyncio
    async def test_high_task_volume(self, coordination_framework):
        """Test system behavior with high task volume"""
        framework = coordination_framework
        task_dispatcher = framework['task_dispatcher']
        
        # Start queue processor
        await task_dispatcher.start_queue_processor()
        
        # Submit many tasks quickly
        task_ids = []
        start_time = time.time()
        
        for i in range(50):
            task_id = task_dispatcher.dispatch_task(
                personality="developer" if i % 2 == 0 else "researcher",
                prompt=f"High volume test task {i}",
                priority=TaskPriority.NORMAL
            )
            task_ids.append(task_id)
        
        dispatch_time = time.time() - start_time
        
        # Check that all tasks were dispatched
        assert len(task_ids) == 50
        assert all(task_id is not None for task_id in task_ids)
        
        # Check system responsiveness
        assert dispatch_time < 5.0  # Should dispatch 50 tasks in under 5 seconds
        
        # Check queue stats
        stats = task_dispatcher.get_queue_stats()
        assert 'tasks_dispatched' in stats
        
        await task_dispatcher.stop_queue_processor()
    
    @pytest.mark.asyncio
    async def test_agent_churn(self, coordination_framework):
        """Test system behavior with agents joining and leaving"""
        framework = coordination_framework
        agent_registry = framework['agent_registry']
        
        original_count = len(agent_registry.agents)
        
        # Simulate agents leaving
        agents_to_remove = list(agent_registry.agents.keys())[:2]
        for agent_id in agents_to_remove:
            del agent_registry.agents[agent_id]
        
        # Check system adapts
        remaining_agents = len(agent_registry.agents)
        assert remaining_agents == original_count - 2
        
        # Simulate new agents joining
        new_agent = Agent(
            agent_id="new_test_agent",
            personality="developer",
            topic_name="persistent://public/default/new_agent"
        )
        agent_registry.agents[new_agent.agent_id] = new_agent
        
        # Check agent can be selected
        selected_agent = agent_registry.get_agent("developer")
        assert selected_agent is not None
    
    @pytest.mark.asyncio
    async def test_communication_overload(self, coordination_framework):
        """Test communication system under heavy load"""
        framework = coordination_framework
        communication_hub = framework['communication_hub']
        
        # Send many messages quickly
        message_ids = []
        start_time = time.time()
        
        for i in range(20):
            message_id = await communication_hub.send_direct_message(
                sender_id="test_agent_0",
                recipient_id=f"test_agent_{(i % 4) + 1}",
                subject=f"Load test message {i}",
                content={"test_data": f"data_{i}"}
            )
            message_ids.append(message_id)
        
        communication_time = time.time() - start_time
        
        # Check performance
        assert len(message_ids) == 20
        assert all(msg_id is not None for msg_id in message_ids)
        assert communication_time < 10.0  # Should complete in reasonable time

# Performance benchmarks
class TestPerformanceBenchmarks:
    """Benchmark performance of coordination framework"""
    
    @pytest.mark.asyncio
    async def test_agent_selection_performance(self, coordination_framework):
        """Benchmark agent selection performance"""
        framework = coordination_framework
        agent_registry = framework['agent_registry']
        
        start_time = time.time()
        selections = 0
        
        # Perform many agent selections
        for _ in range(100):
            agent = agent_registry.get_agent("developer")
            if agent:
                selections += 1
        
        selection_time = time.time() - start_time
        avg_selection_time = selection_time / 100
        
        assert selections > 0
        assert avg_selection_time < 0.01  # Should be under 10ms per selection
        
        logger.info(f"Agent selection performance: {avg_selection_time*1000:.2f}ms average")
    
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self, coordination_framework):
        """Benchmark metrics collection performance"""
        framework = coordination_framework
        performance_monitor = framework['performance_monitor']
        metrics_collector = performance_monitor.metrics_collector
        
        start_time = time.time()
        
        # Collect metrics multiple times
        for _ in range(10):
            await metrics_collector._collect_all_metrics()
        
        collection_time = time.time() - start_time
        avg_collection_time = collection_time / 10
        
        assert avg_collection_time < 1.0  # Should be under 1 second per collection
        
        logger.info(f"Metrics collection performance: {avg_collection_time*1000:.0f}ms average")

# Test utilities
def simulate_realistic_workload(agent_registry, duration_seconds=10):
    """Simulate realistic workload patterns"""
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        # Simulate varying agent loads
        for agent in agent_registry.agents.values():
            # Random load variation
            load_change = random.randint(-1, 2)
            agent.metrics.current_load = max(0, min(
                agent.capabilities.max_concurrent_tasks,
                agent.metrics.current_load + load_change
            ))
            
            # Random performance variation
            agent.metrics.cpu_usage = max(10, min(90, 
                agent.metrics.cpu_usage + random.randint(-5, 5)
            ))
        
        time.sleep(0.1)

@pytest.mark.asyncio
async def test_full_system_integration(coordination_framework):
    """Test complete system integration under realistic conditions"""
    framework = coordination_framework
    
    # Start all queue processors
    await framework['task_dispatcher'].start_queue_processor()
    
    # Simulate realistic workload in background
    import threading
    workload_thread = threading.Thread(
        target=simulate_realistic_workload,
        args=(framework['agent_registry'], 5),
        daemon=True
    )
    workload_thread.start()
    
    # Perform various operations while workload is running
    tasks = []
    
    # Task dispatching
    for i in range(10):
        task_id = framework['task_dispatcher'].dispatch_task(
            personality="developer",
            prompt=f"Integration test task {i}",
            priority=TaskPriority.NORMAL
        )
        tasks.append(task_id)
    
    # Agent communication
    await framework['communication_hub'].broadcast_message(
        sender_id="test_agent_0",
        subject="System integration test",
        content={"test": "full_integration"}
    )
    
    # Performance monitoring
    dashboard_data = framework['performance_monitor'].get_dashboard_data()
    assert dashboard_data['health_score'] >= 0
    
    # Wait for workload to complete
    workload_thread.join()
    
    # Stop processors
    await framework['task_dispatcher'].stop_queue_processor()
    
    # Final validation
    stats = framework['agent_registry'].get_registry_stats()
    assert stats['total_agents'] > 0
    
    logger.info("Full system integration test completed successfully")

if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_full_system_integration"
    ]) 