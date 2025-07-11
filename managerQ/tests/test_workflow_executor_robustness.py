# managerQ/tests/test_workflow_executor_robustness.py
import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, Mock
from collections import defaultdict

from managerQ.app.core.workflow_executor import WorkflowExecutor
from managerQ.app.models import Workflow, WorkflowTask, TaskStatus, WorkflowStatus

@pytest.fixture
def mock_workflow_manager():
    with patch('managerQ.app.core.workflow_executor.workflow_manager', new=MagicMock()) as mock:
        yield mock

@pytest.fixture
def executor():
    """Create a workflow executor with mocked dependencies"""
    with patch('managerQ.app.core.workflow_executor.task_dispatcher'), \
         patch('managerQ.app.core.workflow_executor.observability_manager'):
        return WorkflowExecutor()

@pytest.fixture
def mock_workflow():
    """Create a mock workflow for testing"""
    task1 = WorkflowTask(
        task_id="task_1",
        agent_personality="default",
        prompt="Test task 1"
    )
    task2 = WorkflowTask(
        task_id="task_2", 
        agent_personality="default",
        prompt="Test task 2",
        dependencies=["task_1"]
    )
    
    return Workflow(
        workflow_id="test_workflow",
        original_prompt="Test workflow",
        tasks=[task1, task2]
    )

class TestResourceManagement:
    """Test resource management and limits"""
    
    def test_check_resource_limits_within_bounds(self, executor):
        """Test that resource checks pass when within limits"""
        executor._active_workflows = set(["wf1", "wf2"])  # 2 active workflows
        
        with patch.object(executor, '_resource_limits', {"max_concurrent_workflows": 100, "max_concurrent_tasks": 500}):
            assert executor._check_resource_limits() is True
    
    def test_check_resource_limits_workflow_exceeded(self, executor, mock_workflow_manager):
        """Test that resource checks fail when workflow limit exceeded"""
        # Set up 101 active workflows (exceeding limit of 100)
        executor._active_workflows = set([f"wf{i}" for i in range(101)])
        
        with patch.object(executor, '_resource_limits', {"max_concurrent_workflows": 100, "max_concurrent_tasks": 500}):
            assert executor._check_resource_limits() is False
    
    def test_check_resource_limits_task_exceeded(self, executor, mock_workflow_manager):
        """Test that resource checks fail when task limit exceeded"""
        executor._active_workflows = set(["wf1", "wf2"])
        
        # Mock workflow manager to return workflows with many active tasks
        mock_workflow = Mock()
        mock_task = Mock()
        mock_task.status = TaskStatus.RUNNING
        mock_workflow.get_all_tasks_recursive.return_value = [mock_task] * 300  # 300 tasks per workflow
        mock_workflow_manager.get_workflow.return_value = mock_workflow
        
        with patch.object(executor, '_resource_limits', {"max_concurrent_workflows": 100, "max_concurrent_tasks": 500}):
            # 2 workflows * 300 tasks = 600 tasks (exceeds limit of 500)
            assert executor._check_resource_limits() is False

class TestCircularDependencyDetection:
    """Test circular dependency detection"""
    
    def test_detect_circular_dependencies_none(self, executor):
        """Test detection when no circular dependencies exist"""
        task1 = WorkflowTask(task_id="task_1", agent_personality="default", prompt="Task 1")
        task2 = WorkflowTask(task_id="task_2", agent_personality="default", prompt="Task 2", dependencies=["task_1"])
        task3 = WorkflowTask(task_id="task_3", agent_personality="default", prompt="Task 3", dependencies=["task_2"])
        
        workflow = Workflow(
            workflow_id="test_workflow",
            original_prompt="Test",
            tasks=[task1, task2, task3]
        )
        
        assert executor._detect_circular_dependencies(workflow) is False
    
    def test_detect_circular_dependencies_simple_cycle(self, executor):
        """Test detection of simple circular dependency"""
        task1 = WorkflowTask(task_id="task_1", agent_personality="default", prompt="Task 1", dependencies=["task_2"])
        task2 = WorkflowTask(task_id="task_2", agent_personality="default", prompt="Task 2", dependencies=["task_1"])
        
        workflow = Workflow(
            workflow_id="test_workflow",
            original_prompt="Test",
            tasks=[task1, task2]
        )
        
        assert executor._detect_circular_dependencies(workflow) is True
    
    def test_detect_circular_dependencies_complex_cycle(self, executor):
        """Test detection of complex circular dependency"""
        task1 = WorkflowTask(task_id="task_1", agent_personality="default", prompt="Task 1", dependencies=["task_3"])
        task2 = WorkflowTask(task_id="task_2", agent_personality="default", prompt="Task 2", dependencies=["task_1"])
        task3 = WorkflowTask(task_id="task_3", agent_personality="default", prompt="Task 3", dependencies=["task_2"])
        
        workflow = Workflow(
            workflow_id="test_workflow",
            original_prompt="Test",
            tasks=[task1, task2, task3]
        )
        
        assert executor._detect_circular_dependencies(workflow) is True

class TestTimeoutHandling:
    """Test workflow timeout handling"""
    
    def test_check_workflow_timeout_not_expired(self, executor):
        """Test timeout check when workflow hasn't timed out"""
        workflow_id = "test_workflow"
        future_time = datetime.now() + timedelta(hours=1)
        executor._workflow_timeouts[workflow_id] = future_time
        
        assert executor._check_workflow_timeout(workflow_id) is False
    
    def test_check_workflow_timeout_expired(self, executor):
        """Test timeout check when workflow has timed out"""
        workflow_id = "test_workflow"
        past_time = datetime.now() - timedelta(hours=1)
        executor._workflow_timeouts[workflow_id] = past_time
        
        assert executor._check_workflow_timeout(workflow_id) is True
    
    def test_check_workflow_timeout_no_timeout_set(self, executor):
        """Test timeout check when no timeout is set"""
        workflow_id = "test_workflow"
        
        assert executor._check_workflow_timeout(workflow_id) is False

class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_initial_state(self, executor):
        """Test circuit breaker initial state is CLOSED"""
        service_name = "test_service"
        state = executor._get_circuit_breaker_state(service_name)
        
        assert state["state"] == "CLOSED"
        assert state["failure_count"] == 0
        assert state["success_count"] == 0
    
    def test_circuit_breaker_failure_tracking(self, executor):
        """Test circuit breaker tracks failures correctly"""
        service_name = "test_service"
        
        # Record multiple failures
        for _ in range(3):
            executor._update_circuit_breaker_on_failure(service_name)
        
        state = executor._get_circuit_breaker_state(service_name)
        assert state["failure_count"] == 3
        assert state["state"] == "CLOSED"  # Not yet opened
    
    def test_circuit_breaker_opens_on_threshold(self, executor):
        """Test circuit breaker opens when failure threshold is reached"""
        service_name = "test_service"
        
        # Record failures to exceed threshold (5)
        for _ in range(6):
            executor._update_circuit_breaker_on_failure(service_name)
        
        state = executor._get_circuit_breaker_state(service_name)
        assert state["state"] == "OPEN"
        assert state["failure_count"] == 6
    
    def test_circuit_breaker_success_resets_failures(self, executor):
        """Test circuit breaker resets failure count on success"""
        service_name = "test_service"
        
        # Record some failures
        for _ in range(3):
            executor._update_circuit_breaker_on_failure(service_name)
        
        # Record success
        executor._update_circuit_breaker_on_success(service_name)
        
        state = executor._get_circuit_breaker_state(service_name)
        assert state["failure_count"] == 0
        assert state["state"] == "CLOSED"
    
    def test_circuit_breaker_half_open_recovery(self, executor):
        """Test circuit breaker recovery through HALF_OPEN state"""
        service_name = "test_service"
        
        # Open the circuit breaker
        for _ in range(6):
            executor._update_circuit_breaker_on_failure(service_name)
        
        # Simulate time passage for recovery
        state = executor._get_circuit_breaker_state(service_name)
        state["last_failure_time"] = datetime.now() - timedelta(seconds=120)  # 2 minutes ago
        
        # Check if request should be allowed (should transition to HALF_OPEN)
        assert executor._should_allow_request(service_name) is True
        
        # Verify state transitioned to HALF_OPEN
        state = executor._get_circuit_breaker_state(service_name)
        assert state["state"] == "HALF_OPEN"
    
    def test_circuit_breaker_blocks_requests_when_open(self, executor):
        """Test circuit breaker blocks requests when OPEN"""
        service_name = "test_service"
        
        # Open the circuit breaker
        for _ in range(6):
            executor._update_circuit_breaker_on_failure(service_name)
        
        # Request should be blocked
        assert executor._should_allow_request(service_name) is False

class TestDeadlockDetection:
    """Test deadlock detection"""
    
    @pytest.mark.asyncio
    async def test_deadlock_detection_no_deadlock(self, executor, mock_workflow_manager):
        """Test deadlock detection when no deadlock exists"""
        workflow_id = "test_workflow"
        executor._active_workflows.add(workflow_id)
        
        # Mock workflow with recent activity
        mock_workflow = Mock()
        mock_workflow.created_at = datetime.now() - timedelta(minutes=5)
        mock_task = Mock()
        mock_task.updated_at = datetime.now() - timedelta(minutes=2)  # Recent activity
        mock_task.status = TaskStatus.RUNNING
        mock_workflow.get_all_tasks_recursive.return_value = [mock_task]
        mock_workflow_manager.get_workflow.return_value = mock_workflow
        
        # Should not detect deadlock
        await executor._check_for_deadlocks()
        
        # Workflow should still be active
        assert workflow_id in executor._active_workflows
    
    @pytest.mark.asyncio 
    async def test_deadlock_detection_stuck_workflow(self, executor, mock_workflow_manager):
        """Test deadlock detection identifies stuck workflow"""
        workflow_id = "test_workflow"
        executor._active_workflows.add(workflow_id)
        
        # Mock workflow with old activity and pending tasks but no running tasks
        mock_workflow = Mock()
        mock_workflow.created_at = datetime.now() - timedelta(hours=2)
        mock_workflow.status = WorkflowStatus.RUNNING
        
        # Pending task (stuck)
        mock_pending_task = Mock()
        mock_pending_task.updated_at = datetime.now() - timedelta(hours=1)
        mock_pending_task.status = TaskStatus.PENDING
        
        mock_workflow.get_all_tasks_recursive.return_value = [mock_pending_task]
        mock_workflow_manager.get_workflow.return_value = mock_workflow
        mock_workflow_manager.update_workflow = Mock()
        
        # Should detect deadlock and mark workflow as failed
        await executor._check_for_deadlocks()
        
        # Verify workflow was marked as failed
        mock_workflow_manager.update_workflow.assert_called_once()
        assert workflow_id not in executor._active_workflows

class TestWorkflowCleanup:
    """Test workflow cleanup functionality"""
    
    def test_cleanup_workflow_removes_tracking(self, executor):
        """Test cleanup removes all workflow tracking data"""
        workflow_id = "test_workflow"
        
        # Set up tracking data
        executor._active_workflows.add(workflow_id)
        executor._workflow_timeouts[workflow_id] = datetime.now()
        executor._circuit_breaker_state[workflow_id] = {"test": "data"}
        
        # Cleanup
        executor._cleanup_workflow(workflow_id)
        
        # Verify all tracking data removed
        assert workflow_id not in executor._active_workflows
        assert workflow_id not in executor._workflow_timeouts
        assert workflow_id not in executor._circuit_breaker_state
    
    def test_cleanup_workflow_handles_missing_data(self, executor):
        """Test cleanup handles missing tracking data gracefully"""
        workflow_id = "test_workflow"
        
        # Cleanup should not raise exception even if no tracking data exists
        executor._cleanup_workflow(workflow_id)
        
        # Should be safe to call multiple times
        executor._cleanup_workflow(workflow_id)

class TestWorkflowExecutionRobustness:
    """Test overall workflow execution robustness"""
    
    @pytest.mark.asyncio
    async def test_execute_workflow_with_resource_check(self, executor, mock_workflow_manager):
        """Test workflow execution respects resource limits"""
        workflow_id = "test_workflow"
        user_id = "test_user"
        
        # Mock resource limit exceeded
        with patch.object(executor, '_check_resource_limits', return_value=False):
            await executor.execute_workflow(workflow_id, user_id)
        
        # Workflow should not be retrieved if resources exceeded
        mock_workflow_manager.get_workflow.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_execute_workflow_with_circular_dependency(self, executor, mock_workflow_manager):
        """Test workflow execution rejects circular dependencies"""
        workflow_id = "test_workflow"
        user_id = "test_user"
        
        # Mock workflow with circular dependency
        mock_workflow = Mock()
        mock_workflow.status = WorkflowStatus.PENDING
        mock_workflow_manager.get_workflow.return_value = mock_workflow
        
        with patch.object(executor, '_check_resource_limits', return_value=True), \
             patch.object(executor, '_detect_circular_dependencies', return_value=True):
            
            await executor.execute_workflow(workflow_id, user_id)
        
        # Workflow should be marked as failed
        mock_workflow_manager.update_workflow.assert_called()
        assert mock_workflow.status == WorkflowStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_process_workflow_with_timeout(self, executor, mock_workflow_manager):
        """Test workflow processing handles timeouts"""
        mock_workflow = Mock()
        mock_workflow.workflow_id = "test_workflow"
        
        # Mock timeout check to return True
        with patch.object(executor, '_check_workflow_timeout', return_value=True):
            await executor.process_workflow(mock_workflow)
        
        # Workflow should be marked as failed due to timeout
        assert mock_workflow.status == WorkflowStatus.FAILED
        assert "exceeded maximum execution time" in mock_workflow.final_result
        mock_workflow_manager.update_workflow.assert_called() 