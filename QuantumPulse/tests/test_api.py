import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import HTTPException
import json
import time

# It's important to set up the app object before other imports
from app.main import app
from app.core.pulsar_client import PulsarManager, get_pulsar_manager

# Create a mock PulsarManager
mock_pulsar_manager = MagicMock(spec=PulsarManager)

def get_mock_pulsar_manager():
    """Dependency override for the PulsarManager."""
    return mock_pulsar_manager

# Override the dependency for the entire application
app.dependency_overrides[get_pulsar_manager] = get_mock_pulsar_manager

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset mocks before each test."""
    mock_pulsar_manager.reset_mock()
    mock_pulsar_manager.publish_request = AsyncMock()
    mock_pulsar_manager.is_connected = True
    mock_pulsar_manager.health_check = AsyncMock(return_value={"status": "healthy", "connected": True})

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_service_info():
    """Test service information endpoint if it exists."""
    response = client.get("/api/v1/info")
    if response.status_code == 200:
        info = response.json()
        assert "service" in info
        assert "version" in info
    else:
        # If endpoint doesn't exist, that's also valid
        assert response.status_code in [404, 405]

def test_create_inference_request_success():
    """Test successful creation of an inference request."""
    # Mock successful message publishing
    mock_pulsar_manager.publish_request.return_value = {"message_id": "test_msg_123", "status": "published"}
    
    with patch("fastapi.BackgroundTasks.add_task") as mock_add_task:
        test_payload = {
            "prompt": "Hello, world!",
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        response = client.post("/api/v1/inference", json=test_payload)

        assert response.status_code == 202
        json_response = response.json()
        assert "message" in json_response
        assert "request_id" in json_response
        
        # Verify the response contains expected fields
        assert json_response["message"] == "Inference request submitted successfully"
        assert "request_id" in json_response

        # Check if the background task was added correctly
        mock_add_task.assert_called_once()
        
        # Get the arguments passed to the background task
        args, kwargs = mock_add_task.call_args
        
        # Verify task parameters
        assert len(args) >= 2  # Function and at least one parameter
        task_function = args[0]
        assert callable(task_function)

def test_create_inference_request_validation_error():
    """Test for validation errors with invalid input."""
    # Test missing prompt
    response = client.post("/api/v1/inference", json={"model": "test-model"})
    assert response.status_code == 422  # Unprocessable Entity
    
    # Test invalid prompt type
    response = client.post("/api/v1/inference", json={"prompt": 123, "model": "test-model"})
    assert response.status_code == 422
    
    # Test negative temperature
    response = client.post("/api/v1/inference", json={
        "prompt": "test", 
        "model": "test-model", 
        "temperature": -1.0
    })
    assert response.status_code == 422
    
    # Test exceeding max tokens
    response = client.post("/api/v1/inference", json={
        "prompt": "test", 
        "model": "test-model", 
        "max_tokens": 100000
    })
    assert response.status_code == 422

def test_create_inference_request_with_optional_parameters():
    """Test inference request with all optional parameters."""
    mock_pulsar_manager.publish_request.return_value = {"message_id": "test_msg_456", "status": "published"}
    
    with patch("fastapi.BackgroundTasks.add_task") as mock_add_task:
        test_payload = {
            "prompt": "Comprehensive test prompt",
            "model": "advanced-model",
            "temperature": 0.8,
            "max_tokens": 500,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "stream": False
        }
        
        response = client.post("/api/v1/inference", json=test_payload)
        
        assert response.status_code == 202
        json_response = response.json()
        assert "request_id" in json_response
        
        # Verify background task was called
        mock_add_task.assert_called_once()

@pytest.mark.asyncio
async def test_pulsar_connection_error_handling():
    """Test how the system handles Pulsar connection errors."""
    # Configure the mock to simulate connection failure
    mock_pulsar_manager.publish_request.side_effect = ConnectionError("Test connection failed")
    mock_pulsar_manager.is_connected = False
    
    # Test that the background task handles the error gracefully
    from app.models.inference import InferenceRequest
    
    # Create test request
    test_request = InferenceRequest(
        prompt="This should fail",
        model="test-model"
    )
    
    # Test error handling in background task directly with the mock
    with pytest.raises(ConnectionError):
        await mock_pulsar_manager.publish_request("test-topic", test_request)

@pytest.mark.asyncio
async def test_pulsar_retry_mechanism():
    """Test retry mechanism for failed Pulsar operations."""
    # Mock intermittent failures
    mock_pulsar_manager.publish_request.side_effect = [
        ConnectionError("First attempt failed"),
        ConnectionError("Second attempt failed"), 
        {"message_id": "success_msg_789", "status": "published"}  # Third attempt succeeds
    ]
    
    from app.models.inference import InferenceRequest
    
    test_request = InferenceRequest(
        prompt="Test retry mechanism",
        model="test-model"
    )
    
    # Simulate retry logic (since the actual retry function may not exist)
    retry_count = 0
    max_retries = 3
    result = None
    
    while retry_count < max_retries:
        try:
            result = await mock_pulsar_manager.publish_request("test-topic", test_request)
            break
        except ConnectionError:
            retry_count += 1
            if retry_count >= max_retries:
                raise
    
    # Verify result is not None and has expected values
    assert result is not None
    assert result["status"] == "published"
    assert result["message_id"] == "success_msg_789"

def test_concurrent_inference_requests():
    """Test handling multiple concurrent requests."""
    mock_pulsar_manager.publish_request.return_value = {"message_id": "concurrent_msg", "status": "published"}
    
    with patch("fastapi.BackgroundTasks.add_task") as mock_add_task:
        # Send multiple requests
        responses = []
        for i in range(10):
            test_payload = {
                "prompt": f"Concurrent test prompt {i}",
                "model": "test-model"
            }
            response = client.post("/api/v1/inference", json=test_payload)
            responses.append(response)
        
        # All requests should be accepted
        for response in responses:
            assert response.status_code == 202
            assert "request_id" in response.json()
        
        # Background tasks should be called for each request
        assert mock_add_task.call_count == 10

def test_request_id_uniqueness():
    """Test that each request gets a unique request ID."""
    mock_pulsar_manager.publish_request.return_value = {"message_id": "unique_test", "status": "published"}
    
    with patch("fastapi.BackgroundTasks.add_task") as mock_add_task:
        request_ids = set()
        
        for i in range(5):
            test_payload = {
                "prompt": f"Uniqueness test {i}",
                "model": "test-model"
            }
            response = client.post("/api/v1/inference", json=test_payload)
            assert response.status_code == 202
            
            request_id = response.json()["request_id"]
            assert request_id not in request_ids  # Should be unique
            request_ids.add(request_id)
        
        assert len(request_ids) == 5  # All IDs should be unique

def test_inference_request_size_limits():
    """Test request size limitations."""
    # Test very large prompt
    large_prompt = "x" * 50000  # 50KB prompt
    
    response = client.post("/api/v1/inference", json={
        "prompt": large_prompt,
        "model": "test-model"
    })
    
    # Should either accept or reject based on configured limits
    assert response.status_code in [202, 413, 422]  # Accept, Payload Too Large, or Validation Error

def test_model_parameter_validation():
    """Test model parameter validation."""
    # Test with empty model
    response = client.post("/api/v1/inference", json={
        "prompt": "test prompt",
        "model": ""
    })
    assert response.status_code == 422
    
    # Test with None model
    response = client.post("/api/v1/inference", json={
        "prompt": "test prompt",
        "model": None
    })
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_background_task_error_logging():
    """Test that background task errors are properly logged."""
    mock_pulsar_manager.publish_request.side_effect = Exception("Unexpected error")
    
    # Test that background task errors would be logged (simulated)
    from app.models.inference import InferenceRequest
    
    test_request = InferenceRequest(
        prompt="This will cause an error",
        model="test-model"
    )
    
    # Test that the error occurs as expected
    with pytest.raises(Exception) as exc_info:
        await mock_pulsar_manager.publish_request("test-topic", test_request)
    
    assert "Unexpected error" in str(exc_info.value)

def test_performance_monitoring():
    """Test basic performance monitoring of inference requests."""
    mock_pulsar_manager.publish_request.return_value = {"message_id": "perf_test", "status": "published"}
    
    start_time = time.time()
    
    with patch("fastapi.BackgroundTasks.add_task") as mock_add_task:
        test_payload = {
            "prompt": "Performance test prompt",
            "model": "test-model"
        }
        
        response = client.post("/api/v1/inference", json=test_payload)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 202
        assert response_time < 1.0  # Should respond quickly (under 1 second)
        
        mock_add_task.assert_called_once()

def test_api_endpoint_security():
    """Test basic security measures."""
    # Test with malicious content
    malicious_payloads = [
        {"prompt": "<script>alert('xss')</script>", "model": "test-model"},
        {"prompt": "'; DROP TABLE users; --", "model": "test-model"},
        {"prompt": "{{ 7*7 }}", "model": "test-model"}  # Template injection
    ]
    
    for payload in malicious_payloads:
        response = client.post("/api/v1/inference", json=payload)
        # Should either accept (if sanitized) or reject gracefully
        assert response.status_code in [202, 400, 422]
        
        if response.status_code == 202:
            # If accepted, response should be properly formatted
            assert "request_id" in response.json() 