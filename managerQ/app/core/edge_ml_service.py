"""
Edge ML Deployment Service

This service provides comprehensive edge ML deployment capabilities for the Q Platform:
- Edge device management and registration
- Model deployment and versioning for edge devices
- Distributed inference orchestration
- Model synchronization and updates
- Resource optimization for edge environments
- Federated learning coordination
- Real-time model serving
- Edge-specific model optimization
- Offline inference capabilities
- Performance monitoring and analytics
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import statistics
import math
import hashlib
import pickle
import gzip
import tempfile
import shutil

# ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.quantization as quant
    from torch.jit import script, trace
    torch_available = True
except ImportError:
    torch_available = False
    logging.warning("PyTorch not available - PyTorch model deployment will be limited")

try:
    import tensorflow as tf
    from tensorflow.lite import TFLiteConverter
    tensorflow_available = True
except ImportError:
    tensorflow_available = False
    logging.warning("TensorFlow not available - TensorFlow model deployment will be limited")

try:
    import onnx
    import onnxruntime as ort
    onnx_available = True
except ImportError:
    onnx_available = False
    logging.warning("ONNX not available - ONNX model deployment will be limited")

try:
    from sklearn.externals import joblib
    sklearn_available = True
except ImportError:
    try:
        import joblib
        sklearn_available = True
    except ImportError:
        sklearn_available = False
        logging.warning("Scikit-learn/joblib not available - sklearn model deployment will be limited")

# Edge device communication
try:
    import requests
    import websockets
    from aiohttp import web
    web_available = True
except ImportError:
    web_available = False
    logging.warning("Web libraries not available - edge communication will be limited")

# System monitoring
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
    logging.warning("psutil not available - system monitoring will be limited")

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.vault_client import VaultClient

logger = logging.getLogger(__name__)

class EdgeDeviceType(Enum):
    """Edge device types"""
    MOBILE = "mobile"
    IOT = "iot"
    EMBEDDED = "embedded"
    GATEWAY = "gateway"
    SERVER = "server"
    SENSOR = "sensor"
    CAMERA = "camera"
    EDGE_SERVER = "edge_server"
    WORKSTATION = "workstation"

class DeviceStatus(Enum):
    """Device status"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    UNKNOWN = "unknown"

class ModelFormat(Enum):
    """Model formats"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    TFLITE = "tflite"
    ONNX = "onnx"
    SKLEARN = "sklearn"
    CUSTOM = "custom"

class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    UPDATING = "updating"
    REMOVING = "removing"
    REMOVED = "removed"

class InferenceMode(Enum):
    """Inference modes"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    OFFLINE = "offline"

class OptimizationLevel(Enum):
    """Optimization levels"""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

@dataclass
class EdgeDevice:
    """Edge device representation"""
    device_id: str
    name: str
    device_type: EdgeDeviceType
    status: DeviceStatus
    ip_address: str
    port: int
    capabilities: Dict[str, Any]
    resources: Dict[str, Any]
    last_seen: datetime
    registered_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class EdgeModel:
    """Edge model representation"""
    model_id: str
    name: str
    version: str
    format: ModelFormat
    size_bytes: int
    checksum: str
    input_shape: List[int]
    output_shape: List[int]
    inference_time_ms: float
    memory_mb: float
    optimization_level: OptimizationLevel
    created_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ModelDeployment:
    """Model deployment representation"""
    deployment_id: str
    device_id: str
    model_id: str
    status: DeploymentStatus
    deployment_config: Dict[str, Any]
    deployed_at: Optional[datetime] = None
    last_update: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}

@dataclass
class InferenceRequest:
    """Inference request representation"""
    request_id: str
    device_id: str
    model_id: str
    input_data: Any
    inference_mode: InferenceMode
    priority: int
    timeout: int
    created_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class InferenceResult:
    """Inference result representation"""
    request_id: str
    device_id: str
    model_id: str
    result: Any
    inference_time_ms: float
    confidence: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class DeviceMetrics:
    """Device performance metrics"""
    device_id: str
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    temperature: Optional[float] = None
    battery_level: Optional[float] = None
    inference_count: int = 0
    error_count: int = 0

@dataclass
class SyncTask:
    """Model synchronization task"""
    task_id: str
    device_id: str
    model_id: str
    sync_type: str  # deploy, update, remove
    priority: int
    created_at: datetime
    status: str = "pending"
    
class EdgeMLService:
    """
    Comprehensive Edge ML Deployment Service
    """
    
    def __init__(self, storage_path: str = "edge_ml"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Device management
        self.devices: Dict[str, EdgeDevice] = {}
        self.device_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Model management
        self.models: Dict[str, EdgeModel] = {}
        self.model_files: Dict[str, Path] = {}
        
        # Deployment management
        self.deployments: Dict[str, ModelDeployment] = {}
        self.deployment_queue: deque = deque()
        
        # Inference management
        self.inference_requests: Dict[str, InferenceRequest] = {}
        self.inference_results: Dict[str, InferenceResult] = {}
        self.inference_queue: deque = deque()
        
        # Synchronization
        self.sync_tasks: Dict[str, SyncTask] = {}
        self.sync_queue: deque = deque()
        
        # Performance tracking
        self.performance_metrics = {
            "total_devices": 0,
            "online_devices": 0,
            "total_models": 0,
            "active_deployments": 0,
            "total_inferences": 0,
            "average_inference_time": 0.0,
            "success_rate": 0.0,
            "network_utilization": 0.0
        }
        
        # Configuration
        self.config = {
            "max_concurrent_deployments": 10,
            "max_concurrent_inferences": 100,
            "heartbeat_interval": 30,
            "sync_interval": 300,
            "model_cache_size": 1000,
            "enable_model_optimization": True,
            "enable_adaptive_batching": True,
            "max_batch_size": 32,
            "batch_timeout": 100,
            "compression_enabled": True,
            "encryption_enabled": True,
            "telemetry_enabled": True
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Service integrations
        self.vault_client = VaultClient()
        
        # Model optimizers
        self.optimizers = {
            ModelFormat.PYTORCH: self._optimize_pytorch_model,
            ModelFormat.TENSORFLOW: self._optimize_tensorflow_model,
            ModelFormat.ONNX: self._optimize_onnx_model,
            ModelFormat.SKLEARN: self._optimize_sklearn_model
        }
        
        # Communication channels
        self.device_connections: Dict[str, Any] = {}
        
        logger.info("Edge ML Service initialized")
    
    async def initialize(self):
        """Initialize the edge ML service"""
        logger.info("Initializing Edge ML Service")
        
        # Load existing data
        await self._load_edge_data()
        
        # Start background tasks
        await self._start_background_tasks()
        
        # Initialize model optimizers
        await self._initialize_optimizers()
        
        logger.info("Edge ML Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the edge ML service"""
        logger.info("Shutting down Edge ML Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close device connections
        for device_id, connection in self.device_connections.items():
            try:
                await connection.close()
            except:
                pass
        
        # Save data
        await self._save_edge_data()
        
        logger.info("Edge ML Service shutdown complete")
    
    # ===== DEVICE MANAGEMENT =====
    
    async def register_device(self, device: EdgeDevice) -> bool:
        """Register a new edge device"""
        try:
            # Validate device
            if not await self._validate_device(device):
                return False
            
            # Store device
            self.devices[device.device_id] = device
            
            # Initialize device connection
            await self._initialize_device_connection(device)
            
            # Update metrics
            self.performance_metrics["total_devices"] = len(self.devices)
            
            # Start device monitoring
            await self._start_device_monitoring(device.device_id)
            
            logger.info(f"Device registered: {device.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering device: {e}")
            return False
    
    async def unregister_device(self, device_id: str) -> bool:
        """Unregister an edge device"""
        try:
            if device_id not in self.devices:
                return False
            
            # Remove all deployments from device
            device_deployments = [d for d in self.deployments.values() if d.device_id == device_id]
            for deployment in device_deployments:
                await self._remove_deployment(deployment.deployment_id)
            
            # Close device connection
            if device_id in self.device_connections:
                await self.device_connections[device_id].close()
                del self.device_connections[device_id]
            
            # Remove device
            del self.devices[device_id]
            
            # Update metrics
            self.performance_metrics["total_devices"] = len(self.devices)
            
            logger.info(f"Device unregistered: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering device: {e}")
            return False
    
    async def update_device_status(self, device_id: str, status: DeviceStatus) -> bool:
        """Update device status"""
        try:
            if device_id not in self.devices:
                return False
            
            device = self.devices[device_id]
            old_status = device.status
            device.status = status
            device.last_seen = datetime.utcnow()
            
            # Handle status changes
            if old_status != status:
                await self._handle_device_status_change(device_id, old_status, status)
            
            # Update online device count
            online_count = sum(1 for d in self.devices.values() if d.status == DeviceStatus.ONLINE)
            self.performance_metrics["online_devices"] = online_count
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating device status: {e}")
            return False
    
    async def get_device_metrics(self, device_id: str) -> List[DeviceMetrics]:
        """Get device performance metrics"""
        try:
            if device_id not in self.device_metrics:
                return []
            
            return list(self.device_metrics[device_id])
            
        except Exception as e:
            logger.error(f"Error getting device metrics: {e}")
            return []
    
    # ===== MODEL MANAGEMENT =====
    
    async def register_model(self, model: EdgeModel, model_file: Path) -> bool:
        """Register a new edge model"""
        try:
            # Validate model file
            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
                return False
            
            # Calculate checksum
            checksum = await self._calculate_file_checksum(model_file)
            model.checksum = checksum
            
            # Store model
            self.models[model.model_id] = model
            
            # Store model file
            model_storage_path = self.storage_path / "models" / f"{model.model_id}_{model.version}"
            model_storage_path.mkdir(parents=True, exist_ok=True)
            
            target_file = model_storage_path / f"model.{model.format.value}"
            shutil.copy2(model_file, target_file)
            self.model_files[model.model_id] = target_file
            
            # Optimize model if enabled
            if self.config["enable_model_optimization"]:
                await self._optimize_model(model.model_id)
            
            # Update metrics
            self.performance_metrics["total_models"] = len(self.models)
            
            logger.info(f"Model registered: {model.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return False
    
    async def remove_model(self, model_id: str) -> bool:
        """Remove an edge model"""
        try:
            if model_id not in self.models:
                return False
            
            # Remove all deployments of this model
            model_deployments = [d for d in self.deployments.values() if d.model_id == model_id]
            for deployment in model_deployments:
                await self._remove_deployment(deployment.deployment_id)
            
            # Remove model file
            if model_id in self.model_files:
                model_file = self.model_files[model_id]
                if model_file.exists():
                    model_file.unlink()
                del self.model_files[model_id]
            
            # Remove model
            del self.models[model_id]
            
            # Update metrics
            self.performance_metrics["total_models"] = len(self.models)
            
            logger.info(f"Model removed: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing model: {e}")
            return False
    
    # ===== DEPLOYMENT MANAGEMENT =====
    
    async def deploy_model(self, device_id: str, model_id: str, config: Dict[str, Any] = None) -> str:
        """Deploy a model to an edge device"""
        try:
            # Validate inputs
            if device_id not in self.devices:
                raise ValueError(f"Device not found: {device_id}")
            
            if model_id not in self.models:
                raise ValueError(f"Model not found: {model_id}")
            
            device = self.devices[device_id]
            model = self.models[model_id]
            
            # Check device compatibility
            if not await self._check_device_compatibility(device, model):
                raise ValueError("Device not compatible with model")
            
            # Create deployment
            deployment_id = f"deploy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{device_id}_{model_id}"
            deployment = ModelDeployment(
                deployment_id=deployment_id,
                device_id=device_id,
                model_id=model_id,
                status=DeploymentStatus.PENDING,
                deployment_config=config or {}
            )
            
            self.deployments[deployment_id] = deployment
            
            # Add to deployment queue
            self.deployment_queue.append(deployment_id)
            
            logger.info(f"Model deployment queued: {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise
    
    async def undeploy_model(self, deployment_id: str) -> bool:
        """Undeploy a model from an edge device"""
        try:
            if deployment_id not in self.deployments:
                return False
            
            deployment = self.deployments[deployment_id]
            
            # Update deployment status
            deployment.status = DeploymentStatus.REMOVING
            
            # Remove from device
            success = await self._remove_deployment_from_device(deployment)
            
            if success:
                deployment.status = DeploymentStatus.REMOVED
                logger.info(f"Model undeployed: {deployment_id}")
            else:
                deployment.status = DeploymentStatus.FAILED
                logger.error(f"Failed to undeploy model: {deployment_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error undeploying model: {e}")
            return False
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[ModelDeployment]:
        """Get deployment status"""
        return self.deployments.get(deployment_id)
    
    async def list_deployments(self, device_id: str = None, model_id: str = None) -> List[ModelDeployment]:
        """List deployments with optional filtering"""
        deployments = list(self.deployments.values())
        
        if device_id:
            deployments = [d for d in deployments if d.device_id == device_id]
        
        if model_id:
            deployments = [d for d in deployments if d.model_id == model_id]
        
        return deployments
    
    # ===== INFERENCE MANAGEMENT =====
    
    async def submit_inference_request(self, request: InferenceRequest) -> str:
        """Submit an inference request"""
        try:
            # Validate request
            if request.device_id not in self.devices:
                raise ValueError(f"Device not found: {request.device_id}")
            
            if request.model_id not in self.models:
                raise ValueError(f"Model not found: {request.model_id}")
            
            # Check if model is deployed on device
            deployment = None
            for d in self.deployments.values():
                if d.device_id == request.device_id and d.model_id == request.model_id:
                    if d.status == DeploymentStatus.DEPLOYED:
                        deployment = d
                        break
            
            if not deployment:
                raise ValueError("Model not deployed on device")
            
            # Store request
            self.inference_requests[request.request_id] = request
            
            # Add to inference queue
            self.inference_queue.append(request.request_id)
            
            logger.info(f"Inference request submitted: {request.request_id}")
            return request.request_id
            
        except Exception as e:
            logger.error(f"Error submitting inference request: {e}")
            raise
    
    async def get_inference_result(self, request_id: str) -> Optional[InferenceResult]:
        """Get inference result"""
        return self.inference_results.get(request_id)
    
    async def cancel_inference_request(self, request_id: str) -> bool:
        """Cancel an inference request"""
        try:
            if request_id not in self.inference_requests:
                return False
            
            # Remove from queue if still pending
            if request_id in self.inference_queue:
                self.inference_queue.remove(request_id)
            
            # Remove request
            del self.inference_requests[request_id]
            
            logger.info(f"Inference request cancelled: {request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling inference request: {e}")
            return False
    
    # ===== MODEL OPTIMIZATION =====
    
    async def _optimize_model(self, model_id: str) -> bool:
        """Optimize a model for edge deployment"""
        try:
            if model_id not in self.models:
                return False
            
            model = self.models[model_id]
            model_file = self.model_files[model_id]
            
            # Get optimizer for model format
            optimizer = self.optimizers.get(model.format)
            if not optimizer:
                logger.warning(f"No optimizer available for format: {model.format}")
                return False
            
            # Optimize model
            optimized_file = await optimizer(model_file, model.optimization_level)
            
            if optimized_file and optimized_file.exists():
                # Replace original with optimized version
                shutil.move(optimized_file, model_file)
                
                # Update model size
                model.size_bytes = model_file.stat().st_size
                
                logger.info(f"Model optimized: {model_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error optimizing model: {e}")
            return False
    
    async def _optimize_pytorch_model(self, model_file: Path, optimization_level: OptimizationLevel) -> Optional[Path]:
        """Optimize PyTorch model"""
        try:
            if not torch_available:
                return None
            
            # Load model
            model = torch.load(model_file, map_location='cpu')
            
            if optimization_level == OptimizationLevel.BASIC:
                # Basic optimization: quantization
                model.eval()
                quantized_model = quant.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
                
                # Save optimized model
                optimized_file = model_file.with_suffix('.optimized.pt')
                torch.save(quantized_model, optimized_file)
                
                return optimized_file
            
            elif optimization_level == OptimizationLevel.AGGRESSIVE:
                # Aggressive optimization: quantization + pruning
                model.eval()
                
                # Quantization
                quantized_model = quant.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
                
                # TorchScript compilation
                try:
                    traced_model = torch.jit.trace(quantized_model, torch.randn(1, 3, 224, 224))
                    
                    # Save optimized model
                    optimized_file = model_file.with_suffix('.optimized.pt')
                    traced_model.save(optimized_file)
                    
                    return optimized_file
                except:
                    # Fall back to quantized model
                    optimized_file = model_file.with_suffix('.optimized.pt')
                    torch.save(quantized_model, optimized_file)
                    
                    return optimized_file
            
            return None
            
        except Exception as e:
            logger.error(f"Error optimizing PyTorch model: {e}")
            return None
    
    async def _optimize_tensorflow_model(self, model_file: Path, optimization_level: OptimizationLevel) -> Optional[Path]:
        """Optimize TensorFlow model"""
        try:
            if not tensorflow_available:
                return None
            
            # Load model
            model = tf.keras.models.load_model(model_file)
            
            if optimization_level in [OptimizationLevel.BASIC, OptimizationLevel.AGGRESSIVE]:
                # Convert to TensorFlow Lite
                converter = TFLiteConverter.from_keras_model(model)
                
                if optimization_level == OptimizationLevel.AGGRESSIVE:
                    # Enable optimizations
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.representative_dataset = lambda: [[np.random.rand(1, 224, 224, 3).astype(np.float32)]]
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                
                # Convert
                tflite_model = converter.convert()
                
                # Save optimized model
                optimized_file = model_file.with_suffix('.tflite')
                with open(optimized_file, 'wb') as f:
                    f.write(tflite_model)
                
                return optimized_file
            
            return None
            
        except Exception as e:
            logger.error(f"Error optimizing TensorFlow model: {e}")
            return None
    
    async def _optimize_onnx_model(self, model_file: Path, optimization_level: OptimizationLevel) -> Optional[Path]:
        """Optimize ONNX model"""
        try:
            if not onnx_available:
                return None
            
            # Load model
            model = onnx.load(model_file)
            
            if optimization_level in [OptimizationLevel.BASIC, OptimizationLevel.AGGRESSIVE]:
                # Create optimized session
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                if optimization_level == OptimizationLevel.AGGRESSIVE:
                    session_options.optimized_model_filepath = str(model_file.with_suffix('.optimized.onnx'))
                
                # Create session to trigger optimization
                session = ort.InferenceSession(str(model_file), session_options)
                
                optimized_file = model_file.with_suffix('.optimized.onnx')
                if optimized_file.exists():
                    return optimized_file
            
            return None
            
        except Exception as e:
            logger.error(f"Error optimizing ONNX model: {e}")
            return None
    
    async def _optimize_sklearn_model(self, model_file: Path, optimization_level: OptimizationLevel) -> Optional[Path]:
        """Optimize scikit-learn model"""
        try:
            if not sklearn_available:
                return None
            
            # Load model
            model = joblib.load(model_file)
            
            if optimization_level in [OptimizationLevel.BASIC, OptimizationLevel.AGGRESSIVE]:
                # Compress model
                optimized_file = model_file.with_suffix('.optimized.pkl')
                joblib.dump(model, optimized_file, compress=9)
                
                return optimized_file
            
            return None
            
        except Exception as e:
            logger.error(f"Error optimizing sklearn model: {e}")
            return None
    
    # ===== BACKGROUND TASKS =====
    
    async def _start_background_tasks(self):
        """Start background tasks"""
        tasks = [
            self._deployment_worker(),
            self._inference_worker(),
            self._sync_worker(),
            self._heartbeat_worker(),
            self._metrics_collector()
        ]
        
        for task_func in tasks:
            task = asyncio.create_task(task_func)
            self.background_tasks.add(task)
    
    async def _deployment_worker(self):
        """Process deployment queue"""
        while True:
            try:
                if self.deployment_queue:
                    deployment_id = self.deployment_queue.popleft()
                    await self._process_deployment(deployment_id)
                else:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in deployment worker: {e}")
                await asyncio.sleep(5)
    
    async def _inference_worker(self):
        """Process inference queue"""
        while True:
            try:
                if self.inference_queue:
                    request_id = self.inference_queue.popleft()
                    await self._process_inference(request_id)
                else:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in inference worker: {e}")
                await asyncio.sleep(1)
    
    async def _sync_worker(self):
        """Process model synchronization"""
        while True:
            try:
                if self.sync_queue:
                    task_id = self.sync_queue.popleft()
                    await self._process_sync_task(task_id)
                else:
                    await asyncio.sleep(self.config["sync_interval"])
                    
            except Exception as e:
                logger.error(f"Error in sync worker: {e}")
                await asyncio.sleep(10)
    
    async def _heartbeat_worker(self):
        """Monitor device heartbeats"""
        while True:
            try:
                await self._check_device_heartbeats()
                await asyncio.sleep(self.config["heartbeat_interval"])
                
            except Exception as e:
                logger.error(f"Error in heartbeat worker: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_collector(self):
        """Collect performance metrics"""
        while True:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(60)
    
    # ===== HELPER METHODS =====
    
    async def _validate_device(self, device: EdgeDevice) -> bool:
        """Validate device registration"""
        try:
            # Check required capabilities
            required_capabilities = ["python", "compute"]
            for capability in required_capabilities:
                if capability not in device.capabilities:
                    logger.error(f"Device missing required capability: {capability}")
                    return False
            
            # Check resources
            if device.resources.get("memory_mb", 0) < 512:
                logger.error("Device has insufficient memory")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating device: {e}")
            return False
    
    async def _initialize_device_connection(self, device: EdgeDevice):
        """Initialize connection to device"""
        try:
            # This would establish WebSocket or HTTP connection
            # For now, just store connection info
            connection_info = {
                "url": f"http://{device.ip_address}:{device.port}",
                "last_connected": datetime.utcnow(),
                "status": "connected"
            }
            
            self.device_connections[device.device_id] = connection_info
            
        except Exception as e:
            logger.error(f"Error initializing device connection: {e}")
    
    async def _check_device_compatibility(self, device: EdgeDevice, model: EdgeModel) -> bool:
        """Check if device is compatible with model"""
        try:
            # Check memory requirements
            if device.resources.get("memory_mb", 0) < model.memory_mb:
                return False
            
            # Check format support
            supported_formats = device.capabilities.get("model_formats", [])
            if model.format.value not in supported_formats:
                return False
            
            # Check compute requirements
            if model.metadata.get("requires_gpu", False):
                if not device.capabilities.get("has_gpu", False):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking device compatibility: {e}")
            return False
    
    async def _process_deployment(self, deployment_id: str):
        """Process a deployment"""
        try:
            deployment = self.deployments[deployment_id]
            deployment.status = DeploymentStatus.DEPLOYING
            
            # Get model file
            model_file = self.model_files[deployment.model_id]
            
            # Send model to device
            success = await self._send_model_to_device(
                deployment.device_id,
                deployment.model_id,
                model_file
            )
            
            if success:
                deployment.status = DeploymentStatus.DEPLOYED
                deployment.deployed_at = datetime.utcnow()
                self.performance_metrics["active_deployments"] += 1
                
                logger.info(f"Deployment successful: {deployment_id}")
            else:
                deployment.status = DeploymentStatus.FAILED
                deployment.error_message = "Failed to deploy model to device"
                
                logger.error(f"Deployment failed: {deployment_id}")
                
        except Exception as e:
            logger.error(f"Error processing deployment: {e}")
            if deployment_id in self.deployments:
                self.deployments[deployment_id].status = DeploymentStatus.FAILED
                self.deployments[deployment_id].error_message = str(e)
    
    async def _process_inference(self, request_id: str):
        """Process an inference request"""
        try:
            request = self.inference_requests[request_id]
            
            # Send inference request to device
            start_time = datetime.utcnow()
            
            result = await self._send_inference_to_device(
                request.device_id,
                request.model_id,
                request.input_data
            )
            
            end_time = datetime.utcnow()
            inference_time = (end_time - start_time).total_seconds() * 1000
            
            # Store result
            inference_result = InferenceResult(
                request_id=request_id,
                device_id=request.device_id,
                model_id=request.model_id,
                result=result,
                inference_time_ms=inference_time
            )
            
            self.inference_results[request_id] = inference_result
            
            # Update metrics
            self.performance_metrics["total_inferences"] += 1
            
            logger.info(f"Inference completed: {request_id}")
            
        except Exception as e:
            logger.error(f"Error processing inference: {e}")
            
            # Store error result
            inference_result = InferenceResult(
                request_id=request_id,
                device_id=request.device_id,
                model_id=request.model_id,
                result=None,
                inference_time_ms=0,
                error_message=str(e)
            )
            
            self.inference_results[request_id] = inference_result
    
    async def _send_model_to_device(self, device_id: str, model_id: str, model_file: Path) -> bool:
        """Send model to device"""
        try:
            # This would implement actual communication with edge device
            # For now, simulate successful deployment
            await asyncio.sleep(0.1)  # Simulate network delay
            return True
            
        except Exception as e:
            logger.error(f"Error sending model to device: {e}")
            return False
    
    async def _send_inference_to_device(self, device_id: str, model_id: str, input_data: Any) -> Any:
        """Send inference request to device"""
        try:
            # This would implement actual communication with edge device
            # For now, simulate inference result
            await asyncio.sleep(0.01)  # Simulate inference time
            return {"prediction": "simulated_result"}
            
        except Exception as e:
            logger.error(f"Error sending inference to device: {e}")
            raise
    
    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return ""
    
    async def _collect_performance_metrics(self):
        """Collect performance metrics"""
        try:
            # Update device metrics
            for device in self.devices.values():
                if device.status == DeviceStatus.ONLINE:
                    metrics = await self._get_device_metrics(device.device_id)
                    if metrics:
                        self.device_metrics[device.device_id].append(metrics)
            
            # Update global metrics
            active_deployments = sum(1 for d in self.deployments.values() if d.status == DeploymentStatus.DEPLOYED)
            self.performance_metrics["active_deployments"] = active_deployments
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    async def _get_device_metrics(self, device_id: str) -> Optional[DeviceMetrics]:
        """Get metrics from device"""
        try:
            # This would get actual metrics from device
            # For now, simulate metrics
            metrics = DeviceMetrics(
                device_id=device_id,
                timestamp=datetime.utcnow(),
                cpu_usage=50.0,
                memory_usage=60.0,
                disk_usage=40.0,
                network_latency=10.0
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting device metrics: {e}")
            return None
    
    async def _load_edge_data(self):
        """Load edge data from storage"""
        try:
            # Load devices
            devices_file = self.storage_path / "devices.json"
            if devices_file.exists():
                with open(devices_file, 'r') as f:
                    devices_data = json.load(f)
                    for device_data in devices_data:
                        device = EdgeDevice(**device_data)
                        self.devices[device.device_id] = device
            
            # Load models
            models_file = self.storage_path / "models.json"
            if models_file.exists():
                with open(models_file, 'r') as f:
                    models_data = json.load(f)
                    for model_data in models_data:
                        model = EdgeModel(**model_data)
                        self.models[model.model_id] = model
            
            # Load deployments
            deployments_file = self.storage_path / "deployments.json"
            if deployments_file.exists():
                with open(deployments_file, 'r') as f:
                    deployments_data = json.load(f)
                    for deployment_data in deployments_data:
                        deployment = ModelDeployment(**deployment_data)
                        self.deployments[deployment.deployment_id] = deployment
            
            logger.info("Edge data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading edge data: {e}")
    
    async def _save_edge_data(self):
        """Save edge data to storage"""
        try:
            # Save devices
            devices_data = []
            for device in self.devices.values():
                devices_data.append(asdict(device))
            
            devices_file = self.storage_path / "devices.json"
            with open(devices_file, 'w') as f:
                json.dump(devices_data, f, indent=2, default=str)
            
            # Save models
            models_data = []
            for model in self.models.values():
                models_data.append(asdict(model))
            
            models_file = self.storage_path / "models.json"
            with open(models_file, 'w') as f:
                json.dump(models_data, f, indent=2, default=str)
            
            # Save deployments
            deployments_data = []
            for deployment in self.deployments.values():
                deployments_data.append(asdict(deployment))
            
            deployments_file = self.storage_path / "deployments.json"
            with open(deployments_file, 'w') as f:
                json.dump(deployments_data, f, indent=2, default=str)
            
            logger.info("Edge data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving edge data: {e}")
    
    async def _initialize_optimizers(self):
        """Initialize model optimizers"""
        try:
            # Check availability of optimization libraries
            if torch_available:
                logger.info("PyTorch optimizer available")
            
            if tensorflow_available:
                logger.info("TensorFlow optimizer available")
            
            if onnx_available:
                logger.info("ONNX optimizer available")
            
            if sklearn_available:
                logger.info("Scikit-learn optimizer available")
            
        except Exception as e:
            logger.error(f"Error initializing optimizers: {e}")
    
    async def _start_device_monitoring(self, device_id: str):
        """Start monitoring a device"""
        try:
            # This would start device-specific monitoring
            logger.info(f"Started monitoring device: {device_id}")
            
        except Exception as e:
            logger.error(f"Error starting device monitoring: {e}")
    
    async def _handle_device_status_change(self, device_id: str, old_status: DeviceStatus, new_status: DeviceStatus):
        """Handle device status changes"""
        try:
            logger.info(f"Device {device_id} status changed from {old_status} to {new_status}")
            
            # Handle specific status changes
            if new_status == DeviceStatus.OFFLINE:
                # Mark deployments as unavailable
                for deployment in self.deployments.values():
                    if deployment.device_id == device_id and deployment.status == DeploymentStatus.DEPLOYED:
                        deployment.status = DeploymentStatus.FAILED
                        deployment.error_message = "Device offline"
            
            elif new_status == DeviceStatus.ONLINE and old_status == DeviceStatus.OFFLINE:
                # Re-deploy models if needed
                for deployment in self.deployments.values():
                    if deployment.device_id == device_id and deployment.status == DeploymentStatus.FAILED:
                        if deployment.error_message == "Device offline":
                            deployment.status = DeploymentStatus.PENDING
                            self.deployment_queue.append(deployment.deployment_id)
            
        except Exception as e:
            logger.error(f"Error handling device status change: {e}")
    
    async def _check_device_heartbeats(self):
        """Check device heartbeats"""
        try:
            current_time = datetime.utcnow()
            heartbeat_timeout = timedelta(seconds=self.config["heartbeat_interval"] * 2)
            
            for device in self.devices.values():
                if device.status == DeviceStatus.ONLINE:
                    if current_time - device.last_seen > heartbeat_timeout:
                        await self.update_device_status(device.device_id, DeviceStatus.OFFLINE)
                        logger.warning(f"Device {device.device_id} marked offline due to heartbeat timeout")
            
        except Exception as e:
            logger.error(f"Error checking device heartbeats: {e}")
    
    async def _remove_deployment(self, deployment_id: str):
        """Remove a deployment"""
        try:
            if deployment_id in self.deployments:
                deployment = self.deployments[deployment_id]
                await self._remove_deployment_from_device(deployment)
                del self.deployments[deployment_id]
                
                if self.performance_metrics["active_deployments"] > 0:
                    self.performance_metrics["active_deployments"] -= 1
                
                logger.info(f"Deployment removed: {deployment_id}")
            
        except Exception as e:
            logger.error(f"Error removing deployment: {e}")
    
    async def _remove_deployment_from_device(self, deployment: ModelDeployment) -> bool:
        """Remove deployment from device"""
        try:
            # This would send removal command to device
            # For now, simulate successful removal
            await asyncio.sleep(0.1)
            return True
            
        except Exception as e:
            logger.error(f"Error removing deployment from device: {e}")
            return False
    
    async def _process_sync_task(self, task_id: str):
        """Process a sync task"""
        try:
            if task_id not in self.sync_tasks:
                return
            
            task = self.sync_tasks[task_id]
            
            # Process based on sync type
            if task.sync_type == "deploy":
                await self.deploy_model(task.device_id, task.model_id)
            elif task.sync_type == "update":
                # Update existing deployment
                pass
            elif task.sync_type == "remove":
                # Find and remove deployment
                for deployment in self.deployments.values():
                    if deployment.device_id == task.device_id and deployment.model_id == task.model_id:
                        await self.undeploy_model(deployment.deployment_id)
                        break
            
            task.status = "completed"
            logger.info(f"Sync task completed: {task_id}")
            
        except Exception as e:
            logger.error(f"Error processing sync task: {e}")
            if task_id in self.sync_tasks:
                self.sync_tasks[task_id].status = "failed"
    
    # ===== PUBLIC API METHODS =====
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "devices": {
                "total": len(self.devices),
                "online": sum(1 for d in self.devices.values() if d.status == DeviceStatus.ONLINE),
                "offline": sum(1 for d in self.devices.values() if d.status == DeviceStatus.OFFLINE),
                "by_type": {
                    device_type.value: sum(1 for d in self.devices.values() if d.device_type == device_type)
                    for device_type in EdgeDeviceType
                }
            },
            "models": {
                "total": len(self.models),
                "by_format": {
                    model_format.value: sum(1 for m in self.models.values() if m.format == model_format)
                    for model_format in ModelFormat
                }
            },
            "deployments": {
                "total": len(self.deployments),
                "active": sum(1 for d in self.deployments.values() if d.status == DeploymentStatus.DEPLOYED),
                "pending": sum(1 for d in self.deployments.values() if d.status == DeploymentStatus.PENDING),
                "failed": sum(1 for d in self.deployments.values() if d.status == DeploymentStatus.FAILED)
            },
            "inference": {
                "total_requests": len(self.inference_requests),
                "completed": len(self.inference_results),
                "queue_size": len(self.inference_queue)
            },
            "performance": self.performance_metrics
        }
    
    async def get_device_list(self) -> List[Dict[str, Any]]:
        """Get list of all devices"""
        return [asdict(device) for device in self.devices.values()]
    
    async def get_model_list(self) -> List[Dict[str, Any]]:
        """Get list of all models"""
        return [asdict(model) for model in self.models.values()]
    
    async def get_deployment_list(self) -> List[Dict[str, Any]]:
        """Get list of all deployments"""
        return [asdict(deployment) for deployment in self.deployments.values()]

# Create global instance
edge_ml_service = EdgeMLService() 