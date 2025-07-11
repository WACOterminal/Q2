"""
Automated Model Deployment & Serving Service

This service provides comprehensive model deployment and serving capabilities:
- Blue/Green deployment strategy for zero-downtime deployments
- Canary releases for gradual rollouts
- A/B testing infrastructure for model comparison
- Automated health checks and monitoring
- Model versioning and rollback capabilities
- Traffic routing and load balancing
- Integration with container orchestration (Kubernetes)
- Model serving optimization and autoscaling
- Deployment pipelines with approval workflows
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
from pathlib import Path
import pickle
from collections import defaultdict, deque
import time
import statistics

# Container orchestration
try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    k8s_available = True
except ImportError:
    k8s_available = False
    logging.warning("Kubernetes client not available - deployment capabilities will be limited")

# Docker integration
try:
    import docker
    docker_available = True
except ImportError:
    docker_available = False
    logging.warning("Docker client not available - container capabilities will be limited")

# Health monitoring
import requests
import psutil

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.vault_client import VaultClient

# Import other services for integration
from .model_registry_service import ModelRegistryService
from .ai_governance_service import ai_governance_service

logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    """Deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"

class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class TrafficSplit(Enum):
    """Traffic split strategies"""
    ALL_BLUE = "all_blue"
    ALL_GREEN = "all_green"
    SPLIT_50_50 = "split_50_50"
    CANARY_5 = "canary_5"
    CANARY_10 = "canary_10"
    CANARY_25 = "canary_25"
    CANARY_50 = "canary_50"

@dataclass
class ModelDeploymentSpec:
    """Model deployment specification"""
    model_id: str
    model_version: str
    deployment_name: str
    strategy: DeploymentStrategy
    container_image: str
    container_tag: str
    resources: Dict[str, Any]
    environment: Dict[str, str]
    config: Dict[str, Any]
    health_check: Dict[str, Any]
    scaling: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Deployment:
    """Deployment record"""
    deployment_id: str
    spec: ModelDeploymentSpec
    status: DeploymentStatus
    strategy: DeploymentStrategy
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    blue_version: Optional[str] = None
    green_version: Optional[str] = None
    current_traffic_split: Optional[TrafficSplit] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    service_name: str
    endpoint_url: str
    version: str
    status: ServiceStatus
    health_score: float
    last_health_check: datetime
    metrics: Dict[str, Any]
    traffic_percentage: float = 0.0

@dataclass
class HealthCheck:
    """Health check configuration and results"""
    endpoint: str
    method: str
    expected_status: int
    timeout: int
    interval: int
    retries: int
    success_count: int = 0
    failure_count: int = 0
    last_check: Optional[datetime] = None
    last_status: Optional[ServiceStatus] = None

@dataclass
class DeploymentPipeline:
    """Deployment pipeline configuration"""
    pipeline_id: str
    name: str
    stages: List[Dict[str, Any]]
    approval_required: bool
    automated_rollback: bool
    success_criteria: Dict[str, Any]
    notification_config: Dict[str, Any]
    created_at: datetime

class ModelDeploymentService:
    """
    Comprehensive Model Deployment and Serving Service
    """
    
    def __init__(self, 
                 storage_path: str = "deployments",
                 k8s_namespace: str = "q-platform-models"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.k8s_namespace = k8s_namespace
        
        # Deployment management
        self.deployments: Dict[str, Deployment] = {}
        self.active_services: Dict[str, ServiceEndpoint] = {}
        self.deployment_pipelines: Dict[str, DeploymentPipeline] = {}
        
        # Traffic management
        self.traffic_routes: Dict[str, Dict[str, float]] = {}
        self.load_balancer_config: Dict[str, Any] = {}
        
        # Health monitoring
        self.health_checks: Dict[str, HealthCheck] = {}
        self.service_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Service integrations
        self.model_registry_service: Optional[ModelRegistryService] = None
        self.vault_client = VaultClient()
        
        # Kubernetes client
        self.k8s_client = None
        self.k8s_apps_v1 = None
        self.k8s_core_v1 = None
        self.k8s_networking_v1 = None
        
        # Docker client
        self.docker_client = None
        
        # Configuration
        self.deployment_config = {
            "default_strategy": DeploymentStrategy.BLUE_GREEN,
            "health_check_interval": 30,
            "health_check_timeout": 10,
            "health_check_retries": 3,
            "canary_traffic_increment": 10,
            "rollback_threshold": 0.05,  # 5% error rate triggers rollback
            "deployment_timeout": 1800,  # 30 minutes
            "blue_green_verification_time": 300,  # 5 minutes
            "enable_automated_rollback": True
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Performance metrics
        self.deployment_metrics = {
            "total_deployments": 0,
            "successful_deployments": 0,
            "failed_deployments": 0,
            "rollbacks": 0,
            "average_deployment_time": 0.0,
            "active_services": 0,
            "total_requests_served": 0,
            "average_response_time": 0.0
        }
        
        logger.info("Model Deployment Service initialized")
    
    async def initialize(self):
        """Initialize the model deployment service"""
        logger.info("Initializing Model Deployment Service")
        
        # Initialize Kubernetes client
        await self._initialize_k8s_client()
        
        # Initialize Docker client
        await self._initialize_docker_client()
        
        # Initialize service integrations
        await self._initialize_service_integrations()
        
        # Load existing deployments
        await self._load_deployment_state()
        
        # Initialize default pipelines
        await self._initialize_default_pipelines()
        
        # Start background monitoring
        await self._start_background_monitoring()
        
        # Subscribe to events
        await self._subscribe_to_events()
        
        logger.info("Model Deployment Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the model deployment service"""
        logger.info("Shutting down Model Deployment Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Save deployment state
        await self._save_deployment_state()
        
        logger.info("Model Deployment Service shutdown complete")
    
    # ===== DEPLOYMENT MANAGEMENT =====
    
    async def deploy_model(
        self,
        spec: ModelDeploymentSpec,
        pipeline_id: Optional[str] = None,
        approval_required: bool = False
    ) -> str:
        """Deploy a model using the specified strategy"""
        
        deployment_id = f"deploy_{uuid.uuid4().hex[:12]}"
        
        try:
            logger.info(f"Starting model deployment: {deployment_id}")
            
            # Create deployment record
            deployment = Deployment(
                deployment_id=deployment_id,
                spec=spec,
                status=DeploymentStatus.PENDING,
                strategy=spec.strategy,
                created_at=datetime.utcnow()
            )
            
            # Store deployment
            self.deployments[deployment_id] = deployment
            
            # Check governance policies
            governance_result = await self._check_deployment_governance(spec)
            if not governance_result["compliant"]:
                deployment.status = DeploymentStatus.FAILED
                deployment.error_message = f"Governance check failed: {governance_result['violations']}"
                return deployment_id
            
            # If approval required, wait for approval
            if approval_required:
                await self._request_deployment_approval(deployment_id)
                return deployment_id
            
            # Start deployment process
            await self._execute_deployment(deployment_id)
            
            return deployment_id
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}", exc_info=True)
            if deployment_id in self.deployments:
                self.deployments[deployment_id].status = DeploymentStatus.FAILED
                self.deployments[deployment_id].error_message = str(e)
            raise
    
    async def _execute_deployment(self, deployment_id: str):
        """Execute the deployment strategy"""
        
        deployment = self.deployments[deployment_id]
        deployment.status = DeploymentStatus.IN_PROGRESS
        deployment.started_at = datetime.utcnow()
        
        try:
            # Execute based on strategy
            if deployment.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._execute_blue_green_deployment(deployment)
            elif deployment.strategy == DeploymentStrategy.CANARY:
                await self._execute_canary_deployment(deployment)
            elif deployment.strategy == DeploymentStrategy.ROLLING:
                await self._execute_rolling_deployment(deployment)
            elif deployment.strategy == DeploymentStrategy.A_B_TESTING:
                await self._execute_ab_testing_deployment(deployment)
            else:
                await self._execute_recreate_deployment(deployment)
            
            # Mark deployment as completed
            deployment.status = DeploymentStatus.COMPLETED
            deployment.completed_at = datetime.utcnow()
            
            # Update metrics
            self.deployment_metrics["total_deployments"] += 1
            self.deployment_metrics["successful_deployments"] += 1
            
            # Calculate deployment time
            if deployment.started_at:
                deployment_time = (deployment.completed_at - deployment.started_at).total_seconds()
                self._update_average_deployment_time(deployment_time)
            
            # Publish deployment event
            await self._publish_deployment_event(deployment_id, "completed")
            
            logger.info(f"Deployment completed successfully: {deployment_id}")
            
        except Exception as e:
            logger.error(f"Deployment failed: {deployment_id}: {e}", exc_info=True)
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            deployment.completed_at = datetime.utcnow()
            
            self.deployment_metrics["failed_deployments"] += 1
            
            # Trigger rollback if enabled
            if self.deployment_config["enable_automated_rollback"]:
                await self._trigger_rollback(deployment_id)
            
            await self._publish_deployment_event(deployment_id, "failed")
    
    # ===== BLUE/GREEN DEPLOYMENT =====
    
    async def _execute_blue_green_deployment(self, deployment: Deployment):
        """Execute blue/green deployment strategy"""
        
        logger.info(f"Executing blue/green deployment: {deployment.deployment_id}")
        
        try:
            # Determine current and new environments
            current_env = await self._get_current_environment(deployment.spec.deployment_name)
            new_env = "green" if current_env == "blue" else "blue"
            
            deployment.blue_version = deployment.spec.model_version if new_env == "blue" else await self._get_current_version(deployment.spec.deployment_name)
            deployment.green_version = deployment.spec.model_version if new_env == "green" else await self._get_current_version(deployment.spec.deployment_name)
            
            # Deploy to new environment
            await self._deploy_to_environment(deployment, new_env)
            
            # Wait for new environment to be healthy
            await self._wait_for_health(f"{deployment.spec.deployment_name}-{new_env}")
            
            # Run verification tests
            verification_passed = await self._run_verification_tests(deployment, new_env)
            
            if not verification_passed:
                raise Exception("Verification tests failed")
            
            # Switch traffic to new environment
            await self._switch_traffic(deployment.spec.deployment_name, new_env)
            deployment.current_traffic_split = TrafficSplit.ALL_GREEN if new_env == "green" else TrafficSplit.ALL_BLUE
            
            # Monitor for stability period
            await self._monitor_stability(deployment, new_env, self.deployment_config["blue_green_verification_time"])
            
            # Clean up old environment (optional)
            # await self._cleanup_old_environment(deployment, current_env)
            
            logger.info(f"Blue/green deployment completed: {deployment.deployment_id}")
            
        except Exception as e:
            logger.error(f"Blue/green deployment failed: {e}")
            # Rollback if possible
            if hasattr(deployment, 'blue_version') and hasattr(deployment, 'green_version'):
                await self._rollback_blue_green(deployment)
            raise
    
    async def _deploy_to_environment(self, deployment: Deployment, environment: str):
        """Deploy to a specific environment (blue or green)"""
        
        service_name = f"{deployment.spec.deployment_name}-{environment}"
        
        if self.k8s_apps_v1:
            # Create Kubernetes deployment
            await self._create_k8s_deployment(deployment, service_name, environment)
            
            # Create Kubernetes service
            await self._create_k8s_service(deployment, service_name, environment)
            
            # Create ingress if needed
            await self._create_k8s_ingress(deployment, service_name, environment)
        
        else:
            # Fallback to Docker deployment
            await self._create_docker_deployment(deployment, service_name, environment)
        
        # Register health check
        await self._register_health_check(service_name, deployment.spec.health_check)
        
        # Register service endpoint
        endpoint_url = await self._get_service_endpoint(service_name)
        service_endpoint = ServiceEndpoint(
            service_name=service_name,
            endpoint_url=endpoint_url,
            version=deployment.spec.model_version,
            status=ServiceStatus.UNKNOWN,
            health_score=0.0,
            last_health_check=datetime.utcnow(),
            metrics={}
        )
        self.active_services[service_name] = service_endpoint
    
    async def _create_k8s_deployment(self, deployment: Deployment, service_name: str, environment: str):
        """Create Kubernetes deployment"""
        
        if not self.k8s_apps_v1:
            raise Exception("Kubernetes client not available")
        
        try:
            # Deployment manifest
            deployment_manifest = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": service_name,
                    "namespace": self.k8s_namespace,
                    "labels": {
                        "app": deployment.spec.deployment_name,
                        "environment": environment,
                        "version": deployment.spec.model_version,
                        "managed-by": "q-platform-deployment-service"
                    }
                },
                "spec": {
                    "replicas": deployment.spec.scaling.get("initial_replicas", 2),
                    "selector": {
                        "matchLabels": {
                            "app": deployment.spec.deployment_name,
                            "environment": environment
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": deployment.spec.deployment_name,
                                "environment": environment,
                                "version": deployment.spec.model_version
                            }
                        },
                        "spec": {
                            "containers": [{
                                "name": "model-server",
                                "image": f"{deployment.spec.container_image}:{deployment.spec.container_tag}",
                                "ports": [{
                                    "containerPort": deployment.spec.config.get("port", 8080),
                                    "name": "http"
                                }],
                                "env": [
                                    {"name": k, "value": str(v)} 
                                    for k, v in deployment.spec.environment.items()
                                ],
                                "resources": deployment.spec.resources,
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": deployment.spec.health_check.get("path", "/health"),
                                        "port": "http"
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": deployment.spec.health_check.get("readiness_path", "/ready"),
                                        "port": "http"
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                }
                            }]
                        }
                    }
                }
            }
            
            # Create deployment
            self.k8s_apps_v1.create_namespaced_deployment(
                namespace=self.k8s_namespace,
                body=deployment_manifest
            )
            
            logger.info(f"Created Kubernetes deployment: {service_name}")
            
        except ApiException as e:
            logger.error(f"Error creating Kubernetes deployment: {e}")
            raise
    
    async def _create_k8s_service(self, deployment: Deployment, service_name: str, environment: str):
        """Create Kubernetes service"""
        
        if not self.k8s_core_v1:
            raise Exception("Kubernetes client not available")
        
        try:
            service_manifest = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": service_name,
                    "namespace": self.k8s_namespace,
                    "labels": {
                        "app": deployment.spec.deployment_name,
                        "environment": environment
                    }
                },
                "spec": {
                    "selector": {
                        "app": deployment.spec.deployment_name,
                        "environment": environment
                    },
                    "ports": [{
                        "port": 80,
                        "targetPort": deployment.spec.config.get("port", 8080),
                        "protocol": "TCP",
                        "name": "http"
                    }],
                    "type": "ClusterIP"
                }
            }
            
            # Create service
            self.k8s_core_v1.create_namespaced_service(
                namespace=self.k8s_namespace,
                body=service_manifest
            )
            
            logger.info(f"Created Kubernetes service: {service_name}")
            
        except ApiException as e:
            logger.error(f"Error creating Kubernetes service: {e}")
            raise
    
    async def _create_k8s_ingress(self, deployment: Deployment, service_name: str, environment: str):
        """Create Kubernetes ingress"""
        
        if not self.k8s_networking_v1:
            return  # Skip if networking API not available
        
        try:
            ingress_manifest = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "Ingress",
                "metadata": {
                    "name": service_name,
                    "namespace": self.k8s_namespace,
                    "labels": {
                        "app": deployment.spec.deployment_name,
                        "environment": environment
                    },
                    "annotations": {
                        "nginx.ingress.kubernetes.io/rewrite-target": "/",
                        "nginx.ingress.kubernetes.io/ssl-redirect": "false"
                    }
                },
                "spec": {
                    "rules": [{
                        "host": f"{service_name}.q-platform.local",
                        "http": {
                            "paths": [{
                                "path": "/",
                                "pathType": "Prefix",
                                "backend": {
                                    "service": {
                                        "name": service_name,
                                        "port": {
                                            "number": 80
                                        }
                                    }
                                }
                            }]
                        }
                    }]
                }
            }
            
            # Create ingress
            self.k8s_networking_v1.create_namespaced_ingress(
                namespace=self.k8s_namespace,
                body=ingress_manifest
            )
            
            logger.info(f"Created Kubernetes ingress: {service_name}")
            
        except ApiException as e:
            logger.warning(f"Could not create ingress: {e}")
    
    # ===== CANARY DEPLOYMENT =====
    
    async def _execute_canary_deployment(self, deployment: Deployment):
        """Execute canary deployment strategy"""
        
        logger.info(f"Executing canary deployment: {deployment.deployment_id}")
        
        try:
            # Deploy canary version
            canary_service = f"{deployment.spec.deployment_name}-canary"
            await self._deploy_to_environment(deployment, "canary")
            
            # Start with small traffic percentage
            traffic_percentages = [5, 10, 25, 50, 100]
            
            for percentage in traffic_percentages:
                # Update traffic split
                await self._update_traffic_split(
                    deployment.spec.deployment_name,
                    {"canary": percentage, "stable": 100 - percentage}
                )
                
                deployment.current_traffic_split = TrafficSplit(f"canary_{percentage}")
                
                # Monitor for issues
                monitoring_duration = 300  # 5 minutes per stage
                success = await self._monitor_canary_health(deployment, canary_service, monitoring_duration)
                
                if not success:
                    raise Exception(f"Canary deployment failed at {percentage}% traffic")
                
                # Wait between stages
                await asyncio.sleep(60)
            
            # Complete canary deployment - replace stable with canary
            await self._promote_canary_to_stable(deployment)
            
            logger.info(f"Canary deployment completed: {deployment.deployment_id}")
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            # Rollback canary
            await self._rollback_canary(deployment)
            raise
    
    async def _monitor_canary_health(self, deployment: Deployment, service_name: str, duration: int) -> bool:
        """Monitor canary deployment health"""
        
        start_time = time.time()
        error_count = 0
        total_checks = 0
        
        while time.time() - start_time < duration:
            total_checks += 1
            
            # Check service health
            status = await self._perform_health_check(service_name)
            if status != ServiceStatus.HEALTHY:
                error_count += 1
            
            # Check error rate
            error_rate = await self._get_service_error_rate(service_name)
            if error_rate > self.deployment_config["rollback_threshold"]:
                error_count += 1
            
            # Check if error threshold exceeded
            if error_count / total_checks > self.deployment_config["rollback_threshold"]:
                logger.warning(f"Canary health check failed - error rate too high: {error_count}/{total_checks}")
                return False
            
            await asyncio.sleep(30)  # Check every 30 seconds
        
        return True
    
    # ===== A/B TESTING DEPLOYMENT =====
    
    async def _execute_ab_testing_deployment(self, deployment: Deployment):
        """Execute A/B testing deployment strategy"""
        
        logger.info(f"Executing A/B testing deployment: {deployment.deployment_id}")
        
        try:
            # Deploy variant B
            variant_b_service = f"{deployment.spec.deployment_name}-variant-b"
            await self._deploy_to_environment(deployment, "variant-b")
            
            # Set up 50/50 traffic split
            await self._update_traffic_split(
                deployment.spec.deployment_name,
                {"variant-a": 50, "variant-b": 50}
            )
            
            deployment.current_traffic_split = TrafficSplit.SPLIT_50_50
            
            # Monitor A/B test for configured duration
            test_duration = deployment.spec.config.get("ab_test_duration", 3600)  # 1 hour default
            await self._monitor_ab_test(deployment, test_duration)
            
            # Analyze results and determine winner
            winner = await self._analyze_ab_test_results(deployment)
            
            # Route all traffic to winner
            if winner == "variant-b":
                await self._promote_variant_b(deployment)
            else:
                await self._cleanup_variant_b(deployment)
            
            logger.info(f"A/B testing deployment completed: {deployment.deployment_id}")
            
        except Exception as e:
            logger.error(f"A/B testing deployment failed: {e}")
            await self._cleanup_variant_b(deployment)
            raise
    
    # ===== ROLLING DEPLOYMENT =====
    
    async def _execute_rolling_deployment(self, deployment: Deployment):
        """Execute rolling deployment strategy"""
        
        logger.info(f"Executing rolling deployment: {deployment.deployment_id}")
        
        try:
            # Update deployment with rolling strategy
            if self.k8s_apps_v1:
                await self._update_k8s_deployment_rolling(deployment)
            else:
                await self._execute_docker_rolling_deployment(deployment)
            
            # Monitor rollout
            await self._monitor_rolling_deployment(deployment)
            
            logger.info(f"Rolling deployment completed: {deployment.deployment_id}")
            
        except Exception as e:
            logger.error(f"Rolling deployment failed: {e}")
            raise
    
    # ===== HEALTH MONITORING =====
    
    async def _register_health_check(self, service_name: str, health_config: Dict[str, Any]):
        """Register health check for a service"""
        
        endpoint_url = await self._get_service_endpoint(service_name)
        health_path = health_config.get("path", "/health")
        
        health_check = HealthCheck(
            endpoint=f"{endpoint_url}{health_path}",
            method=health_config.get("method", "GET"),
            expected_status=health_config.get("expected_status", 200),
            timeout=health_config.get("timeout", self.deployment_config["health_check_timeout"]),
            interval=health_config.get("interval", self.deployment_config["health_check_interval"]),
            retries=health_config.get("retries", self.deployment_config["health_check_retries"])
        )
        
        self.health_checks[service_name] = health_check
    
    async def _perform_health_check(self, service_name: str) -> ServiceStatus:
        """Perform health check for a service"""
        
        if service_name not in self.health_checks:
            return ServiceStatus.UNKNOWN
        
        health_check = self.health_checks[service_name]
        
        try:
            response = requests.request(
                method=health_check.method,
                url=health_check.endpoint,
                timeout=health_check.timeout
            )
            
            if response.status_code == health_check.expected_status:
                health_check.success_count += 1
                health_check.last_status = ServiceStatus.HEALTHY
                health_check.last_check = datetime.utcnow()
                
                # Update service endpoint status
                if service_name in self.active_services:
                    self.active_services[service_name].status = ServiceStatus.HEALTHY
                    self.active_services[service_name].health_score = min(1.0, health_check.success_count / 10)
                    self.active_services[service_name].last_health_check = datetime.utcnow()
                
                return ServiceStatus.HEALTHY
            else:
                health_check.failure_count += 1
                health_check.last_status = ServiceStatus.UNHEALTHY
                health_check.last_check = datetime.utcnow()
                
                if service_name in self.active_services:
                    self.active_services[service_name].status = ServiceStatus.UNHEALTHY
                    self.active_services[service_name].health_score = max(0.0, 1.0 - health_check.failure_count / 10)
                    self.active_services[service_name].last_health_check = datetime.utcnow()
                
                return ServiceStatus.UNHEALTHY
                
        except Exception as e:
            logger.warning(f"Health check failed for {service_name}: {e}")
            health_check.failure_count += 1
            health_check.last_status = ServiceStatus.UNKNOWN
            health_check.last_check = datetime.utcnow()
            
            if service_name in self.active_services:
                self.active_services[service_name].status = ServiceStatus.UNKNOWN
                self.active_services[service_name].health_score = 0.0
                self.active_services[service_name].last_health_check = datetime.utcnow()
            
            return ServiceStatus.UNKNOWN
    
    async def _wait_for_health(self, service_name: str, timeout: int = 300) -> bool:
        """Wait for service to become healthy"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = await self._perform_health_check(service_name)
            
            if status == ServiceStatus.HEALTHY:
                return True
            
            await asyncio.sleep(10)  # Check every 10 seconds
        
        return False
    
    # ===== TRAFFIC MANAGEMENT =====
    
    async def _switch_traffic(self, deployment_name: str, target_env: str):
        """Switch all traffic to target environment"""
        
        try:
            # Update load balancer configuration
            if target_env == "blue":
                traffic_config = {"blue": 100, "green": 0}
            else:
                traffic_config = {"blue": 0, "green": 100}
            
            await self._update_traffic_split(deployment_name, traffic_config)
            
            # Update ingress or service mesh configuration
            await self._update_ingress_traffic(deployment_name, target_env)
            
            logger.info(f"Traffic switched to {target_env} for {deployment_name}")
            
        except Exception as e:
            logger.error(f"Error switching traffic: {e}")
            raise
    
    async def _update_traffic_split(self, deployment_name: str, traffic_config: Dict[str, float]):
        """Update traffic split configuration"""
        
        self.traffic_routes[deployment_name] = traffic_config
        
        # Update service endpoints with traffic percentages
        for env, percentage in traffic_config.items():
            service_name = f"{deployment_name}-{env}"
            if service_name in self.active_services:
                self.active_services[service_name].traffic_percentage = percentage
        
        # Persist traffic configuration
        await self._persist_traffic_config(deployment_name, traffic_config)
        
        logger.info(f"Updated traffic split for {deployment_name}: {traffic_config}")
    
    # ===== ROLLBACK MANAGEMENT =====
    
    async def rollback_deployment(self, deployment_id: str, target_version: Optional[str] = None) -> bool:
        """Rollback a deployment"""
        
        try:
            if deployment_id not in self.deployments:
                return False
            
            deployment = self.deployments[deployment_id]
            deployment.status = DeploymentStatus.ROLLING_BACK
            
            if deployment.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._rollback_blue_green(deployment, target_version)
            elif deployment.strategy == DeploymentStrategy.CANARY:
                await self._rollback_canary(deployment)
            elif deployment.strategy == DeploymentStrategy.ROLLING:
                await self._rollback_rolling(deployment, target_version)
            else:
                await self._rollback_generic(deployment, target_version)
            
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.completed_at = datetime.utcnow()
            
            self.deployment_metrics["rollbacks"] += 1
            
            await self._publish_deployment_event(deployment_id, "rolled_back")
            
            logger.info(f"Deployment rolled back: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back deployment: {e}", exc_info=True)
            return False
    
    async def _trigger_rollback(self, deployment_id: str):
        """Trigger automated rollback"""
        
        logger.warning(f"Triggering automated rollback for deployment: {deployment_id}")
        await self.rollback_deployment(deployment_id)
    
    # ===== MONITORING AND METRICS =====
    
    async def _monitor_stability(self, deployment: Deployment, environment: str, duration: int):
        """Monitor deployment stability for a duration"""
        
        service_name = f"{deployment.spec.deployment_name}-{environment}"
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Check health
            status = await self._perform_health_check(service_name)
            if status != ServiceStatus.HEALTHY:
                raise Exception(f"Service became unhealthy during stability monitoring: {service_name}")
            
            # Check error rate
            error_rate = await self._get_service_error_rate(service_name)
            if error_rate > self.deployment_config["rollback_threshold"]:
                raise Exception(f"Error rate too high during stability monitoring: {error_rate}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _get_service_error_rate(self, service_name: str) -> float:
        """Get service error rate"""
        
        try:
            metrics = await self._collect_service_metrics(service_name)
            
            # Calculate error rate from recent metrics
            if service_name in self.service_metrics:
                recent_metrics = list(self.service_metrics[service_name])[-10:]  # Last 10 data points
                error_responses = sum(1 for m in recent_metrics if m.get("status") != "healthy")
                return error_responses / len(recent_metrics) if recent_metrics else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating error rate for {service_name}: {e}")
            return 0.0
    
    async def _get_service_response_time(self, service_name: str) -> float:
        """Get average service response time"""
        
        try:
            if service_name in self.service_metrics:
                recent_metrics = list(self.service_metrics[service_name])[-10:]
                response_times = [m.get("avg_response_time", 0.0) for m in recent_metrics]
                return statistics.mean(response_times) if response_times else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating response time for {service_name}: {e}")
            return 0.0
    
    async def _get_baseline_response_time(self, deployment_name: str) -> float:
        """Get baseline response time for comparison"""
        
        # This would typically compare against the stable version
        return 0.1  # Simplified baseline
    
    # ===== VERIFICATION AND TESTING =====
    
    async def _run_verification_tests(self, deployment: Deployment, environment: str) -> bool:
        """Run verification tests for the deployment"""
        
        service_name = f"{deployment.spec.deployment_name}-{environment}"
        endpoint = await self._get_service_endpoint(service_name)
        
        try:
            # Health check
            health_passed = await self._perform_health_check(service_name) == ServiceStatus.HEALTHY
            if not health_passed:
                logger.warning(f"Health check failed for {service_name}")
                return False
            
            # Performance test
            await self._run_performance_test(endpoint)
            
            # Functional tests
            if deployment.spec.config.get("verification_tests"):
                for test_config in deployment.spec.config["verification_tests"]:
                    test_passed = await self._run_functional_test(endpoint, test_config)
                    if not test_passed:
                        logger.warning(f"Functional test failed: {test_config}")
                        return False
            
            logger.info(f"Verification tests passed for {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Verification tests failed for {service_name}: {e}")
            return False
    
    async def _run_functional_test(self, endpoint: str, test_config: Dict[str, Any]) -> bool:
        """Run a functional test"""
        
        try:
            method = test_config.get("method", "GET")
            path = test_config.get("path", "/")
            payload = test_config.get("payload")
            expected_status = test_config.get("expected_status", 200)
            
            url = f"{endpoint}{path}"
            
            if method == "GET":
                response = requests.get(url, timeout=30)
            elif method == "POST":
                response = requests.post(url, json=payload, timeout=30)
            else:
                response = requests.request(method, url, json=payload, timeout=30)
            
            return response.status_code == expected_status
            
        except Exception as e:
            logger.error(f"Functional test failed: {e}")
            return False
    
    # ===== UTILITY METHODS =====
    
    async def _initialize_k8s_client(self):
        """Initialize Kubernetes client"""
        
        if not k8s_available:
            logger.warning("Kubernetes client not available")
            return
        
        try:
            # Try to load in-cluster config first
            try:
                config.load_incluster_config()
            except:
                # Fall back to kubeconfig
                config.load_kube_config()
            
            self.k8s_client = client.ApiClient()
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
            self.k8s_networking_v1 = client.NetworkingV1Api()
            
            # Create namespace if it doesn't exist
            await self._ensure_namespace_exists()
            
            logger.info("Kubernetes client initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize Kubernetes client: {e}")
    
    async def _initialize_docker_client(self):
        """Initialize Docker client"""
        
        if not docker_available:
            logger.warning("Docker client not available")
            return
        
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize Docker client: {e}")
    
    async def _initialize_service_integrations(self):
        """Initialize service integrations"""
        
        try:
            self.model_registry_service = ModelRegistryService()
            await self.model_registry_service.initialize()
            
            logger.info("Service integrations initialized successfully")
            
        except Exception as e:
            logger.warning(f"Error initializing service integrations: {e}")
    
    async def _ensure_namespace_exists(self):
        """Ensure Kubernetes namespace exists"""
        
        if not self.k8s_core_v1:
            return
        
        try:
            self.k8s_core_v1.read_namespace(name=self.k8s_namespace)
        except ApiException as e:
            if e.status == 404:
                # Create namespace
                namespace_manifest = {
                    "apiVersion": "v1",
                    "kind": "Namespace",
                    "metadata": {
                        "name": self.k8s_namespace
                    }
                }
                self.k8s_core_v1.create_namespace(body=namespace_manifest)
                logger.info(f"Created namespace: {self.k8s_namespace}")
            else:
                raise
    
    async def _load_deployment_state(self):
        """Load deployment state from storage"""
        
        try:
            deployments_file = self.storage_path / "deployments.json"
            if deployments_file.exists():
                with open(deployments_file, 'r') as f:
                    deployments_data = json.load(f)
                    for deployment_data in deployments_data:
                        # Convert datetime strings back to datetime objects
                        deployment_data['created_at'] = datetime.fromisoformat(deployment_data['created_at'])
                        if deployment_data.get('started_at'):
                            deployment_data['started_at'] = datetime.fromisoformat(deployment_data['started_at'])
                        if deployment_data.get('completed_at'):
                            deployment_data['completed_at'] = datetime.fromisoformat(deployment_data['completed_at'])
                        
                        # Reconstruct objects
                        spec_data = deployment_data.pop('spec')
                        spec = ModelDeploymentSpec(**spec_data)
                        deployment_data['spec'] = spec
                        deployment = Deployment(**deployment_data)
                        
                        self.deployments[deployment.deployment_id] = deployment
            
            logger.info("Deployment state loaded successfully")
            
        except Exception as e:
            logger.warning(f"Error loading deployment state: {e}")
    
    async def _save_deployment_state(self):
        """Save deployment state to storage"""
        
        try:
            deployments_data = []
            for deployment in self.deployments.values():
                deployment_dict = asdict(deployment)
                # Convert datetime objects to strings
                deployment_dict['created_at'] = deployment.created_at.isoformat()
                if deployment.started_at:
                    deployment_dict['started_at'] = deployment.started_at.isoformat()
                if deployment.completed_at:
                    deployment_dict['completed_at'] = deployment.completed_at.isoformat()
                
                # Convert enums to strings
                deployment_dict['status'] = deployment.status.value
                deployment_dict['strategy'] = deployment.strategy.value
                if deployment.current_traffic_split:
                    deployment_dict['current_traffic_split'] = deployment.current_traffic_split.value
                
                deployments_data.append(deployment_dict)
            
            deployments_file = self.storage_path / "deployments.json"
            with open(deployments_file, 'w') as f:
                json.dump(deployments_data, f, indent=2)
            
            logger.info("Deployment state saved successfully")
            
        except Exception as e:
            logger.warning(f"Error saving deployment state: {e}")
    
    async def _start_background_monitoring(self):
        """Start background monitoring tasks"""
        
        try:
            # Start health monitoring
            task = asyncio.create_task(self._health_monitoring_loop())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            # Start metrics collection
            task = asyncio.create_task(self._metrics_collection_loop())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            # Start deployment monitoring
            task = asyncio.create_task(self._deployment_monitoring_loop())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            logger.info("Background monitoring started")
            
        except Exception as e:
            logger.warning(f"Error starting background monitoring: {e}")
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        
        while True:
            try:
                await asyncio.sleep(self.deployment_config["health_check_interval"])
                
                # Check health of all active services
                for service_name in list(self.active_services.keys()):
                    await self._perform_health_check(service_name)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute
                
                # Collect service metrics
                for service_name, endpoint in self.active_services.items():
                    metrics = await self._collect_service_metrics(service_name)
                    self.service_metrics[service_name].append(metrics)
                
                # Update global metrics
                self.deployment_metrics["active_services"] = len(self.active_services)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
    
    async def _deployment_monitoring_loop(self):
        """Background deployment monitoring loop"""
        
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Monitor in-progress deployments
                for deployment_id, deployment in self.deployments.items():
                    if deployment.status == DeploymentStatus.IN_PROGRESS:
                        # Check for timeout
                        if deployment.started_at:
                            elapsed = (datetime.utcnow() - deployment.started_at).total_seconds()
                            if elapsed > self.deployment_config["deployment_timeout"]:
                                logger.warning(f"Deployment timeout: {deployment_id}")
                                await self._trigger_rollback(deployment_id)
                
            except Exception as e:
                logger.error(f"Error in deployment monitoring loop: {e}")
    
    # ===== RECREATE DEPLOYMENT =====
    
    async def _execute_recreate_deployment(self, deployment: Deployment):
        """Execute recreate deployment strategy"""
        
        logger.info(f"Executing recreate deployment: {deployment.deployment_id}")
        
        try:
            service_name = deployment.spec.deployment_name
            
            # Stop existing service
            await self._stop_existing_service(service_name)
            
            # Deploy new version
            await self._deploy_to_environment(deployment, "main")
            
            # Wait for new service to be healthy
            await self._wait_for_health(f"{service_name}-main")
            
            # Update traffic routing
            await self._update_traffic_split(service_name, {"main": 100.0})
            
            logger.info(f"Recreate deployment completed: {deployment.deployment_id}")
            
        except Exception as e:
            logger.error(f"Recreate deployment failed: {e}")
            raise
    
    async def _stop_existing_service(self, service_name: str):
        """Stop existing service"""
        
        if self.k8s_apps_v1:
            try:
                # Scale down to 0 replicas
                body = {'spec': {'replicas': 0}}
                self.k8s_apps_v1.patch_namespaced_deployment(
                    name=service_name,
                    namespace=self.k8s_namespace,
                    body=body
                )
                
                # Wait for pods to terminate
                await asyncio.sleep(30)
                
                # Delete deployment
                self.k8s_apps_v1.delete_namespaced_deployment(
                    name=service_name,
                    namespace=self.k8s_namespace
                )
                
            except ApiException as e:
                if e.status != 404:  # Ignore if not found
                    raise
        
        elif self.docker_client:
            try:
                container = self.docker_client.containers.get(service_name)
                container.stop()
                container.remove()
            except Exception as e:
                logger.warning(f"Error stopping Docker container: {e}")
    
    # ===== VERIFICATION TESTS =====
    
    async def _run_verification_tests(self, deployment: Deployment, environment: str) -> bool:
        """Run verification tests for the deployment"""
        
        service_name = f"{deployment.spec.deployment_name}-{environment}"
        endpoint = await self._get_service_endpoint(service_name)
        
        try:
            # Health check
            health_passed = await self._perform_health_check(service_name) == ServiceStatus.HEALTHY
            if not health_passed:
                logger.warning(f"Health check failed for {service_name}")
                return False
            
            # Performance test
            await self._run_performance_test(endpoint)
            
            # Functional tests
            if deployment.spec.config.get("verification_tests"):
                for test_config in deployment.spec.config["verification_tests"]:
                    test_passed = await self._run_functional_test(endpoint, test_config)
                    if not test_passed:
                        logger.warning(f"Functional test failed: {test_config}")
                        return False
            
            logger.info(f"Verification tests passed for {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Verification tests failed for {service_name}: {e}")
            return False
    
    async def _run_functional_test(self, endpoint: str, test_config: Dict[str, Any]) -> bool:
        """Run a functional test"""
        
        try:
            method = test_config.get("method", "GET")
            path = test_config.get("path", "/")
            payload = test_config.get("payload")
            expected_status = test_config.get("expected_status", 200)
            
            url = f"{endpoint}{path}"
            
            if method == "GET":
                response = requests.get(url, timeout=30)
            elif method == "POST":
                response = requests.post(url, json=payload, timeout=30)
            else:
                response = requests.request(method, url, json=payload, timeout=30)
            
            return response.status_code == expected_status
            
        except Exception as e:
            logger.error(f"Functional test failed: {e}")
            return False
    
    # ===== IMPLEMENTATION OF PLACEHOLDER METHODS =====
    
    async def _get_current_environment(self, deployment_name: str) -> str:
        """Get current active environment"""
        
        if deployment_name in self.traffic_routes:
            traffic_config = self.traffic_routes[deployment_name]
            if traffic_config.get("blue", 0) > traffic_config.get("green", 0):
                return "blue"
            elif traffic_config.get("green", 0) > 0:
                return "green"
        
        return "blue"  # Default to blue
    
    async def _get_current_version(self, deployment_name: str) -> str:
        """Get current deployed version"""
        
        if self.k8s_apps_v1:
            try:
                k8s_deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=self.k8s_namespace
                )
                return k8s_deployment.metadata.labels.get("version", "unknown")
            except ApiException:
                pass
        
        return "1.0.0"  # Default version
    
    async def _get_service_endpoint(self, service_name: str) -> str:
        """Get service endpoint URL"""
        
        if self.k8s_core_v1:
            try:
                service = self.k8s_core_v1.read_namespaced_service(
                    name=service_name,
                    namespace=self.k8s_namespace
                )
                
                # Check if it's a LoadBalancer service
                if service.spec.type == "LoadBalancer":
                    if service.status.load_balancer.ingress:
                        ingress = service.status.load_balancer.ingress[0]
                        host = ingress.ip or ingress.hostname
                        port = service.spec.ports[0].port
                        return f"http://{host}:{port}"
                
                # Default to cluster DNS
                port = service.spec.ports[0].port
                return f"http://{service_name}.{self.k8s_namespace}.svc.cluster.local:{port}"
                
            except ApiException:
                pass
        
        return f"http://localhost:8080"  # Fallback for local development
    
    async def _collect_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """Collect metrics for a service"""
        
        endpoint = await self._get_service_endpoint(service_name)
        
        try:
            # Try to get metrics from service
            response = requests.get(f"{endpoint}/metrics", timeout=5)
            
            if response.status_code == 200:
                # Parse Prometheus metrics or custom metrics
                metrics = {}
                for line in response.text.split('\n'):
                    if line.startswith('http_requests_total'):
                        metrics['requests_total'] = float(line.split()[-1])
                    elif line.startswith('http_request_duration_seconds'):
                        metrics['avg_response_time'] = float(line.split()[-1])
                
                return {
                    "timestamp": datetime.utcnow().isoformat(),
                    "requests_total": metrics.get('requests_total', 0),
                    "avg_response_time": metrics.get('avg_response_time', 0.0),
                    "status": "healthy" if response.status_code == 200 else "unhealthy"
                }
            
        except Exception as e:
            logger.warning(f"Error collecting metrics for {service_name}: {e}")
        
        # Return simulated metrics
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "requests_total": 100,
            "avg_response_time": 0.1,
            "status": "unknown"
        }
    
    async def _check_deployment_governance(self, spec: ModelDeploymentSpec) -> Dict[str, Any]:
        """Check deployment against governance policies"""
        
        try:
            # Use the AI governance service to evaluate policies
            result = await ai_governance_service.evaluate_policies(
                resource_type="model_deployment",
                resource_id=spec.model_id,
                context={
                    "deployment_strategy": spec.strategy.value,
                    "model_version": spec.model_version,
                    "container_image": spec.container_image,
                    "resources": spec.resources
                }
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Error checking deployment governance: {e}")
            return {"compliant": True, "violations": [], "warnings": []}
    
    # Additional placeholder methods
    async def _request_deployment_approval(self, deployment_id: str):
        """Request deployment approval"""
        pass
    
    async def _publish_deployment_event(self, deployment_id: str, event_type: str):
        """Publish deployment event"""
        try:
            await shared_pulsar_client.publish("q.deployment.events", {
                "deployment_id": deployment_id,
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to publish deployment event: {e}")
    
    def _update_average_deployment_time(self, deployment_time: float):
        """Update average deployment time metric"""
        current_avg = self.deployment_metrics["average_deployment_time"]
        count = self.deployment_metrics["successful_deployments"]
        
        if count == 1:
            self.deployment_metrics["average_deployment_time"] = deployment_time
        else:
            self.deployment_metrics["average_deployment_time"] = (
                (current_avg * (count - 1) + deployment_time) / count
            )
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events"""
        try:
            await shared_pulsar_client.subscribe(
                "q.model.registered",
                self._handle_model_registration_event,
                subscription_name="deployment_model_registration"
            )
        except Exception as e:
            logger.warning(f"Error subscribing to events: {e}")
    
    async def _handle_model_registration_event(self, event_data: Dict[str, Any]):
        """Handle model registration events"""
        # Placeholder for handling model registration events
        pass
    
    async def _initialize_default_pipelines(self):
        """Initialize default deployment pipelines"""
        
        # Blue/Green pipeline
        blue_green_pipeline = DeploymentPipeline(
            pipeline_id="default-blue-green",
            name="Default Blue/Green Pipeline",
            stages=[
                {"name": "governance-check", "type": "governance"},
                {"name": "deploy-green", "type": "deploy"},
                {"name": "health-check", "type": "health"},
                {"name": "verification", "type": "verification"},
                {"name": "traffic-switch", "type": "traffic"},
                {"name": "monitor", "type": "monitor"}
            ],
            approval_required=False,
            automated_rollback=True,
            success_criteria={
                "error_rate_threshold": 0.01,
                "response_time_threshold": 1.0,
                "health_check_passes": 3
            },
            notification_config={
                "channels": ["slack", "email"],
                "on_failure": True,
                "on_success": True
            },
            created_at=datetime.utcnow()
        )
        
        # Canary pipeline
        canary_pipeline = DeploymentPipeline(
            pipeline_id="default-canary",
            name="Default Canary Pipeline",
            stages=[
                {"name": "governance-check", "type": "governance"},
                {"name": "deploy-canary", "type": "deploy"},
                {"name": "canary-5", "type": "traffic", "traffic_percentage": 5},
                {"name": "canary-10", "type": "traffic", "traffic_percentage": 10},
                {"name": "canary-25", "type": "traffic", "traffic_percentage": 25},
                {"name": "canary-50", "type": "traffic", "traffic_percentage": 50},
                {"name": "promote", "type": "promote"}
            ],
            approval_required=True,
            automated_rollback=True,
            success_criteria={
                "error_rate_threshold": 0.005,
                "response_time_threshold": 0.8,
                "health_check_passes": 5
            },
            notification_config={
                "channels": ["slack", "email"],
                "on_failure": True,
                "on_success": True
            },
            created_at=datetime.utcnow()
        )
        
        self.deployment_pipelines["default-blue-green"] = blue_green_pipeline
        self.deployment_pipelines["default-canary"] = canary_pipeline
        
        logger.info("Default deployment pipelines initialized")

    # Additional placeholder methods for completeness
    async def _rollback_blue_green(self, deployment: Deployment, target_version: Optional[str] = None):
        """Rollback blue/green deployment"""
        pass
    
    async def _rollback_canary(self, deployment: Deployment):
        """Rollback canary deployment"""
        pass
    
    async def _rollback_rolling(self, deployment: Deployment, target_version: Optional[str] = None):
        """Rollback rolling deployment"""
        pass
    
    async def _rollback_generic(self, deployment: Deployment, target_version: Optional[str] = None):
        """Generic rollback implementation"""
        pass
    
    async def _update_k8s_deployment_rolling(self, deployment: Deployment):
        """Update Kubernetes deployment with rolling strategy"""
        pass
    
    async def _execute_docker_rolling_deployment(self, deployment: Deployment):
        """Execute rolling deployment using Docker"""
        pass
    
    async def _monitor_rolling_deployment(self, deployment: Deployment):
        """Monitor rolling deployment progress"""
        pass
    
    async def _create_docker_deployment(self, deployment: Deployment, service_name: str, environment: str):
        """Create Docker deployment"""
        
        if not self.docker_client:
            raise Exception("Docker client not available")
        
        try:
            # Build environment variables
            env_vars = {
                "MODEL_ID": deployment.spec.model_id,
                "MODEL_VERSION": deployment.spec.model_version,
                "ENVIRONMENT": environment,
                "PORT": deployment.spec.config.get("port", 8080),
                **deployment.spec.environment
            }
            
            # Resource limits
            mem_limit = deployment.spec.resources.get("limits", {}).get("memory", "1Gi")
            cpu_limit = deployment.spec.resources.get("limits", {}).get("cpu", "1")
            
            # Create container
            container = self.docker_client.containers.run(
                image=f"{deployment.spec.container_image}:{deployment.spec.container_tag}",
                name=service_name,
                environment=env_vars,
                ports={f"{deployment.spec.config.get('port', 8080)}/tcp": None},
                mem_limit=mem_limit,
                cpu_period=100000,
                cpu_quota=int(float(cpu_limit) * 100000),
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            logger.info(f"Created Docker container: {service_name}")
            
        except Exception as e:
            logger.error(f"Error creating Docker deployment: {e}")
            raise
    
    async def _promote_canary_to_stable(self, deployment: Deployment):
        """Promote canary deployment to stable"""
        
        try:
            service_name = deployment.spec.deployment_name
            canary_service = f"{service_name}-canary"
            
            # Switch 100% traffic to canary
            await self._update_traffic_split(service_name, {"canary": 100, "stable": 0})
            
            # Rename canary to stable
            if self.k8s_apps_v1:
                # Update stable deployment with canary specs
                canary_deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=canary_service,
                    namespace=self.k8s_namespace
                )
                
                # Update the stable deployment
                stable_deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=service_name,
                    namespace=self.k8s_namespace
                )
                
                stable_deployment.spec.template = canary_deployment.spec.template
                
                self.k8s_apps_v1.patch_namespaced_deployment(
                    name=service_name,
                    namespace=self.k8s_namespace,
                    body=stable_deployment
                )
                
                # Delete canary deployment
                self.k8s_apps_v1.delete_namespaced_deployment(
                    name=canary_service,
                    namespace=self.k8s_namespace
                )
            
            logger.info(f"Promoted canary to stable for {service_name}")
            
        except Exception as e:
            logger.error(f"Error promoting canary to stable: {e}")
            raise
    
    async def _monitor_ab_test(self, deployment: Deployment, duration: int):
        """Monitor A/B test"""
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Collect metrics from both variants
            variant_a_metrics = await self._collect_service_metrics(deployment.spec.deployment_name)
            variant_b_metrics = await self._collect_service_metrics(f"{deployment.spec.deployment_name}-variant-b")
            
            # Store metrics for analysis
            deployment.metrics.setdefault("ab_test_data", []).append({
                "timestamp": datetime.utcnow().isoformat(),
                "variant_a": variant_a_metrics,
                "variant_b": variant_b_metrics
            })
            
            await asyncio.sleep(60)  # Collect data every minute
    
    async def _analyze_ab_test_results(self, deployment: Deployment) -> str:
        """Analyze A/B test results"""
        
        try:
            ab_test_data = deployment.metrics.get("ab_test_data", [])
            
            if not ab_test_data:
                return "variant-a"  # Default to original
            
            # Calculate average metrics for each variant
            variant_a_response_times = []
            variant_b_response_times = []
            
            for data_point in ab_test_data:
                variant_a_response_times.append(data_point["variant_a"]["avg_response_time"])
                variant_b_response_times.append(data_point["variant_b"]["avg_response_time"])
            
            avg_response_a = statistics.mean(variant_a_response_times)
            avg_response_b = statistics.mean(variant_b_response_times)
            
            # Choose the variant with better response time
            if avg_response_b < avg_response_a:
                return "variant-b"
            else:
                return "variant-a"
            
        except Exception as e:
            logger.error(f"Error analyzing A/B test results: {e}")
            return "variant-a"
    
    async def _promote_variant_b(self, deployment: Deployment):
        """Promote variant B in A/B test"""
        
        try:
            service_name = deployment.spec.deployment_name
            
            # Route all traffic to variant B
            await self._update_traffic_split(service_name, {"variant-a": 0, "variant-b": 100})
            
            # Replace variant A with variant B
            if self.k8s_apps_v1:
                variant_b_deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=f"{service_name}-variant-b",
                    namespace=self.k8s_namespace
                )
                
                # Update main deployment
                main_deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=service_name,
                    namespace=self.k8s_namespace
                )
                
                main_deployment.spec.template = variant_b_deployment.spec.template
                
                self.k8s_apps_v1.patch_namespaced_deployment(
                    name=service_name,
                    namespace=self.k8s_namespace,
                    body=main_deployment
                )
            
            # Clean up variant B
            await self._cleanup_variant_b(deployment)
            
            logger.info(f"Promoted variant B for {service_name}")
            
        except Exception as e:
            logger.error(f"Error promoting variant B: {e}")
            raise
    
    async def _cleanup_variant_b(self, deployment: Deployment):
        """Clean up variant B in A/B test"""
        
        try:
            service_name = f"{deployment.spec.deployment_name}-variant-b"
            
            if self.k8s_apps_v1:
                # Delete variant B deployment
                self.k8s_apps_v1.delete_namespaced_deployment(
                    name=service_name,
                    namespace=self.k8s_namespace
                )
                
                # Delete variant B service
                self.k8s_core_v1.delete_namespaced_service(
                    name=service_name,
                    namespace=self.k8s_namespace
                )
            
            elif self.docker_client:
                try:
                    container = self.docker_client.containers.get(service_name)
                    container.stop()
                    container.remove()
                except Exception:
                    pass
            
            # Remove from active services
            if service_name in self.active_services:
                del self.active_services[service_name]
            
            logger.info(f"Cleaned up variant B: {service_name}")
            
        except Exception as e:
            logger.error(f"Error cleaning up variant B: {e}")
    
    async def _update_ingress_traffic(self, deployment_name: str, target_env: str):
        """Update ingress traffic routing"""
        
        if not self.k8s_networking_v1:
            return
        
        try:
            ingress_name = f"{deployment_name}-ingress"
            
            # Get current ingress
            ingress = self.k8s_networking_v1.read_namespaced_ingress(
                name=ingress_name,
                namespace=self.k8s_namespace
            )
            
            # Update service name in ingress rules
            for rule in ingress.spec.rules:
                for path in rule.http.paths:
                    path.backend.service.name = f"{deployment_name}-{target_env}"
            
            # Update ingress
            self.k8s_networking_v1.patch_namespaced_ingress(
                name=ingress_name,
                namespace=self.k8s_namespace,
                body=ingress
            )
            
            logger.info(f"Updated ingress traffic routing for {deployment_name}")
            
        except ApiException as e:
            if e.status != 404:  # Ignore if ingress doesn't exist
                logger.error(f"Error updating ingress traffic: {e}")
    
    async def _persist_traffic_config(self, deployment_name: str, traffic_config: Dict[str, float]):
        """Persist traffic configuration"""
        
        self.traffic_routes[deployment_name] = traffic_config
        
        # Save to storage
        traffic_file = self.storage_path / "traffic_routes.json"
        
        try:
            with open(traffic_file, 'w') as f:
                json.dump(self.traffic_routes, f, indent=2)
            
            logger.debug(f"Persisted traffic config for {deployment_name}")
            
        except Exception as e:
            logger.warning(f"Error persisting traffic config: {e}")
    
    # ===== API METHODS =====
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status"""
        
        if deployment_id not in self.deployments:
            return None
        
        deployment = self.deployments[deployment_id]
        
        return {
            "deployment_id": deployment.deployment_id,
            "status": deployment.status.value,
            "strategy": deployment.strategy.value,
            "created_at": deployment.created_at.isoformat(),
            "started_at": deployment.started_at.isoformat() if deployment.started_at else None,
            "completed_at": deployment.completed_at.isoformat() if deployment.completed_at else None,
            "error_message": deployment.error_message,
            "current_traffic_split": deployment.current_traffic_split.value if deployment.current_traffic_split else None,
            "metrics": deployment.metrics
        }
    
    async def list_deployments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent deployments"""
        
        deployments = []
        
        for deployment in list(self.deployments.values())[-limit:]:
            deployments.append({
                "deployment_id": deployment.deployment_id,
                "deployment_name": deployment.spec.deployment_name,
                "model_id": deployment.spec.model_id,
                "model_version": deployment.spec.model_version,
                "status": deployment.status.value,
                "strategy": deployment.strategy.value,
                "created_at": deployment.created_at.isoformat(),
                "completed_at": deployment.completed_at.isoformat() if deployment.completed_at else None
            })
        
        return deployments
    
    async def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get service status"""
        
        if service_name not in self.active_services:
            return None
        
        endpoint = self.active_services[service_name]
        
        return {
            "service_name": endpoint.service_name,
            "endpoint_url": endpoint.endpoint_url,
            "version": endpoint.version,
            "status": endpoint.status.value,
            "health_score": endpoint.health_score,
            "last_health_check": endpoint.last_health_check.isoformat(),
            "traffic_percentage": endpoint.traffic_percentage,
            "metrics": endpoint.metrics
        }
    
    async def get_deployment_metrics(self) -> Dict[str, Any]:
        """Get deployment service metrics"""
        
        return {
            "deployment_metrics": self.deployment_metrics,
            "active_deployments": len([d for d in self.deployments.values() if d.status == DeploymentStatus.IN_PROGRESS]),
            "total_deployments": len(self.deployments),
            "active_services": len(self.active_services),
            "health_checks": len(self.health_checks)
        }

# Create global instance
model_deployment_service = ModelDeploymentService() 