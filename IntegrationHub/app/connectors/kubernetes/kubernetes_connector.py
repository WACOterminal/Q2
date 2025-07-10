# IntegrationHub/app/connectors/kubernetes/kubernetes_connector.py
import logging
from typing import Dict, Any, List, Optional
import base64
import json
from datetime import datetime
import asyncio

try:
    from kubernetes import client, config, watch
    from kubernetes.client.rest import ApiException
    from kubernetes.stream import stream
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    logging.warning("kubernetes library not available. K8s connector will not work.")

from app.models.connector import BaseConnector, ConnectorAction
from app.core.vault_client import vault_client

logger = logging.getLogger(__name__)


class KubernetesConnector(BaseConnector):
    """
    Kubernetes connector for managing K8s resources and operations.
    
    Supported actions:
    - list_pods: List pods with optional namespace and label filters
    - get_pod_logs: Retrieve logs from a specific pod
    - create_deployment: Create a new deployment from manifest
    - update_deployment: Update an existing deployment
    - delete_resource: Delete any K8s resource
    - scale_deployment: Scale deployment replicas
    - rollback_deployment: Rollback to previous deployment version
    - get_events: Get cluster/namespace events
    - execute_command: Execute commands inside pods
    - port_forward: Set up port forwarding to pods
    - get_resource_metrics: Get CPU/memory metrics
    - apply_manifest: Apply YAML/JSON manifests
    - watch_resource: Watch for resource changes
    """
    
    def __init__(self):
        self.v1 = None
        self.apps_v1 = None
        self.batch_v1 = None
        self.networking_v1 = None
        self.rbac_v1 = None
        self.custom_objects = None
        self._initialized = False
        
    @property
    def connector_id(self) -> str:
        return "kubernetes"
        
    async def _initialize_client(self, configuration: Dict[str, Any]):
        """Initialize Kubernetes client with configuration"""
        if self._initialized:
            return
            
        if not KUBERNETES_AVAILABLE:
            raise Exception("Kubernetes library not installed. Run: pip install kubernetes")
            
        try:
            in_cluster = configuration.get("in_cluster", False)
            kubeconfig_path = configuration.get("kubeconfig_path")
            context = configuration.get("context")
            
            if in_cluster:
                # Running inside a Kubernetes cluster
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes configuration")
            else:
                # Running outside cluster
                if kubeconfig_path:
                    config.load_kube_config(
                        config_file=kubeconfig_path,
                        context=context
                    )
                else:
                    config.load_kube_config(context=context)
                logger.info(f"Loaded kubeconfig from {kubeconfig_path or 'default location'}")
                
            # Initialize API clients
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.batch_v1 = client.BatchV1Api()
            self.networking_v1 = client.NetworkingV1Api()
            self.rbac_v1 = client.RbacAuthorizationV1Api()
            self.custom_objects = client.CustomObjectsApi()
            
            # Verify connection
            version_info = self.v1.get_api_resources()
            logger.info(f"Connected to Kubernetes cluster")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to connect to Kubernetes: {e}")
            raise
            
    async def execute(self, action: ConnectorAction, configuration: Dict[str, Any], data_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a Kubernetes action"""
        # Initialize client if needed
        await self._initialize_client(configuration)
        
        # Get namespace from configuration or use default
        self.namespace_default = configuration.get("default_namespace", "default")
        
        action_map = {
            "list_pods": self._list_pods,
            "get_pod_logs": self._get_pod_logs,
            "create_deployment": self._create_deployment,
            "update_deployment": self._update_deployment,
            "delete_resource": self._delete_resource,
            "scale_deployment": self._scale_deployment,
            "rollback_deployment": self._rollback_deployment,
            "get_events": self._get_events,
            "execute_command": self._execute_command,
            "port_forward": self._port_forward,
            "get_resource_metrics": self._get_resource_metrics,
            "apply_manifest": self._apply_manifest,
            "watch_resource": self._watch_resource,
            "create_job": self._create_job,
            "get_job_status": self._get_job_status,
            "create_configmap": self._create_configmap,
            "create_secret": self._create_secret,
            "get_node_info": self._get_node_info,
            "get_service_endpoints": self._get_service_endpoints
        }
        
        if action.action_id not in action_map:
            raise ValueError(f"Unsupported action: {action.action_id}")
            
        # Merge configuration parameters with data context
        parameters = {**configuration, **data_context}
        
        result = await action_map[action.action_id](parameters)
        return result if isinstance(result, dict) else {"result": result}
        
    async def _list_pods(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """List pods with optional filters"""
        namespace = params.get("namespace", self.namespace_default)
        label_selector = params.get("label_selector")
        field_selector = params.get("field_selector")
        
        try:
            if namespace == "all":
                pods = self.v1.list_pod_for_all_namespaces(
                    label_selector=label_selector,
                    field_selector=field_selector
                )
            else:
                pods = self.v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=label_selector,
                    field_selector=field_selector
                )
                
            return [
                {
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "status": pod.status.phase,
                    "ready": all(c.ready for c in pod.status.container_statuses or []),
                    "restarts": sum(c.restart_count for c in pod.status.container_statuses or []),
                    "age": self._calculate_age(pod.metadata.creation_timestamp),
                    "node": pod.spec.node_name,
                    "ip": pod.status.pod_ip,
                    "containers": [c.name for c in pod.spec.containers]
                }
                for pod in pods.items
            ]
            
        except ApiException as e:
            logging.error(f"Failed to list pods: {e}")
            raise
            
    async def _get_pod_logs(self, params: Dict[str, Any]) -> str:
        """Get logs from a pod"""
        pod_name = params["pod_name"]
        namespace = params.get("namespace", self.namespace_default)
        container = params.get("container")
        previous = params.get("previous", False)
        tail_lines = params.get("tail_lines", 100)
        since_seconds = params.get("since_seconds")
        
        try:
            logs = self.v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                container=container,
                previous=previous,
                tail_lines=tail_lines,
                since_seconds=since_seconds
            )
            
            return logs
            
        except ApiException as e:
            logging.error(f"Failed to get pod logs: {e}")
            raise
            
    async def _create_deployment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new deployment"""
        manifest = params["manifest"]
        namespace = params.get("namespace", self.namespace_default)
        
        try:
            # Parse manifest if it's a string
            if isinstance(manifest, str):
                manifest = json.loads(manifest)
                
            # Ensure it's a Deployment
            if manifest.get("kind") != "Deployment":
                raise ValueError("Manifest must be a Deployment")
                
            # Create deployment object
            deployment = client.V1Deployment(
                api_version=manifest.get("apiVersion", "apps/v1"),
                kind="Deployment",
                metadata=client.V1ObjectMeta(**manifest["metadata"]),
                spec=client.V1DeploymentSpec(**manifest["spec"])
            )
            
            # Create deployment
            response = self.apps_v1.create_namespaced_deployment(
                namespace=namespace,
                body=deployment
            )
            
            return {
                "name": response.metadata.name,
                "namespace": response.metadata.namespace,
                "created": response.metadata.creation_timestamp.isoformat(),
                "replicas": response.spec.replicas,
                "status": "created"
            }
            
        except ApiException as e:
            logging.error(f"Failed to create deployment: {e}")
            raise
            
    async def _update_deployment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing deployment"""
        name = params["name"]
        namespace = params.get("namespace", self.namespace_default)
        
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(name, namespace)
            
            # Apply updates
            if "replicas" in params:
                deployment.spec.replicas = params["replicas"]
                
            if "image" in params:
                # Update first container's image (simplified)
                deployment.spec.template.spec.containers[0].image = params["image"]
                
            if "env" in params:
                # Update environment variables
                for container in deployment.spec.template.spec.containers:
                    container.env = [
                        client.V1EnvVar(name=k, value=v)
                        for k, v in params["env"].items()
                    ]
                    
            # Apply the update
            response = self.apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=namespace,
                body=deployment
            )
            
            return {
                "name": response.metadata.name,
                "namespace": response.metadata.namespace,
                "replicas": response.spec.replicas,
                "status": "updated"
            }
            
        except ApiException as e:
            logging.error(f"Failed to update deployment: {e}")
            raise
            
    async def _scale_deployment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scale a deployment to specified replicas"""
        name = params["name"]
        namespace = params.get("namespace", self.namespace_default)
        replicas = params["replicas"]
        
        try:
            # Create scale object
            scale = client.V1Scale(
                metadata=client.V1ObjectMeta(name=name, namespace=namespace),
                spec=client.V1ScaleSpec(replicas=replicas)
            )
            
            # Apply scale
            response = self.apps_v1.patch_namespaced_deployment_scale(
                name=name,
                namespace=namespace,
                body=scale
            )
            
            return {
                "name": name,
                "namespace": namespace,
                "previous_replicas": response.status.replicas,
                "new_replicas": replicas,
                "status": "scaled"
            }
            
        except ApiException as e:
            logging.error(f"Failed to scale deployment: {e}")
            raise
            
    async def _delete_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a Kubernetes resource"""
        kind = params["kind"].lower()
        name = params["name"]
        namespace = params.get("namespace", self.namespace_default)
        
        try:
            delete_options = client.V1DeleteOptions()
            
            if kind == "pod":
                self.v1.delete_namespaced_pod(name, namespace, body=delete_options)
            elif kind == "deployment":
                self.apps_v1.delete_namespaced_deployment(name, namespace, body=delete_options)
            elif kind == "service":
                self.v1.delete_namespaced_service(name, namespace, body=delete_options)
            elif kind == "configmap":
                self.v1.delete_namespaced_config_map(name, namespace, body=delete_options)
            elif kind == "secret":
                self.v1.delete_namespaced_secret(name, namespace, body=delete_options)
            elif kind == "job":
                self.batch_v1.delete_namespaced_job(name, namespace, body=delete_options)
            else:
                raise ValueError(f"Unsupported resource kind: {kind}")
                
            return {
                "kind": kind,
                "name": name,
                "namespace": namespace,
                "status": "deleted"
            }
            
        except ApiException as e:
            logging.error(f"Failed to delete resource: {e}")
            raise
            
    async def _rollback_deployment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback a deployment to previous revision"""
        name = params["name"]
        namespace = params.get("namespace", self.namespace_default)
        revision = params.get("revision")  # If not specified, rollback to previous
        
        try:
            # Get deployment
            deployment = self.apps_v1.read_namespaced_deployment(name, namespace)
            
            # Get replica sets
            replica_sets = self.apps_v1.list_namespaced_replica_set(
                namespace=namespace,
                label_selector=f"app={name}"
            )
            
            # Sort by revision
            sorted_rs = sorted(
                replica_sets.items,
                key=lambda rs: int(rs.metadata.annotations.get("deployment.kubernetes.io/revision", "0")),
                reverse=True
            )
            
            if len(sorted_rs) < 2:
                raise ValueError("No previous revision available for rollback")
                
            # Get target revision (previous if not specified)
            target_rs = sorted_rs[1] if revision is None else None
            
            if revision:
                for rs in sorted_rs:
                    if rs.metadata.annotations.get("deployment.kubernetes.io/revision") == str(revision):
                        target_rs = rs
                        break
                        
            if not target_rs:
                raise ValueError(f"Revision {revision} not found")
                
            # Update deployment with previous template
            deployment.spec.template = target_rs.spec.template
            
            response = self.apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=namespace,
                body=deployment
            )
            
            return {
                "name": name,
                "namespace": namespace,
                "rolled_back_to_revision": target_rs.metadata.annotations.get("deployment.kubernetes.io/revision"),
                "status": "rolled_back"
            }
            
        except ApiException as e:
            logging.error(f"Failed to rollback deployment: {e}")
            raise
            
    async def _get_events(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get Kubernetes events"""
        namespace = params.get("namespace", self.namespace_default)
        field_selector = params.get("field_selector")
        limit = params.get("limit", 50)
        
        try:
            if namespace == "all":
                events = self.v1.list_event_for_all_namespaces(
                    field_selector=field_selector,
                    limit=limit
                )
            else:
                events = self.v1.list_namespaced_event(
                    namespace=namespace,
                    field_selector=field_selector,
                    limit=limit
                )
                
            return [
                {
                    "timestamp": event.first_timestamp.isoformat() if event.first_timestamp else None,
                    "last_timestamp": event.last_timestamp.isoformat() if event.last_timestamp else None,
                    "namespace": event.namespace,
                    "name": event.metadata.name,
                    "reason": event.reason,
                    "message": event.message,
                    "type": event.type,
                    "count": event.count,
                    "source": f"{event.source.component}/{event.source.host}" if event.source else None,
                    "object": f"{event.involved_object.kind}/{event.involved_object.name}"
                }
                for event in sorted(events.items, key=lambda e: e.last_timestamp or e.first_timestamp, reverse=True)
            ]
            
        except ApiException as e:
            logging.error(f"Failed to get events: {e}")
            raise
            
    async def _execute_command(self, params: Dict[str, Any]) -> str:
        """Execute a command inside a pod"""
        pod_name = params["pod_name"]
        namespace = params.get("namespace", self.namespace_default)
        container = params.get("container")
        command = params["command"]
        
        # Ensure command is a list
        if isinstance(command, str):
            command = command.split()
            
        try:
            resp = stream(
                self.v1.connect_get_namespaced_pod_exec,
                pod_name,
                namespace,
                container=container,
                command=command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False
            )
            
            return resp
            
        except ApiException as e:
            logging.error(f"Failed to execute command: {e}")
            raise
            
    async def _port_forward(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set up port forwarding to a pod (returns config, actual forwarding needs separate process)"""
        pod_name = params["pod_name"]
        namespace = params.get("namespace", self.namespace_default)
        ports = params["ports"]  # List of "local:remote" or just "port"
        
        # Parse ports
        port_mappings = []
        for port in ports:
            if ":" in str(port):
                local, remote = str(port).split(":")
                port_mappings.append({"local": int(local), "remote": int(remote)})
            else:
                port_mappings.append({"local": int(port), "remote": int(port)})
                
        # Note: Actual port forwarding requires a separate process/thread
        # This returns the configuration needed
        return {
            "pod_name": pod_name,
            "namespace": namespace,
            "port_mappings": port_mappings,
            "status": "configuration_ready",
            "note": "Use kubectl port-forward or implement forwarding logic"
        }
        
    async def _get_resource_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get resource metrics (requires metrics-server)"""
        namespace = params.get("namespace", self.namespace_default)
        
        try:
            # Get metrics from metrics API
            metrics = self.custom_objects.list_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=namespace,
                plural="pods"
            )
            
            return {
                "namespace": namespace,
                "metrics": [
                    {
                        "pod": item["metadata"]["name"],
                        "containers": [
                            {
                                "name": container["name"],
                                "cpu": container["usage"]["cpu"],
                                "memory": container["usage"]["memory"]
                            }
                            for container in item["containers"]
                        ]
                    }
                    for item in metrics.get("items", [])
                ]
            }
            
        except ApiException as e:
            if e.status == 404:
                return {
                    "error": "Metrics API not available. Ensure metrics-server is installed."
                }
            logging.error(f"Failed to get metrics: {e}")
            raise
            
    async def _apply_manifest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a YAML/JSON manifest"""
        manifest = params["manifest"]
        
        # Parse manifest if string
        if isinstance(manifest, str):
            import yaml
            manifest = yaml.safe_load(manifest)
            
        kind = manifest["kind"]
        api_version = manifest.get("apiVersion", "v1")
        
        # Route to appropriate API based on kind and version
        # This is a simplified version - full implementation would handle all resource types
        
        try:
            if kind == "Deployment" and "apps/" in api_version:
                response = self.apps_v1.create_namespaced_deployment(
                    namespace=manifest["metadata"].get("namespace", "default"),
                    body=manifest
                )
            elif kind == "Service":
                response = self.v1.create_namespaced_service(
                    namespace=manifest["metadata"].get("namespace", "default"),
                    body=manifest
                )
            elif kind == "ConfigMap":
                response = self.v1.create_namespaced_config_map(
                    namespace=manifest["metadata"].get("namespace", "default"),
                    body=manifest
                )
            else:
                # Use dynamic client for other resources
                return {"error": f"Resource kind {kind} not yet implemented"}
                
            return {
                "kind": kind,
                "name": response.metadata.name,
                "namespace": response.metadata.namespace,
                "status": "applied"
            }
            
        except ApiException as e:
            if e.status == 409:  # Already exists, try update
                return await self._update_manifest(manifest)
            logging.error(f"Failed to apply manifest: {e}")
            raise
            
    async def _watch_resource(self, params: Dict[str, Any]) -> Any:
        """Watch for resource changes (returns generator)"""
        kind = params["kind"].lower()
        namespace = params.get("namespace", self.namespace_default)
        timeout_seconds = params.get("timeout_seconds", 60)
        
        w = watch.Watch()
        
        try:
            if kind == "pod":
                return w.stream(
                    self.v1.list_namespaced_pod,
                    namespace=namespace,
                    timeout_seconds=timeout_seconds
                )
            elif kind == "deployment":
                return w.stream(
                    self.apps_v1.list_namespaced_deployment,
                    namespace=namespace,
                    timeout_seconds=timeout_seconds
                )
            else:
                raise ValueError(f"Watch not implemented for kind: {kind}")
                
        except ApiException as e:
            logging.error(f"Failed to watch resource: {e}")
            raise
            
    async def _create_job(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Kubernetes Job"""
        manifest = params["manifest"]
        namespace = params.get("namespace", self.namespace_default)
        
        try:
            if isinstance(manifest, str):
                manifest = json.loads(manifest)
                
            job = client.V1Job(
                api_version="batch/v1",
                kind="Job",
                metadata=client.V1ObjectMeta(**manifest["metadata"]),
                spec=client.V1JobSpec(**manifest["spec"])
            )
            
            response = self.batch_v1.create_namespaced_job(
                namespace=namespace,
                body=job
            )
            
            return {
                "name": response.metadata.name,
                "namespace": response.metadata.namespace,
                "created": response.metadata.creation_timestamp.isoformat(),
                "status": "created"
            }
            
        except ApiException as e:
            logging.error(f"Failed to create job: {e}")
            raise
            
    async def _get_job_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of a Kubernetes Job"""
        name = params["name"]
        namespace = params.get("namespace", self.namespace_default)
        
        try:
            job = self.batch_v1.read_namespaced_job_status(name, namespace)
            
            return {
                "name": name,
                "namespace": namespace,
                "active": job.status.active or 0,
                "succeeded": job.status.succeeded or 0,
                "failed": job.status.failed or 0,
                "conditions": [
                    {
                        "type": c.type,
                        "status": c.status,
                        "reason": c.reason,
                        "message": c.message
                    }
                    for c in job.status.conditions or []
                ],
                "start_time": job.status.start_time.isoformat() if job.status.start_time else None,
                "completion_time": job.status.completion_time.isoformat() if job.status.completion_time else None
            }
            
        except ApiException as e:
            logging.error(f"Failed to get job status: {e}")
            raise
            
    async def _create_configmap(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a ConfigMap"""
        name = params["name"]
        namespace = params.get("namespace", self.namespace_default)
        data = params["data"]
        
        try:
            configmap = client.V1ConfigMap(
                api_version="v1",
                kind="ConfigMap",
                metadata=client.V1ObjectMeta(name=name, namespace=namespace),
                data=data
            )
            
            response = self.v1.create_namespaced_config_map(
                namespace=namespace,
                body=configmap
            )
            
            return {
                "name": response.metadata.name,
                "namespace": response.metadata.namespace,
                "created": response.metadata.creation_timestamp.isoformat(),
                "data_keys": list(response.data.keys()) if response.data else []
            }
            
        except ApiException as e:
            logging.error(f"Failed to create configmap: {e}")
            raise
            
    async def _create_secret(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Secret"""
        name = params["name"]
        namespace = params.get("namespace", self.namespace_default)
        data = params["data"]
        secret_type = params.get("type", "Opaque")
        
        try:
            # Encode data values to base64
            encoded_data = {
                k: base64.b64encode(v.encode()).decode() 
                for k, v in data.items()
            }
            
            secret = client.V1Secret(
                api_version="v1",
                kind="Secret",
                metadata=client.V1ObjectMeta(name=name, namespace=namespace),
                type=secret_type,
                data=encoded_data
            )
            
            response = self.v1.create_namespaced_secret(
                namespace=namespace,
                body=secret
            )
            
            return {
                "name": response.metadata.name,
                "namespace": response.metadata.namespace,
                "type": response.type,
                "created": response.metadata.creation_timestamp.isoformat(),
                "data_keys": list(response.data.keys()) if response.data else []
            }
            
        except ApiException as e:
            logging.error(f"Failed to create secret: {e}")
            raise
            
    async def _get_node_info(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get information about cluster nodes"""
        try:
            nodes = self.v1.list_node()
            
            return [
                {
                    "name": node.metadata.name,
                    "status": next(
                        (c.type for c in node.status.conditions if c.type == "Ready" and c.status == "True"),
                        "NotReady"
                    ),
                    "roles": [
                        label.split("/")[1] 
                        for label in node.metadata.labels 
                        if label.startswith("node-role.kubernetes.io/")
                    ],
                    "version": node.status.node_info.kubelet_version,
                    "os": node.status.node_info.operating_system,
                    "architecture": node.status.node_info.architecture,
                    "capacity": {
                        "cpu": node.status.capacity.get("cpu"),
                        "memory": node.status.capacity.get("memory"),
                        "pods": node.status.capacity.get("pods")
                    },
                    "allocatable": {
                        "cpu": node.status.allocatable.get("cpu"),
                        "memory": node.status.allocatable.get("memory"),
                        "pods": node.status.allocatable.get("pods")
                    }
                }
                for node in nodes.items
            ]
            
        except ApiException as e:
            logging.error(f"Failed to get node info: {e}")
            raise
            
    async def _get_service_endpoints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get service endpoints"""
        name = params["name"]
        namespace = params.get("namespace", self.namespace_default)
        
        try:
            service = self.v1.read_namespaced_service(name, namespace)
            endpoints = self.v1.read_namespaced_endpoints(name, namespace)
            
            return {
                "service": {
                    "name": service.metadata.name,
                    "namespace": service.metadata.namespace,
                    "type": service.spec.type,
                    "cluster_ip": service.spec.cluster_ip,
                    "ports": [
                        {
                            "name": port.name,
                            "port": port.port,
                            "target_port": str(port.target_port),
                            "protocol": port.protocol
                        }
                        for port in service.spec.ports
                    ]
                },
                "endpoints": [
                    {
                        "addresses": [addr.ip for addr in subset.addresses or []],
                        "ports": [
                            {"port": port.port, "protocol": port.protocol}
                            for port in subset.ports or []
                        ]
                    }
                    for subset in endpoints.subsets or []
                ]
            }
            
        except ApiException as e:
            logging.error(f"Failed to get service endpoints: {e}")
            raise
            
    def _calculate_age(self, timestamp) -> str:
        """Calculate age from timestamp"""
        if not timestamp:
            return "unknown"
            
        delta = datetime.now(timestamp.tzinfo) - timestamp
        
        if delta.days > 0:
            return f"{delta.days}d"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60}m"
        else:
            return f"{delta.seconds}s"
            
    async def _update_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing resource from manifest"""
        kind = manifest["kind"]
        name = manifest["metadata"]["name"]
        namespace = manifest["metadata"].get("namespace", "default")
        
        try:
            if kind == "Deployment":
                response = self.apps_v1.patch_namespaced_deployment(
                    name=name,
                    namespace=namespace,
                    body=manifest
                )
            elif kind == "Service":
                response = self.v1.patch_namespaced_service(
                    name=name,
                    namespace=namespace,
                    body=manifest
                )
            elif kind == "ConfigMap":
                response = self.v1.patch_namespaced_config_map(
                    name=name,
                    namespace=namespace,
                    body=manifest
                )
            else:
                return {"error": f"Update for kind {kind} not yet implemented"}
                
            return {
                "kind": kind,
                "name": response.metadata.name,
                "namespace": response.metadata.namespace,
                "status": "updated"
            }
            
        except ApiException as e:
            logging.error(f"Failed to update manifest: {e}")
            raise


# Instantiate a single instance
kubernetes_connector = KubernetesConnector() 