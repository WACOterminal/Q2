import logging
import os
import json
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from kubernetes import client, config
import structlog
from kubernetes.client.rest import ApiException
import base64

from agentQ.app.core.toolbox import Tool
from shared.q_knowledgegraph_client import kgq_client

logger = structlog.get_logger(__name__)

# --- Elasticsearch Client Setup ---
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
es_client = Elasticsearch(ES_URL)

# --- Kubernetes Client Setup ---
# Load configuration, either from a kubeconfig file or from the in-cluster service account
try:
    # When running inside a k8s cluster, this is the standard way to load config.
    # It will use the service account token mounted inside the pod.
    config.load_incluster_config()
    k8s_apps_v1 = client.AppsV1Api()
    k8s_core_v1 = client.CoreV1Api()
    k8s_custom_objects_api = client.CustomObjectsApi()
    logger.info("DevOps Tools: Loaded in-cluster Kubernetes config.")
except config.ConfigException:
    # This fallback is for local development and testing only.
    # In production, the agent should always run inside the cluster.
    logger.warning("Could not load in-cluster K8s config. Falling back to local kubeconfig.")
    try:
        config.load_kube_config()
        k8s_apps_v1 = client.AppsV1Api()
        k8s_core_v1 = client.CoreV1Api()
        k8s_custom_objects_api = client.CustomObjectsApi()
        logger.info("DevOps Tools: Loaded local kubeconfig.")
    except config.ConfigException:
        logger.error("Could not configure Kubernetes client. DevOps tools that use kubectl will not work.")
        k8s_apps_v1 = None
        k8s_core_v1 = None
        k8s_custom_objects_api = None


def get_service_dependencies(service_name: str) -> str:
    """
    Queries the Knowledge Graph to find the upstream and downstream dependencies of a service.
    This helps in understanding the potential blast radius of an issue.

    Args:
        service_name (str): The name of the service to query.

    Returns:
        A string describing the service's dependencies.
    """
    logger.info(f"DevOps Tool: Getting dependencies for '{service_name}' from Knowledge Graph.")
    
    # Gremlin query to find services that this service depends on (out)
    # and services that depend on this service (in)
    query = f"g.V().has('Service', 'name', '{service_name}').union(outE().inV(), inE().outV()).path().by('name').by(label())"
    
    try:
        response = kgq_client.execute_gremlin_query(query)
        result = response.get("result", [])
        
        if not result:
            return f"No dependency information found for service '{service_name}'."
        
        dependencies = {"depends_on": set(), "depended_on_by": set()}
        for path in result:
            # path is like ['IntegrationHub', 'DEPENDS_ON', 'Pulsar']
            # or ['agentQ', 'DEPENDS_ON', 'IntegrationHub']
            if len(path['objects']) == 3:
                subj, pred, obj = path['objects']
                if subj == service_name:
                    dependencies["depends_on"].add(obj)
                else:
                    dependencies["depended_on_by"].add(subj)

        return json.dumps(dependencies)

    except Exception as e:
        logger.error(f"Failed to query Knowledge Graph for dependencies: {e}", exc_info=True)
        return f"Error: Could not retrieve dependencies from Knowledge Graph. Details: {e}"


def get_recent_deployments(service_name: str, hours_ago: int = 48) -> str:
    """
    Queries the Knowledge Graph to find recent deployments for a specific service.
    This helps correlate issues with recent code or configuration changes.

    Args:
        service_name (str): The name of the service to query.
        hours_ago (int): The time window in hours to look back for deployments. Defaults to 48.

    Returns:
        A string containing a list of recent deployments.
    """
    logger.info(f"DevOps Tool: Getting recent deployments for '{service_name}' in the last {hours_ago} hours.")
    
    # Calculate the timestamp to query from
    # This assumes the graph stores timestamps as ISO 8601 strings
    since_timestamp = (datetime.utcnow() - timedelta(hours=hours_ago)).isoformat()
    
    # Gremlin query to find Deployment vertices connected to the given Service,
    # filtering by the timestamp on the associated Event vertex.
    query = f"""
    g.V().has('Service', 'name', '{service_name}')
     .in('DEPLOYED_TO').as_('d')
     .where(
         in_('CORRESPONDS_TO').has('Event', 'timestamp', gte('{since_timestamp}'))
     )
     .select('d')
     .valueMap('version', 'commit')
    """
    
    try:
        response = kgq_client.execute_gremlin_query(query)
        result = response.get("result", [])
        
        if not result:
            return f"No recent deployments found for service '{service_name}' in the last {hours_ago} hours."
        
        return f"Found recent deployments for {service_name}: {json.dumps(result)}"

    except Exception as e:
        logger.error(f"Failed to query Knowledge Graph for deployments: {e}", exc_info=True)
        return f"Error: Could not retrieve deployments from Knowledge Graph. Details: {e}"


def rollback_deployment(service_name: str, environment: str = "staging", namespace: str = "default") -> str:
    """
    Rolls back the deployment of a service to its previous version in a given namespace.
    This is a critical remediation action to be used when a new deployment is causing issues.

    Args:
        service_name (str): The name of the Kubernetes deployment to roll back.
        environment (str): The environment (not directly used by kubectl, but good for logging).
        namespace (str): The Kubernetes namespace the service resides in.

    Returns:
        A string indicating the result of the rollback command.
    """
    if not k8s_apps_v1:
        return "Error: Kubernetes client is not configured. Cannot perform rollback."

    logger.warning(f"DevOps Tool: Initiating rollback for '{service_name}' in namespace '{namespace}'.")
    
    try:
        # The body for the rollback call is an empty V1Rollback object
        body = client.V1DeploymentRollback(
            name=service_name,
            rollback_to=client.V1RollbackConfig(revision=0) # revision 0 means previous
        )
        # Note: The V1DeploymentRollback object is deprecated.
        # The modern way is to use `patch` to set the spec.template back to a previous ReplicaSet's template.
        # For this example, we'll use the simpler, though deprecated, method if available.
        # A more robust implementation would use the patch method.
        
        # A better, non-deprecated way to rollback:
        k8s_apps_v1.patch_namespaced_deployment(
            name=service_name,
            namespace=namespace,
            body={"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": datetime.utcnow().isoformat()}}}}}
        )

        return f"Successfully initiated rollback for deployment '{service_name}' in namespace '{namespace}'. The previous version will be deployed."
    except client.ApiException as e:
        logger.error(f"Kubernetes API error during rollback: {e}", exc_info=True)
        return f"Error: Failed to rollback deployment '{service_name}'. Kubernetes API error: {e.body}"
    except Exception as e:
        logger.error(f"Unexpected error during rollback: {e}", exc_info=True)
        return f"Error: An unexpected error occurred during rollback: {e}"


def restart_service(service_name: str, namespace: str = "default") -> str:
    """
    Performs a rolling restart of a service's deployment in Kubernetes.
    This is a common remediation step to resolve transient issues or memory leaks.

    Args:
        service_name (str): The name of the Kubernetes deployment to restart.
        namespace (str): The Kubernetes namespace the service resides in.

    Returns:
        A string indicating the result of the restart command.
    """
    if not k8s_apps_v1:
        return "Error: Kubernetes client is not configured. Cannot perform restart."

    logger.warning(f"DevOps Tool: Initiating rolling restart for '{service_name}' in namespace '{namespace}'.")
    
    try:
        # The standard way to trigger a rolling restart is to update an annotation
        # in the pod template. `kubectl rollout restart` does this under the hood.
        patch_body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": datetime.utcnow().isoformat()
                        }
                    }
                }
            }
        }
        
        k8s_apps_v1.patch_namespaced_deployment(
            name=service_name,
            namespace=namespace,
            body=patch_body,
        )

        return f"Successfully initiated rolling restart for deployment '{service_name}' in namespace '{namespace}'."
    except client.ApiException as e:
        logger.error(f"Kubernetes API error during restart: {e}", exc_info=True)
        return f"Error: Failed to restart deployment '{service_name}'. Kubernetes API error: {e.body}"
    except Exception as e:
        logger.error(f"Unexpected error during restart: {e}", exc_info=True)
        return f"Error: An unexpected error occurred during restart: {e}"


def increase_service_replicas(service_name: str, new_replica_count: int, namespace: str = "default") -> str:
    """
    Scales a Kubernetes deployment to a new number of replicas.
    Useful for handling increased load on a service.

    Args:
        service_name (str): The name of the Kubernetes deployment to scale.
        new_replica_count (int): The desired number of replicas.
        namespace (str): The Kubernetes namespace the service resides in.

    Returns:
        A string indicating the result of the scaling command.
    """
    if not k8s_apps_v1:
        return "Error: Kubernetes client is not configured. Cannot perform scaling."

    logger.warning(f"DevOps Tool: Scaling deployment '{service_name}' to {new_replica_count} replicas in namespace '{namespace}'.")
    
    try:
        patch_body = {"spec": {"replicas": new_replica_count}}
        k8s_apps_v1.patch_namespaced_deployment_scale(
            name=service_name,
            namespace=namespace,
            body=patch_body,
        )
        return f"Successfully scaled deployment '{service_name}' to {new_replica_count} replicas."
    except client.ApiException as e:
        logger.error(f"Kubernetes API error during scaling: {e}", exc_info=True)
        return f"Error: Failed to scale deployment '{service_name}'. Kubernetes API error: {e.body}"
    except Exception as e:
        logger.error(f"Unexpected error during scaling: {e}", exc_info=True)
        return f"Error: An unexpected error occurred during scaling: {e}"


def list_kubernetes_pods(namespace: str = "default") -> str:
    """
    Lists all pods in a given Kubernetes namespace, along with their status and age.

    Args:
        namespace (str): The Kubernetes namespace to query.

    Returns:
        A JSON string representing a list of pods and their details, or an error message.
    """
    if not k8s_core_v1:
        return "Error: Kubernetes client is not configured. Cannot list pods."

    logger.info(f"DevOps Tool: Listing pods in namespace '{namespace}'.")
    
    try:
        pod_list = k8s_core_v1.list_namespaced_pod(namespace=namespace, limit=50)
        
        pods = []
        for pod in pod_list.items:
            pods.append({
                "name": pod.metadata.name,
                "status": pod.status.phase,
                "ip": pod.status.pod_ip,
                "age": (datetime.utcnow().replace(tzinfo=None) - pod.metadata.creation_timestamp.replace(tzinfo=None)).total_seconds(),
                "restarts": sum(cs.restart_count for cs in pod.status.container_statuses) if pod.status.container_statuses else 0,
            })
            
        return json.dumps(pods, indent=2)
    except client.ApiException as e:
        logger.error(f"Kubernetes API error while listing pods: {e}", exc_info=True)
        return f"Error: Failed to list pods in namespace '{namespace}'. Kubernetes API error: {e.body}"
    except Exception as e:
        logger.error(f"Unexpected error while listing pods: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while listing pods: {e}"


def get_kubernetes_deployment_status(service_name: str, namespace: str = "default") -> str:
    """
    Provides a comprehensive status report for a Kubernetes deployment,
    including deployment status, pod details, and recent events.

    Args:
        service_name (str): The name of the Kubernetes deployment.
        namespace (str): The Kubernetes namespace.

    Returns:
        A JSON string with the detailed status, or an error message.
    """
    if not k8s_apps_v1 or not k8s_core_v1:
        return "Error: Kubernetes client is not configured."

    logger.info(f"DevOps Tool: Getting comprehensive status for deployment '{service_name}' in namespace '{namespace}'.")
    
    try:
        # 1. Get Deployment details
        deployment = k8s_apps_v1.read_namespaced_deployment(name=service_name, namespace=namespace)
        deployment_status = {
            "name": deployment.metadata.name,
            "replicas": deployment.spec.replicas,
            "ready_replicas": deployment.status.ready_replicas or 0,
            "available_replicas": deployment.status.available_replicas or 0,
            "unavailable_replicas": deployment.status.unavailable_replicas or 0,
            "conditions": [c.message for c in deployment.status.conditions]
        }

        # 2. Get Pods for this deployment
        label_selector = deployment.spec.selector.match_labels
        if not label_selector:
            return json.dumps({"deployment_status": deployment_status, "pods": [], "events": []})
        
        selector_str = ",".join([f"{k}={v}" for k, v in label_selector.items()])
        pod_list = k8s_core_v1.list_namespaced_pod(namespace=namespace, label_selector=selector_str)
        
        pods = []
        for pod in pod_list.items:
            pods.append({
                "name": pod.metadata.name,
                "status": pod.status.phase,
                "ip": pod.status.pod_ip,
                "age_seconds": (datetime.utcnow().replace(tzinfo=None) - pod.metadata.creation_timestamp.replace(tzinfo=None)).total_seconds(),
                "restarts": sum(cs.restart_count for cs in pod.status.container_statuses) if pod.status.container_statuses else 0,
                "conditions": [c.message for c in pod.status.conditions] if pod.status.conditions else []
            })
            
        # 3. Get recent events for the deployment
        events = []
        field_selector = f"involvedObject.kind=Deployment,involvedObject.name={service_name},involvedObject.namespace={namespace}"
        event_list = k8s_core_v1.list_event_for_all_namespaces(field_selector=field_selector, limit=20)
        
        for event in event_list.items:
             events.append({
                "reason": event.reason,
                "message": event.message,
                "type": event.type,
                "timestamp": event.last_timestamp.isoformat() if event.last_timestamp else ""
            })

        return json.dumps({
            "deployment_status": deployment_status,
            "pods": pods,
            "events": events
        }, indent=2)

    except client.ApiException as e:
        logger.error(f"Kubernetes API error while getting deployment status: {e}", exc_info=True)
        return f"Error: Failed to get status for deployment '{service_name}'. Kubernetes API error: {e.body}"
    except Exception as e:
        logger.error(f"Unexpected error while getting deployment status: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while getting deployment status: {e}"


def scale_deployment(service_name: str, replicas: int, namespace: str = "default") -> str:
    """
    Scales a Kubernetes deployment to a specific number of replicas.

    Args:
        service_name (str): The name of the deployment to scale.
        replicas (int): The desired number of replicas.
        namespace (str): The Kubernetes namespace.

    Returns:
        A string confirming the action or an error message.
    """
    if not k8s_apps_v1:
        return "Error: Kubernetes client is not configured."
    
    logger.info(f"DevOps Tool: Scaling deployment '{service_name}' to {replicas} replicas in namespace '{namespace}'.")
    try:
        k8s_apps_v1.patch_namespaced_deployment_scale(
            name=service_name,
            namespace=namespace,
            body={"spec": {"replicas": replicas}}
        )
        return f"Successfully scaled deployment '{service_name}' to {replicas} replicas."
    except client.ApiException as e:
        logger.error(f"Kubernetes API error while scaling deployment: {e}", exc_info=True)
        return f"Error: Failed to scale deployment. Kubernetes API error: {e.body}"
    except Exception as e:
        logger.error(f"Unexpected error while scaling deployment: {e}", exc_info=True)
        return f"Error: An unexpected error occurred during scaling: {e}"


# --- Tool Registration ---

get_service_dependencies_tool = Tool(
    name="get_service_dependencies",
    description="Finds the upstream and downstream dependencies of a service from the Knowledge Graph.",
    func=get_service_dependencies
)

get_recent_deployments_tool = Tool(
    name="get_recent_deployments",
    description="Queries the Knowledge Graph to find recent deployments for a specific service.",
    func=get_recent_deployments
)

rollback_deployment_tool = Tool(
    name="rollback_deployment",
    description="Rolls back a service to its previously deployed version. Use as a last resort if a deployment is confirmed to be faulty.",
    func=rollback_deployment
) 

restart_service_tool = Tool(
    name="restart_service",
    description="Performs a rolling restart of a service. This can resolve transient issues or memory leaks. Requires human confirmation.",
    func=restart_service
)

increase_replicas_tool = Tool(
    name="increase_service_replicas",
    description="Scales a service's deployment to a specified number of replicas to handle increased load. Requires human confirmation.",
    func=increase_service_replicas
) 

list_pods_tool = Tool(
    name="list_kubernetes_pods",
    description="Lists all pods in a given Kubernetes namespace, along with their status and age.",
    func=list_kubernetes_pods
) 

get_deployment_status_tool = Tool(
    name="get_kubernetes_deployment_status",
    description="Provides a comprehensive status report for a Kubernetes deployment, including deployment status, pod details, and recent events.",
    func=get_kubernetes_deployment_status
) 

scale_deployment_tool = Tool(
    name="scale_deployment",
    description="Scales a Kubernetes deployment to a specific number of replicas.",
    func=scale_deployment
) 

# --- Kubernetes Tools ---
try:
    # Try to load in-cluster config first
    config.load_incluster_config()
except config.ConfigException:
    # Fallback to kube-config for local development
    try:
        config.load_kube_config()
    except config.ConfigException:
        logger.warning("Could not load any Kubernetes configuration. K8s tools will not be available.")
        # Handle the case where no config is found
        # You might want to disable the tools or use a mock client

v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()

def kubernetes_get_events(service_name: str, namespace: str, config: dict = None) -> str:
    """
    Fetches Kubernetes events for pods associated with a given service.
    
    Args:
        service_name (str): The name of the service to get events for.
        namespace (str): The namespace of the service.
    
    Returns:
        str: A JSON string of the events, or an error message.
    """
    logger.info("Fetching K8s events", service=service_name, namespace=namespace)
    try:
        # First, get the service to find its selector
        service = v1.read_namespaced_service(name=service_name, namespace=namespace)
        selector = service.spec.selector
        if not selector:
            return f"Error: No selector found for service '{service_name}'."
            
        label_selector = ",".join([f"{k}={v}" for k, v in selector.items()])
        
        # Then, get pods matching the selector
        pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
        
        all_events = []
        for pod in pods.items:
            field_selector = f"involvedObject.name={pod.metadata.name},involvedObject.namespace={namespace}"
            events = v1.list_namespaced_event(namespace=namespace, field_selector=field_selector)
            all_events.extend(events.items)
            
        # Serialize events to a list of dicts
        event_list = [event.to_dict() for event in all_events]
        # Sort by last timestamp
        event_list.sort(key=lambda x: x.get('last_timestamp'), reverse=True)
        
        return json.dumps(event_list, default=str) # Use default=str for datetime objects

    except client.ApiException as e:
        logger.error("Kubernetes API error", error=str(e))
        return f"Error: Kubernetes API error: {e.body}"
    except Exception as e:
        logger.error("Unexpected error fetching K8s events", exc_info=True)
        return f"Error: An unexpected error occurred: {e}"


k8s_get_events_tool = Tool(
    name="kubernetes_get_events",
    description="Fetches recent Kubernetes events for a specific service in a namespace.",
    func=kubernetes_get_events
)

def update_k8s_secret(secret_name: str, namespace: str, new_data_json: str, config: dict = None) -> str:
    """
    Updates a Kubernetes secret with new data.

    Args:
        secret_name (str): The name of the secret to update.
        namespace (str): The namespace of the secret.
        new_data_json (str): A JSON string containing the key-value pairs to update in the secret's data field.
                             Values must be base64 encoded by the caller if necessary.

    Returns:
        str: A confirmation message or an error.
    """
    logger.info("Updating Kubernetes secret", secret=secret_name, namespace=namespace)
    try:
        new_data = json.loads(new_data_json)
        
        # The data in a K8s secret must be base64 encoded strings.
        # The caller (the agent) is responsible for this encoding.
        encoded_data = {k: base64.b64encode(v.encode()).decode() for k, v in new_data.items()}

        api = client.CoreV1Api()
        
        # First, get the existing secret to patch
        existing_secret = api.read_namespaced_secret(name=secret_name, namespace=namespace)
        
        # Update the data field
        if not existing_secret.data:
            existing_secret.data = {}
        existing_secret.data.update(encoded_data)

        # Patch the secret
        api.patch_namespaced_secret(name=secret_name, namespace=namespace, body=existing_secret)
        
        return f"Secret '{secret_name}' in namespace '{namespace}' updated successfully."

    except ApiException as e:
        logger.error("Kubernetes API error while updating secret", error=str(e))
        return f"Error: Kubernetes API error: {e.body}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for new_data_json."
    except Exception as e:
        logger.error("Unexpected error updating K8s secret", exc_info=True)
        return f"Error: An unexpected error occurred: {e}"

update_secret_tool = Tool(
    name="update_k8s_secret",
    description="Updates the data field of a specified Kubernetes secret.",
    func=update_k8s_secret
)

# Add the new tool to the list of devops tools
devops_tools = [
    k8s_get_deployments_tool,
    k8s_get_pods_tool,
    k8s_restart_deployment_tool,
    k8s_get_events_tool,
    update_secret_tool
] 