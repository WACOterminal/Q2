import logging
import json
import pulsar
import structlog
import asyncio
import yaml
from jinja2 import Template
import httpx

from managerQ.app.core.task_dispatcher import task_dispatcher
from managerQ.app.core.agent_registry import agent_registry
from managerQ.app.api.dashboard_ws import manager as dashboard_manager
from managerQ.app.core.workflow_manager import workflow_manager
from managerQ.app.models import Workflow
from managerQ.app.core.observability_manager import observability_manager
from managerQ.app.core.goal_manager import goal_manager
from managerQ.app.core.goal_monitor import proactive_goal_monitor
from managerQ.app.models import Goal


logger = structlog.get_logger(__name__)

class EventListener:
    """
    Listens for platform events and triggers agent tasks accordingly.
    """

    def __init__(self, service_url: str, event_topic: str):
        self.service_url = service_url
        self.topics = [
            event_topic, 
            "persistent://public/default/q.goals.new",
            "persistent://public/default/q.goals.bug_report" # Listen for bug reports
        ]
        self.client = None
        self.consumer = None
        self._running = False

    def start(self):
        """Starts the Pulsar consumer in a separate thread."""
        logger.info("Starting EventListener...", topics=self.topics)
        self.client = pulsar.Client(self.service_url)
        self.consumer = self.client.subscribe(
            self.topics,
            subscription_name="managerq-event-listener-sub",
            consumer_type=pulsar.ConsumerType.Shared
        )
        self._running = True
        
        # In a real application, you'd run this in a background thread.
        # For simplicity, we'll assume the main event loop allows this.
        # This is a blocking call, so this approach needs to be refined in a real system.
        logger.info("EventListener started and waiting for messages.")
        while self._running:
            try:
                msg = self.consumer.receive(timeout_millis=1000)
                self.handle_message(msg)
                self.consumer.acknowledge(msg)
            except pulsar.Timeout:
                continue
            except Exception as e:
                logger.error("Error in EventListener loop", error=str(e), exc_info=True)
                if 'msg' in locals():
                    self.consumer.negative_acknowledge(msg)
    
    def stop(self):
        self._running = False
        if self.consumer:
            self.consumer.close()
        if self.client:
            self.client.close()
        logger.info("EventListener stopped.")

    def handle_message(self, msg):
        """Processes a single message from the event topic."""
        try:
            event_data = json.loads(msg.data().decode('utf-8'))
            event_type = event_data.get("event_type")
            logger.info("Received platform event", event_type=event_type)

            if event_type == "anomaly.detected.error_rate":
                self.handle_anomaly_event(event_data)
            elif event_type == "MODEL_FEEDBACK_RECEIVED":
                self.handle_model_feedback_event(event_data)
            elif event_type == "goal.new.from_openproject":
                self.handle_new_goal_event(event_data)
            elif event_type == "goal.new.bug_report":
                self.handle_bug_report_event(event_data)
        
        except json.JSONDecodeError:
            logger.warning("Could not decode event message", raw_data=msg.data())
        except Exception as e:
            logger.error("Failed to handle event message", error=str(e), exc_info=True)

    def handle_model_feedback_event(self, event_data: dict):
        """Handles a model feedback event by broadcasting it to the dashboard."""
        logger.info(f"Broadcasting model feedback event: {event_data['payload']}")
        asyncio.run(observability_manager.broadcast({
            "type": "MODEL_A/B_TEST_UPDATE",
            "payload": event_data["payload"]
        }))

    def handle_bug_report_event(self, event_data: dict):
        """Handles a new bug report by triggering the triage workflow."""
        payload = event_data.get("payload", {})
        prompt = payload.get("prompt")
        source_id = payload.get("source_id", "unknown_bug_report")
        
        logger.info(f"Processing new bug report from '{source_id}': {prompt}")
        
        try:
            # This goal uses a pre-defined workflow template for triage.
            goal = Goal(
                objective=f"Triage and fix bug from {source_id}: {prompt}",
                remediation_workflow_id="wf_triage_bug_report",
                # The event payload itself becomes the context for the trigger
                context_overrides={"trigger": event_data}
            )
            # Use the goal monitor's trigger logic to start the workflow
            proactive_goal_monitor.trigger_goal_workflow(goal)
            logger.info(f"Successfully triggered bug triage workflow for '{source_id}'.")
        except Exception as e:
            logger.error(f"Failed to trigger bug triage workflow for '{source_id}'", error=str(e), exc_info=True)


    def handle_new_goal_event(self, event_data: dict):
        """Handles a new goal submitted from an external source like OpenProject."""
        payload = event_data.get("payload", {})
        prompt = payload.get("prompt")
        source_id = payload.get("source_id", "unknown_source") # e.g., "openproject-123"

        if not prompt:
            logger.warning("New goal event received without a prompt.", payload=payload)
            return
            
        logger.info(f"Processing new goal received from event source '{source_id}'.")
        
        # We need to run the async goal manager function in our synchronous event loop.
        # This is a common pattern when bridging sync/async code.
        try:
            # The user_id can be a system identifier for goals from external sources.
            user_id = f"system_user:{source_id}"
            asyncio.run(goal_manager.process_new_goal(prompt, user_id))
            logger.info(f"Successfully started processing goal from source '{source_id}'.")
        except Exception as e:
            # The goal_manager will raise AmbiguousGoalError, but we also catch others.
            logger.error(f"Failed to process new goal from event source '{source_id}'", error=str(e), exc_info=True)


    def handle_anomaly_event(self, event_data: dict):
        """Handles an error rate anomaly event by calling the planner to create an investigation workflow."""
        payload = event_data.get("payload", {})
        service_name = payload.get("service_name")
        event_id = event_data.get("event_id")
        
        if not service_name or not event_id:
            logger.warning("Anomaly event received without a service_name or event_id", payload=payload)
            return

        logger.info(f"Anomaly detected in '{service_name}'. Requesting a new investigation plan.", event_id=event_id)
        
        # Broadcast the raw anomaly event to any connected dashboards
        asyncio.run(dashboard_manager.broadcast({
            "event_type": "anomaly_detected",
            "data": event_data
        }))

        try:
            # The EventListener runs in a separate thread, so we can use a synchronous HTTP client.
            # In a fully async app, we would use an async client here.
            goal = f"An automated alert has been triggered for the service '{service_name}'. The service is experiencing an anomalous error rate. Create a workflow to diagnose the root cause and propose a remediation plan."
            
            # This requires AuthQ to be running and accessible.
            # For now, we'll assume a local, unauthenticated endpoint for simplicity.
            # In a real system, this would need a service-to-service auth mechanism.
            with httpx.Client() as client:
                response = client.post(
                    "http://localhost:8000/api/v1/planner/plan", # Assuming managerQ runs on 8000
                    json={"goal": goal},
                    headers={"Authorization": "Bearer service-account-token"} # Placeholder
                )
                response.raise_for_status()
                workflow = response.json()
                logger.info(f"Successfully created investigation workflow '{workflow['workflow_id']}' from planner for event '{event_id}'.")

        except Exception as e:
            logger.error(f"Failed to create investigation workflow from planner for event '{event_id}'", error=str(e), exc_info=True) 