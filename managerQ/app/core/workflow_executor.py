import logging
import threading
from typing import Optional, List, Set, Tuple
import jinja2
import json
from datetime import datetime, timedelta
import time # Added for workflow duration metric
import asyncio
import pulsar
from collections import defaultdict

from managerQ.app.core.workflow_manager import workflow_manager
from managerQ.app.core.task_dispatcher import task_dispatcher
from managerQ.app.models import TaskStatus, WorkflowStatus, Workflow, TaskBlock, WorkflowTask, ConditionalBlock, ApprovalBlock, LoopBlock, WorkflowEvent, EthicalReviewStatus
from managerQ.app.api.dashboard_ws import broadcast_workflow_event
from managerQ.app.config import settings
from shared.pulsar_tracing import inject_trace_context, extract_trace_context
from opentelemetry import trace
from shared.observability.metrics import WORKFLOW_COMPLETED_COUNTER, WORKFLOW_DURATION_HISTOGRAM, TASK_COMPLETED_COUNTER
from managerQ.app.dependencies import get_kg_client, get_qpulse_client # Import the dependency provider
from agentQ.app.guardian_agent import critique_action as guardian_critique_action # Import the critique function
from managerQ.app.core.observability_manager import observability_manager

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class WorkflowExecutor:
    """
    An event-driven process that listens for task status changes and advances
    workflows accordingly. Enhanced with robust error handling and recovery mechanisms.
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pulsar_client: Optional[pulsar.Client] = None
        self._task_producer: Optional[pulsar.Producer] = None
        self._conditional_producer: Optional[pulsar.Producer] = None
        self._status_consumer: Optional[pulsar.Consumer] = None
        self._jinja_env = jinja2.Environment()
        self.task_dispatcher = task_dispatcher
        
        # Enhanced robustness features
        self._workflow_timeouts = {}  # Track workflow execution timeouts
        self._circuit_breaker_state = defaultdict(dict)  # Track circuit breaker states
        self._resource_limits = {
            "max_concurrent_workflows": 100,
            "max_concurrent_tasks": 500,
            "max_workflow_duration_hours": 24,
            "max_retry_attempts": 3
        }
        self._active_workflows = set()  # Track active workflows for resource management
        self._deadlock_detection_interval = 300  # 5 minutes
        self._last_deadlock_check = time.time()

    def start(self):
        """Starts the executor in a background thread to listen for events."""
        if self._running:
            logger.warning("WorkflowExecutor is already running.")
            return
            
        self._pulsar_client = pulsar.Client(settings.pulsar.service_url)
        self._task_producer = self._pulsar_client.create_producer(settings.pulsar.topics.tasks_dispatch)
        self._conditional_producer = self._pulsar_client.create_producer(settings.pulsar.topics.tasks_conditional)

        # Subscribe to the task status update topic
        self._status_consumer = self._pulsar_client.subscribe(
            settings.pulsar.topics.tasks_status_update,
            subscription_name="managerq-workflow-executor-status-sub",
            consumer_type=pulsar.ConsumerType.Shared
        )

        self._running = True
        # The loop will now be an asyncio task
        self._thread = threading.Thread(target=self._start_async_loop, daemon=True)
        self._thread.start()
        logger.info("WorkflowExecutor started.")

    def pause_workflow(self, workflow_id: str):
        """Pauses a running workflow."""
        workflow = workflow_manager.get_workflow(workflow_id)
        if workflow and workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.PAUSED
            workflow_manager.update_workflow(workflow)
            logger.info(f"Workflow '{workflow_id}' has been paused.")
            asyncio.run(observability_manager.broadcast({"type": "WORKFLOW_UPDATE", "payload": workflow.dict()}))
    
    def resume_workflow(self, workflow_id: str):
        """Resumes a paused workflow."""
        workflow = workflow_manager.get_workflow(workflow_id)
        if workflow and workflow.status == WorkflowStatus.PAUSED:
            workflow.status = WorkflowStatus.RUNNING
            workflow_manager.update_workflow(workflow)
            logger.info(f"Workflow '{workflow_id}' has been resumed.")
            # Kick off processing again
            asyncio.run(self.process_workflow(workflow))
            asyncio.run(observability_manager.broadcast({"type": "WORKFLOW_UPDATE", "payload": workflow.dict()}))

    def cancel_workflow(self, workflow_id: str):
        """Cancels a workflow."""
        workflow = workflow_manager.get_workflow(workflow_id)
        if workflow and workflow.status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED, WorkflowStatus.PENDING]:
            workflow.status = WorkflowStatus.CANCELLED
            workflow_manager.update_workflow(workflow)
            logger.info(f"Workflow '{workflow_id}' has been cancelled.")
            asyncio.run(observability_manager.broadcast({"type": "WORKFLOW_UPDATE", "payload": workflow.dict()}))

    async def execute_workflow(self, workflow_id: str, user_id: str):
        """
        Starts the execution of a new workflow.
        This is the primary entry point for kicking off a workflow.
        Enhanced with resource limits and robustness checks.
        """
        with tracer.start_as_current_span("execute_workflow", attributes={"workflow_id": workflow_id, "user_id": user_id}) as span:
            logger.info(f"Starting execution for workflow '{workflow_id}' for user '{user_id}'.")
            
            # Check resource limits before starting
            if not self._check_resource_limits():
                logger.error(f"Resource limits exceeded. Cannot start workflow '{workflow_id}'.")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Resource limits exceeded"))
                return

            workflow = workflow_manager.get_workflow(workflow_id, user_id)
            if not workflow:
                logger.error(f"Cannot execute workflow '{workflow_id}': Not found for user '{user_id}'.")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Workflow not found"))
                return

            if workflow.status != WorkflowStatus.PENDING:
                logger.warning(f"Workflow '{workflow_id}' is already in status '{workflow.status.value}' and cannot be started again.")
                return

            # Check for circular dependencies before starting
            if self._detect_circular_dependencies(workflow):
                logger.error(f"Circular dependency detected in workflow '{workflow_id}'. Cannot start.")
                workflow.status = WorkflowStatus.FAILED
                workflow_manager.update_workflow(workflow)
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Circular dependency detected"))
                return

            workflow.status = WorkflowStatus.RUNNING
            workflow_manager.update_workflow(workflow)
            
            # Track workflow for resource management and timeout
            self._active_workflows.add(workflow_id)
            timeout_time = datetime.now() + timedelta(hours=self._resource_limits["max_workflow_duration_hours"])
            self._workflow_timeouts[workflow_id] = timeout_time

            # Broadcast the start event
            await broadcast_workflow_event(WorkflowEvent(
                event_type="WORKFLOW_STARTED",
                workflow_id=workflow_id,
                data=workflow.dict()
            ))
            await observability_manager.broadcast({
                "type": "WORKFLOW_UPDATE",
                "payload": workflow.dict()
            })
            
            # This triggers the first pass to dispatch initial tasks
            await self.process_workflow(workflow)
            span.add_event("Workflow processing initiated.")


    def _start_async_loop(self):
        """Creates and runs the asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._consumer_loop())

    def stop(self):
        """Stops the executor loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join()
        if self._task_producer:
            self._task_producer.close()
        if self._conditional_producer:
            self._conditional_producer.close()
        if self._status_consumer:
            self._status_consumer.close()
        if self._pulsar_client:
            self._pulsar_client.close()
        logger.info("WorkflowExecutor stopped.")

    async def _consumer_loop(self):
        """The main async loop for consuming task status updates."""
        while self._running:
            try:
                msg = self._status_consumer.receive(timeout_millis=1000) # This is blocking, will be addressed
                await self._handle_status_update(msg)
                self._status_consumer.acknowledge(msg)
            except pulsar.Timeout:
                continue
            except Exception as e:
                logger.error(f"Error in WorkflowExecutor consumer loop: {e}", exc_info=True)
                if 'msg' in locals():
                    self._status_consumer.negative_acknowledge(msg)
                asyncio.sleep(5)

    async def _handle_status_update(self, msg: pulsar.Message):
        """Processes a task status update message and triggers workflow progression."""
        context = extract_trace_context(msg.properties())
        with tracer.start_as_current_span("handle_status_update", context=context) as span:
            payload = json.loads(msg.data().decode('utf-8'))
            workflow_id = payload.get("workflow_id")
            task_id = payload.get("task_id")
            status_str = payload.get("status")
            raw_result = payload.get("result") # Result is now a JSON string

            if not all([workflow_id, task_id, status_str]):
                logger.error(f"Invalid status update message received: {payload}")
                return

            span.set_attributes({
                "workflow_id": workflow_id,
                "task_id": task_id,
                "status": status_str
            })
            logger.info(f"Received status update for task {task_id}: {status_str}")

            # New: Parse the structured result from the agent
            thought = None
            final_result = raw_result
            if raw_result and isinstance(raw_result, str) and raw_result.strip().startswith('{'):
                try:
                    parsed_result = json.loads(raw_result)
                    thought = parsed_result.get("thought")
                    final_result = parsed_result.get("result", raw_result)
                except json.JSONDecodeError:
                    logger.warning("Could not parse task result as JSON, treating as raw string.", task_id=task_id)

            try:
                status = TaskStatus(status_str) if status_str else None
            except ValueError:
                logger.error(f"Invalid status '{status_str}' for task {task_id}.")
                return

            # NEW: Intercept failed status to trigger self-correction
            if status == TaskStatus.FAILED:
                logger.warning(f"Task {task_id} in workflow {workflow_id} has failed. Initiating self-correction process.")
                await self._handle_task_failure(workflow_id, task_id, final_result)
                return # Stop normal processing for this failed task

            # >>> NEW: Intercept COMPLETED status for ethical review <<<
            workflow = workflow_manager.get_workflow(workflow_id) # Get workflow once
            if not workflow:
                logger.error(f"Could not find workflow {workflow_id} for status update.")
                return

            if status == TaskStatus.COMPLETED:
                task_to_review = workflow.get_task(task_id)
                if task_to_review and not task_to_review.bypass_ethical_review:
                    review_status, review_reason = await self._perform_ethical_review(task_to_review, workflow, final_result)
                    
                    if review_status == EthicalReviewStatus.VETOED:
                        logger.warning(f"Guardian VETOED task {task_id} in workflow {workflow_id}. Reason: {review_reason}")
                        # Update task status to FAILED and trigger the failure handling logic
                        workflow_manager.update_task_status(workflow_id, task_id, TaskStatus.FAILED, review_reason, thought)
                        await self._handle_task_failure(workflow_id, task_id, review_reason)
                        return # Stop normal processing

            # Instrument task status metric
            if status and status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                TASK_COMPLETED_COUNTER.labels(status=status.value).inc()

            # Update task with both thought and result
            workflow_manager.update_task_status(workflow_id, task_id, status, final_result, thought)

            if workflow and workflow.status == WorkflowStatus.RUNNING:
                await self.process_workflow(workflow)

    async def _handle_task_failure(self, workflow_id: str, task_id: str, result: str):
        """
        Handles a failed task. If retries are available, it constructs a new prompt
        with context about the failure and re-dispatches the task. Otherwise, it
        fails the workflow.
        """
        workflow = workflow_manager.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Cannot handle failure for workflow '{workflow_id}': not found.")
            return

        failed_task = workflow.get_task(task_id)
        if not failed_task or not isinstance(failed_task, WorkflowTask):
            logger.error(f"Cannot handle failure for task '{task_id}': not found or not a valid task in workflow.")
            # Fail the workflow if we can't find the task to retry
            workflow.status = WorkflowStatus.FAILED
            workflow_manager.update_workflow(workflow)
            return

        failed_task.retry_count += 1
        logger.info(f"Handling failure for task {task_id}. Retry attempt {failed_task.retry_count} of {failed_task.max_retries}.")

        if failed_task.retry_count > failed_task.max_retries:
            logger.error(f"Task {task_id} has exceeded max retries. Failing workflow {workflow_id}.")
            workflow.status = WorkflowStatus.FAILED
            workflow_manager.update_workflow(workflow)
            # Broadcast failure event
            await observability_manager.broadcast({"type": "WORKFLOW_UPDATE", "payload": workflow.dict()})
            return

        # Construct a new prompt for the retry
        correction_prompt = f"""
        A previous attempt to execute this task failed. You must correct the approach.
        
        Original Prompt:
        ---
        {failed_task.prompt}
        ---

        Reason for Failure / Veto:
        ---
        {result}
        ---
        
        Please analyze the reason for failure and provide a corrected plan or action.
        Your output must follow the original format and constraints.
        """

        try:
            # Re-dispatch the task with the new correction prompt.
            # We keep the original task_id to track it.
            self.task_dispatcher.dispatch_task(
                personality=failed_task.agent_personality,
                prompt=correction_prompt,
                workflow_id=workflow_id,
                # Pass through other relevant properties
            )
            
            # Reset the task status to PENDING so it can be picked up again
            workflow_manager.update_task_status(workflow_id, task_id, TaskStatus.PENDING, result=f"Retrying after failure: {result}")
            logger.info(f"Re-dispatched task {task_id} for retry.")
            
            # Immediately re-process the workflow to see if the retried task can run
            await self.process_workflow(workflow)

        except Exception as e:
            logger.error(f"Self-correction failed for task '{task_id}': {e}", exc_info=True)
            workflow.status = WorkflowStatus.FAILED
            workflow_manager.update_workflow(workflow)


    async def process_workflow(self, workflow: Workflow):
        """
        Processes a single workflow's execution state, dispatching any new tasks that are ready.
        Enhanced with timeout checking and deadlock detection.
        """
        logger.info(f"Processing workflow '{workflow.workflow_id}'...")
        
        # Check for workflow timeout
        if self._check_workflow_timeout(workflow.workflow_id):
            logger.warning(f"Workflow '{workflow.workflow_id}' has exceeded maximum execution time. Marking as failed.")
            workflow.status = WorkflowStatus.FAILED
            workflow.final_result = "Workflow exceeded maximum execution time limit"
            workflow_manager.update_workflow(workflow)
            self._cleanup_workflow(workflow.workflow_id)
            return

        # Periodic deadlock detection
        if time.time() - self._last_deadlock_check > self._deadlock_detection_interval:
            await self._check_for_deadlocks()
            self._last_deadlock_check = time.time()

        await self._process_blocks(workflow.tasks, workflow)

        all_blocks_after = workflow.get_all_tasks_recursive()
        is_complete = all(block.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED} for block in all_blocks_after)
        
        if is_complete:
            final_status = WorkflowStatus.FAILED if any(b.status == TaskStatus.FAILED for b in all_blocks_after) else WorkflowStatus.COMPLETED
            
            # --- New: Extract and save the final result ---
            if final_status == WorkflowStatus.COMPLETED:
                final_tasks = [task for task in all_blocks_after if not any(other_task.task_id in task.dependencies for other_task in all_blocks_after if other_task is not task)]
                # For simplicity, if there are multiple final tasks, we'll concatenate their results.
                final_results = [t.result for t in final_tasks if t.result]
                workflow.final_result = "\n".join(final_results)
            # -----------------------------------------

            # Instrument workflow metrics
            WORKFLOW_COMPLETED_COUNTER.labels(status=final_status.value).inc()
            duration_seconds = time.time() - workflow.created_at.timestamp()
            WORKFLOW_DURATION_HISTOGRAM.observe(duration_seconds)

            workflow.status = final_status
            workflow_manager.update_workflow(workflow)
            logger.info(f"Workflow '{workflow.workflow_id}' has finished with status '{final_status.value}'.")
            
            # Cleanup workflow resources
            self._cleanup_workflow(workflow.workflow_id)
            
            # Broadcast to the old dashboard and the new observability dashboard
            await broadcast_workflow_event(WorkflowEvent(
                event_type="WORKFLOW_COMPLETED",
                workflow_id=workflow.workflow_id,
                data={"status": final_status.value}
            ))
            await observability_manager.broadcast({
                "type": "WORKFLOW_UPDATE",
                "payload": workflow.dict()
            })
            
            # Check if this was an AIOps workflow and handle the final report
            if workflow.event_id and final_status == WorkflowStatus.COMPLETED:
                await self._handle_completed_aiops_workflow(workflow)
            else:
                await self._trigger_reflection_task(workflow)


    async def _handle_completed_aiops_workflow(self, workflow: Workflow):
        """
        Handles the completion of a event-driven workflow, like the AIOps one,
        by ingesting the final report into the Knowledge Graph.
        """
        logger.info(f"Handling completed AIOps workflow '{workflow.workflow_id}'.")
        
        # Find the final task (a task with no other tasks depending on it)
        all_task_ids = {task.task_id for task in workflow.get_all_tasks_recursive()}
        dependency_ids = set()
        for task in workflow.get_all_tasks_recursive():
            dependency_ids.update(task.dependencies)
            
        final_task_ids = all_task_ids - dependency_ids
        
        if not final_task_ids:
            logger.warning(f"Could not determine final task for AIOps workflow '{workflow.workflow_id}'.")
            return
            
        # Assuming one final report task for simplicity
        final_task_id = final_task_ids.pop()
        final_task = workflow.get_task(final_task_id)
        
        if not final_task or not final_task.result:
            logger.warning(f"Final task '{final_task_id}' for AIOps workflow has no result to ingest.")
            return
            
        service_name = workflow.shared_context.get("service_name")
        if not service_name:
            logger.warning("AIOps workflow is missing 'service_name' in shared_context.")
            return

        logger.info(f"Ingesting final report from task '{final_task_id}' into Knowledge Graph.")
        
        try:
            # This is not ideal, but a workaround for the client management issue
            kg_client = get_kg_client()
            
            report_content = final_task.result.replace("'", "\\'") # Simple escaping for Gremlin
            
            # Gremlin query to create the report and link it
            query = f"""
            def report = g.addV('Report')
                           .property('source', 'AIOpsWorkflow')
                           .property('content', '{report_content}')
                           .property('createdAt', '{datetime.utcnow().isoformat()}')
                           .next()
            
            def service = g.V().has('Service', 'name', '{service_name}').tryNext().orElse(null)
            if (service != null) {{
                g.V(report).addE('REPORT_FOR').to(service).iterate()
            }}
            
            def event = g.V().has('Event', 'eventId', '{workflow.event_id}').tryNext().orElse(null)
            if (event != null) {{
                g.V(report).addE('GENERATED_FROM').to(event).iterate()
            }}
            
            return report
            """
            
            # We need to run this async function in our sync thread
            await kg_client.execute_gremlin_query(query)
            logger.info("Successfully ingested AIOps report into Knowledge Graph.")
        
        except Exception as e:
            logger.error(f"Failed to ingest AIOps report into Knowledge Graph: {e}", exc_info=True)


    async def _trigger_reflection_task(self, workflow: Workflow):
        """Creates and dispatches a task for a 'reflector' agent to analyze a completed workflow."""
        logger.info(f"Triggering reflection for workflow '{workflow.workflow_id}'.")
        
        # Serialize the workflow to JSON to pass to the reflector
        workflow_dump = workflow.json(indent=2)

        prompt = (
            "You are a Reflector Agent. Your purpose is to analyze a completed workflow to find insights and lessons.\n\n"
            "Analyze the following workflow execution record. Identify key successes, failures, and reasons for the outcome. "
            "Formulate a concise 'lesson learned' that can be stored in our knowledge graph to improve future planning.\n\n"
            f"Workflow Analysis Request for: {workflow.workflow_id}\n"
            f"Original Goal: {workflow.original_prompt}\n"
            f"Final Status: {workflow.status.value}\n\n"
            f"Full Workflow Record:\n{workflow_dump}"
        )

        try:
            # Dispatch to a reflector agent. This is a "fire-and-forget" task.
            self.task_dispatcher.dispatch_task(
                prompt=prompt,
                agent_personality='reflector'
            )
            logger.info(f"Successfully dispatched reflection task for workflow '{workflow.workflow_id}'.")
        except RuntimeError as e:
            logger.error(f"Failed to dispatch reflection task for workflow '{workflow.workflow_id}': {e}", exc_info=True)


    async def _process_blocks(self, blocks: List[TaskBlock], workflow: Workflow):
        """Recursively processes a list of tasks, dispatching all that are ready."""
        all_blocks = workflow.get_all_tasks_recursive()
        completed_ids = {block.task_id for block in all_blocks if block.status == TaskStatus.COMPLETED}

        for block in blocks:
            if block.status == TaskStatus.PENDING and set(block.dependencies).issubset(completed_ids):
                if isinstance(block, WorkflowTask):
                    # NEW: Check for task-level condition before dispatching
                    if block.condition:
                        eval_context = self._get_evaluation_context(workflow)
                        try:
                            template = self._jinja_env.from_string(block.condition)
                            if template.render(eval_context):
                                await self._dispatch_task(block, workflow)
                            else:
                                # If condition is not met, mark the task as cancelled (or a new 'SKIPPED' status)
                                logger.info(f"Skipping task '{block.task_id}' due to unmet condition.")
                                workflow_manager.update_task_status(workflow.workflow_id, block.task_id, TaskStatus.CANCELLED, result="Condition not met.")
                        except Exception as e:
                            logger.error(f"Failed to evaluate condition for task '{block.task_id}': {e}", exc_info=True)
                            workflow_manager.update_task_status(workflow.workflow_id, block.task_id, TaskStatus.FAILED, result=f"Condition evaluation failed: {e}")
                    else:
                        await self._dispatch_task(block, workflow)
                elif isinstance(block, ConditionalBlock):
                    self._evaluate_conditional(block, workflow)
                elif isinstance(block, ApprovalBlock):
                    await self._handle_approval_block(block, workflow)
                elif isinstance(block, LoopBlock):
                    await self._handle_loop_block(block, workflow)
            
            # This recursive call is problematic and can lead to re-processing.
            # A better approach would be a single pass. Refactoring this is out of scope for now.
            if hasattr(block, 'tasks') and block.tasks:
                await self._process_blocks(block.tasks, workflow)


    async def _handle_loop_block(self, block: LoopBlock, workflow: Workflow):
        """Handles the logic for a loop block."""
        logger.info(f"Evaluating loop block '{block.task_id}' for workflow '{workflow.workflow_id}'.")
        
        eval_context = self._get_evaluation_context(workflow)
        
        # Track iterations in the shared context
        iteration_key = f"loop_{block.task_id}_iterations"
        current_iteration = workflow.shared_context.get(iteration_key, 0)
        
        if current_iteration >= block.max_iterations:
            logger.warning(f"Loop block '{block.task_id}' exceeded max iterations. Marking as complete.")
            workflow_manager.update_task_status(workflow.workflow_id, block.task_id, TaskStatus.COMPLETED, result="Max iterations reached.")
            return

        try:
            template = self._jinja_env.from_string(block.condition)
            should_continue = template.render(eval_context)
            
            # Jinja render can return empty strings for falsey conditions
            should_continue = str(should_continue).lower() in ['true']

            if should_continue:
                logger.info(f"Loop condition for '{block.task_id}' is true. Starting iteration {current_iteration + 1}.")
                # Increment iteration count
                workflow.shared_context[iteration_key] = current_iteration + 1
                
                # Reset all tasks within the loop to PENDING to allow re-execution
                for task in block.tasks:
                    workflow_manager.update_task_status(workflow.workflow_id, task.task_id, TaskStatus.PENDING)
                
                # Re-process the blocks within the loop
                await self._process_blocks(block.tasks, workflow)
            else:
                logger.info(f"Loop condition for '{block.task_id}' is false. Marking as complete.")
                workflow_manager.update_task_status(workflow.workflow_id, block.task_id, TaskStatus.COMPLETED, result="Condition became false.")

        except Exception as e:
            logger.error(f"Failed to evaluate condition for loop block '{block.task_id}': {e}", exc_info=True)
            workflow_manager.update_task_status(workflow.workflow_id, block.task_id, TaskStatus.FAILED, result=f"Loop condition evaluation failed: {e}")


    async def _handle_approval_block(self, block: ApprovalBlock, workflow: Workflow):
        """Handles a workflow block that requires human approval."""
        logger.info(f"Pausing workflow '{workflow.workflow_id}' for human approval on task '{block.task_id}'.")
        
        # Update the task's status to indicate it's waiting for a decision.
        # The workflow will not proceed down this path until an external API call
        # changes this status to 'COMPLETED' (approved) or 'FAILED' (rejected).
        workflow_manager.update_task_status(workflow.workflow_id, block.task_id, TaskStatus.PENDING_APPROVAL)
        
        # Broadcast the event so the UI can update
        await broadcast_workflow_event(WorkflowEvent(
            event_type="APPROVAL_REQUIRED",
            workflow_id=workflow.workflow_id,
            task_id=block.task_id,
            data={"message": block.message}
        ))

    async def _dispatch_task(self, task: WorkflowTask, workflow: Workflow):
        """
        Dispatches a task for execution, potentially after an ethical review.
        """
        # --- NEW: Ethical Review Gate ---
        if self._is_critical_task(task):
            logger.info("Critical task requires ethical review", task_id=task.task_id)
            review_status, review_details = await self._perform_ethical_review(task, workflow, task.result)
            
            if review_status == EthicalReviewStatus.VETOED:
                logger.warning("Ethical review VETOED, halting workflow.", task_id=task.task_id, details=review_details)
                workflow.status = WorkflowStatus.FAILED
                workflow.final_result = f"Ethical Veto: {review_details}"
                workflow_manager.update_workflow(workflow)
                return # Stop execution
            
            logger.info("Ethical review APPROVED", task_id=task.task_id)
        # --- End Ethical Review Gate ---
        
        logger.info(f"Dispatching task '{task.task_id}' for workflow '{workflow.workflow_id}' to Pulsar.")
        
        try:
            # Render prompt using Jinja2 and the workflow's shared context
            template = self._jinja_env.from_string(task.prompt)
            rendered_prompt = template.render(self._get_evaluation_context(workflow))
            
            task_payload = {
                "task_id": task.task_id,
                "workflow_id": workflow.workflow_id,
                "agent_personality": task.agent_personality,
                "prompt": rendered_prompt,
            }
            
            properties = inject_trace_context({})
            self._task_producer.send(
                json.dumps(task_payload).encode('utf-8'),
                properties=properties
            )
            
            workflow_manager.update_task_status(workflow.workflow_id, task.task_id, TaskStatus.DISPATCHED)
            
            # Broadcast dispatch event
            await broadcast_workflow_event(WorkflowEvent(
                event_type="TASK_STATUS_UPDATE",
                workflow_id=workflow.workflow_id,
                task_id=task.task_id,
                data={"status": TaskStatus.DISPATCHED.value}
            ))
            await observability_manager.broadcast({
                "type": "WORKFLOW_UPDATE",
                "payload": workflow.dict()
            })

        except jinja2.TemplateError as e:
            logger.error(f"Failed to render prompt for task '{task.task_id}': {e}", exc_info=True)
            workflow_manager.update_task_status(workflow.workflow_id, task.task_id, TaskStatus.FAILED, result=f"Prompt rendering failed: {e}")
        except Exception as e:
            logger.error(f"Failed to publish task '{task.task_id}' to Pulsar: {e}", exc_info=True)
            # Optionally, set the task to FAILED here as well
            workflow_manager.update_task_status(workflow.workflow_id, task.task_id, TaskStatus.FAILED, result="Failed to publish to message queue.")


    def _evaluate_conditional(self, block: ConditionalBlock, workflow: Workflow):
        """Publishes a conditional evaluation task to Pulsar."""
        logger.info(f"Publishing conditional evaluation for block '{block.task_id}' to Pulsar.")
        
        eval_context = self._get_evaluation_context(workflow)

        conditional_payload = {
            "block_id": block.task_id,
            "workflow_id": workflow.workflow_id,
            "evaluation_context": eval_context,
            "branches": [branch.dict() for branch in block.branches]
        }
        
        properties = inject_trace_context({})
        self._conditional_producer.send(
            json.dumps(conditional_payload).encode('utf-8'),
            properties=properties
        )
        
        # We don't mark as complete here anymore. The worker will do that.
        # We can, however, mark it as "evaluating" if we add such a status.
        # For now, we'll leave it as PENDING until the worker picks it up.

    def _get_evaluation_context(self, workflow: Workflow) -> dict:
        """
        Creates a context for Jinja2 rendering, combining workflow's shared_context
        with the results of all completed tasks.
        """
        context = workflow.shared_context.copy()
        
        # Add task results to the context under a 'tasks' key for easy access.
        # e.g., {{ tasks.task_1.result }}
        task_results = {}
        for task in workflow.get_all_tasks_recursive():
            if task.status == TaskStatus.COMPLETED and isinstance(task, WorkflowTask):
                 # Try to parse JSON result, otherwise use the raw string
                try:
                    task_results[task.task_id] = json.loads(task.result) if task.result and task.result.strip().startswith(('{', '[')) else task.result
                except (json.JSONDecodeError, TypeError):
                    task_results[task.task_id] = task.result

        context['tasks'] = task_results
        return context

    def _is_critical_task(self, task: WorkflowTask) -> bool:
        """Determines if a task is critical based on its properties or context."""
        # Simple logic: check for a 'critical' tag or a specific tool.
        # This could be expanded significantly.
        return "critical" in task.tags or task.tool_name in {"delete_production_database"}


    async def _perform_ethical_review(self, task: WorkflowTask, workflow: Workflow, task_result: str) -> Tuple[EthicalReviewStatus, str]:
        """
        Sends the proposed action (represented by the task and its result) to the Guardian Agent for review.
        """
        # We only review tasks that have a tangible, potentially impactful result.
        if not task_result or not self._is_critical_task(task):
            return EthicalReviewStatus.APPROVED, "Task is not critical or has no result; review bypassed."

        logger.info(f"Submitting task {task.task_id} for ethical review...")

        proposed_action_plan = {
            "workflow_id": workflow.workflow_id,
            "task_id": task.task_id,
            "task_prompt": task.prompt,
            "tool_name": task.tool_name,
            "parameters": task.parameters,
            "proposed_result": task_result
        }

        try:
            # Get clients using dependency injectors
            qpulse_client = get_qpulse_client()
            llm_config = {"model": "q-pulse-guardian-v1"} # Use a specialized model for this

            # Call the guardian agent's critique function
            critique = await guardian_critique_action(proposed_action_plan, qpulse_client, llm_config)

            if critique["verdict"] == "VETO":
                return EthicalReviewStatus.VETOED, critique["reasoning"]
            else:
                return EthicalReviewStatus.APPROVED, critique["reasoning"]

        except Exception as e:
            logger.error(f"Ethical review for task {task.task_id} failed with an exception: {e}", exc_info=True)
            # Fail-safe: Veto the action if the review process itself fails.
            return EthicalReviewStatus.VETOED, "The ethical review process encountered an internal error."


    def _evaluate_conditional(self, block: ConditionalBlock, workflow: Workflow):
        """
        Evaluates the condition for a conditional block and marks the appropriate
        """
        logger.info(f"Publishing conditional evaluation for block '{block.task_id}' to Pulsar.")
        
        eval_context = self._get_evaluation_context(workflow)

        conditional_payload = {
            "block_id": block.task_id,
            "workflow_id": workflow.workflow_id,
            "evaluation_context": eval_context,
            "branches": [branch.dict() for branch in block.branches]
        }
        
        properties = inject_trace_context({})
        self._conditional_producer.send(
            json.dumps(conditional_payload).encode('utf-8'),
            properties=properties
        )
        
        # We don't mark as complete here anymore. The worker will do that.
        # We can, however, mark it as "evaluating" if we add such a status.
        # For now, we'll leave it as PENDING until the worker picks it up.

    def _check_resource_limits(self) -> bool:
        """
        Checks if the system has sufficient resources to start a new workflow.
        Returns True if resources are available, False otherwise.
        """
        active_workflow_count = len(self._active_workflows)
        
        if active_workflow_count >= self._resource_limits["max_concurrent_workflows"]:
            logger.warning(f"Maximum concurrent workflows ({self._resource_limits['max_concurrent_workflows']}) exceeded. Active: {active_workflow_count}")
            return False
        
        # Check active task count across all workflows
        active_task_count = 0
        for workflow_id in self._active_workflows:
            workflow = workflow_manager.get_workflow(workflow_id)
            if workflow:
                active_task_count += len([t for t in workflow.get_all_tasks_recursive() 
                                        if t.status in [TaskStatus.PENDING, TaskStatus.DISPATCHED, TaskStatus.RUNNING]])
        
        if active_task_count >= self._resource_limits["max_concurrent_tasks"]:
            logger.warning(f"Maximum concurrent tasks ({self._resource_limits['max_concurrent_tasks']}) exceeded. Active: {active_task_count}")
            return False
        
        return True

    def _detect_circular_dependencies(self, workflow: Workflow) -> bool:
        """
        Detects circular dependencies in workflow tasks using DFS.
        Returns True if circular dependencies are found, False otherwise.
        """
        try:
            all_tasks = workflow.get_all_tasks_recursive()
            task_map = {task.task_id: task for task in all_tasks}
            
            # Build adjacency list
            graph = defaultdict(list)
            for task in all_tasks:
                for dep in task.dependencies:
                    if dep in task_map:
                        graph[dep].append(task.task_id)
            
            # DFS to detect cycles
            visited = set()
            rec_stack = set()
            
            def has_cycle(node):
                if node in rec_stack:
                    return True
                if node in visited:
                    return False
                
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in graph[node]:
                    if has_cycle(neighbor):
                        return True
                
                rec_stack.remove(node)
                return False
            
            # Check all nodes
            for task_id in task_map:
                if task_id not in visited:
                    if has_cycle(task_id):
                        logger.error(f"Circular dependency detected involving task: {task_id}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting circular dependencies: {e}", exc_info=True)
            return True  # Fail safe - assume circular dependency if we can't check

    def _check_workflow_timeout(self, workflow_id: str) -> bool:
        """
        Checks if a workflow has exceeded its maximum execution time.
        Returns True if the workflow has timed out, False otherwise.
        """
        timeout_time = self._workflow_timeouts.get(workflow_id)
        if timeout_time and datetime.now() > timeout_time:
            logger.warning(f"Workflow {workflow_id} has exceeded maximum execution time")
            return True
        return False

    def _cleanup_workflow(self, workflow_id: str):
        """
        Cleans up workflow resources and tracking data.
        """
        self._active_workflows.discard(workflow_id)
        self._workflow_timeouts.pop(workflow_id, None)
        # Clean up circuit breaker state for this workflow
        if workflow_id in self._circuit_breaker_state:
            del self._circuit_breaker_state[workflow_id]

    async def _check_for_deadlocks(self):
        """
        Performs deadlock detection across all active workflows.
        Looks for workflows that have been stuck without progress for too long.
        """
        logger.debug("Performing deadlock detection check")
        
        stuck_threshold = timedelta(minutes=30)  # Consider workflow stuck if no progress for 30 minutes
        current_time = datetime.now()
        
        for workflow_id in list(self._active_workflows):
            workflow = workflow_manager.get_workflow(workflow_id)
            if not workflow:
                continue
                
            # Check if workflow has made progress recently
            last_activity = workflow.created_at
            for task in workflow.get_all_tasks_recursive():
                if hasattr(task, 'updated_at') and task.updated_at:
                    last_activity = max(last_activity, task.updated_at)
            
            if current_time - last_activity > stuck_threshold:
                # Check if workflow is actually stuck (has pending tasks but no running tasks)
                all_tasks = workflow.get_all_tasks_recursive()
                pending_tasks = [t for t in all_tasks if t.status == TaskStatus.PENDING]
                running_tasks = [t for t in all_tasks if t.status in [TaskStatus.DISPATCHED, TaskStatus.RUNNING]]
                
                if pending_tasks and not running_tasks:
                    logger.warning(f"Potential deadlock detected in workflow {workflow_id}. "
                                 f"Has {len(pending_tasks)} pending tasks but no running tasks.")
                    
                    # Try to resolve deadlock by checking if any pending tasks can be dispatched
                    try:
                        await self.process_workflow(workflow)
                    except Exception as e:
                        logger.error(f"Failed to resolve potential deadlock in workflow {workflow_id}: {e}")
                        
                        # If we can't resolve it, mark workflow as failed
                        workflow.status = WorkflowStatus.FAILED
                        workflow.final_result = "Workflow stuck in deadlock - terminated"
                        workflow_manager.update_workflow(workflow)
                        self._cleanup_workflow(workflow_id)

    def _get_circuit_breaker_state(self, service_name: str) -> dict:
        """
        Gets the circuit breaker state for a service.
        """
        if service_name not in self._circuit_breaker_state:
            self._circuit_breaker_state[service_name] = {
                "state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
                "failure_count": 0,
                "last_failure_time": None,
                "success_count": 0,
                "recovery_timeout": 60  # seconds
            }
        return self._circuit_breaker_state[service_name]

    def _update_circuit_breaker_on_failure(self, service_name: str):
        """
        Updates circuit breaker state when a service call fails.
        """
        cb_state = self._get_circuit_breaker_state(service_name)
        cb_state["failure_count"] += 1
        cb_state["last_failure_time"] = datetime.now()
        
        # Open circuit breaker if failure threshold is reached
        if cb_state["failure_count"] >= 5:  # Threshold of 5 failures
            cb_state["state"] = "OPEN"
            logger.warning(f"Circuit breaker opened for service {service_name}")

    def _update_circuit_breaker_on_success(self, service_name: str):
        """
        Updates circuit breaker state when a service call succeeds.
        """
        cb_state = self._get_circuit_breaker_state(service_name)
        
        if cb_state["state"] == "HALF_OPEN":
            cb_state["success_count"] += 1
            # Close circuit breaker if we get enough consecutive successes
            if cb_state["success_count"] >= 3:
                cb_state["state"] = "CLOSED"
                cb_state["failure_count"] = 0
                cb_state["success_count"] = 0
                logger.info(f"Circuit breaker closed for service {service_name}")
        elif cb_state["state"] == "CLOSED":
            # Reset failure count on success
            cb_state["failure_count"] = 0

    def _should_allow_request(self, service_name: str) -> bool:
        """
        Determines if a request should be allowed based on circuit breaker state.
        """
        cb_state = self._get_circuit_breaker_state(service_name)
        
        if cb_state["state"] == "CLOSED":
            return True
        elif cb_state["state"] == "OPEN":
            # Check if we should transition to HALF_OPEN
            if cb_state["last_failure_time"]:
                time_since_failure = (datetime.now() - cb_state["last_failure_time"]).total_seconds()
                if time_since_failure >= cb_state["recovery_timeout"]:
                    cb_state["state"] = "HALF_OPEN"
                    cb_state["success_count"] = 0
                    logger.info(f"Circuit breaker transitioning to HALF_OPEN for service {service_name}")
                    return True
            return False
        elif cb_state["state"] == "HALF_OPEN":
            return True
        
        return False

# Singleton instance
workflow_executor = WorkflowExecutor() 