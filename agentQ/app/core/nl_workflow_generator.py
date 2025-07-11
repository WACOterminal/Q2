"""
Natural Language Workflow Generator

This service generates workflows from natural language descriptions using:
- Pattern mining insights for workflow structure
- NLP processing for intent understanding
- Template matching and adaptation
- Workflow optimization based on historical patterns
- Dynamic parameter extraction
- Validation and error handling
"""

import structlog
import yaml
from shared.q_pulse_client import QuantumPulseClient, QPChatRequest, QPChatMessage
from agentQ.app.core.toolbox import Toolbox # Assuming this can be imported or copied
import re

logger = structlog.get_logger(__name__)

# This is a simplified list of tools for the prompt.
# In a real system, this would be dynamically sourced.
AVAILABLE_TOOLS = """
- k8s_get_deployments
- k8s_get_pods
- k8s_restart_deployment
- k8s_scale_deployment
- prometheus_query_range
- elasticsearch_query
- sast_scan_directory
- http_get
- openproject_create_ticket
- get_cloud_spend
- get_llm_usage_costs
"""

GENERATION_SYSTEM_PROMPT = """
You are an expert in generating workflow YAML for the Q Platform. Your task is to convert a user's natural language request into a valid `workflow.yaml` file.

**Instructions:**
1.  Read the user's request carefully.
2.  Identify the necessary steps, tools, and agent personalities.
3.  Construct a valid YAML file that follows the platform's schema.
4.  Use the `shared_context` to pass variables between tasks.
5.  Set dependencies correctly to ensure proper execution order.
6.  Use the 'default' agent personality unless a specialized one is clearly needed (e.g., 'devops', 'security_analyst').

**Available Tools:**
{tools}

**Example Workflow:**

User Request: "When a new file is uploaded, scan it for viruses and then send a notification to the user."

Your YAML Output:
```yaml
workflow_id: "wf_virus_scan_and_notify"
original_prompt: "When a new file is uploaded, scan it for viruses and then send a notification to the user."
shared_context:
  file_path: "{{ trigger.file_path }}" # Injected by the trigger
  user_email: "{{ trigger.user_email }}"

tasks:
  - task_id: "scan_file_for_viruses"
    type: "task"
    agent_personality: "default"
    prompt: "Scan the file at '{{ shared_context.file_path }}' for viruses."
    dependencies: []
    
  - task_id: "send_notification"
    type: "task"
    agent_personality: "default"
    prompt: "Send a notification to '{{ shared_context.user_email }}' that the scan of '{{ shared_context.file_path }}' is complete. Result: {{ tasks.scan_file_for_viruses.result }}"
    dependencies: ["scan_file_for_viruses"]
```

Now, generate the YAML for the following user request. Output ONLY the raw YAML content inside a ```yaml block.
"""

class NLWorkflowGenerator:
    """
    Service for generating workflows from natural language using an LLM.
    """
    def __init__(self, qpulse_url: str):
        self.qpulse_client = QuantumPulseClient(base_url=qpulse_url)
        logger.info("NLWorkflowGenerator initialized.")

    async def generate_workflow_yaml(self, description: str) -> str:
        """
        Generates workflow YAML from a natural language description using QuantumPulse.
        
        Args:
            description: The user's natural language request.
            
        Returns:
            A string containing the generated YAML.
        """
        logger.info("Generating workflow from description via LLM", description=description)
        
        prompt = GENERATION_SYSTEM_PROMPT.format(tools=AVAILABLE_TOOLS)
        
        request = QPChatRequest(
            model="default",
            messages=[
                QPChatMessage(role="system", content=prompt),
                QPChatMessage(role="user", content=description)
            ],
            temperature=0.0, # Be precise
            max_tokens=2000
        )
        
        try:
            response = await self.qpulse_client.get_chat_completion(request)
            if not response.choices:
                raise Exception("LLM returned no choices.")
            
            generated_text = response.choices[0].message.content
            
            # Extract the YAML from the ```yaml block
            yaml_match = re.search(r"```yaml\n(.*?)\n```", generated_text, re.DOTALL)
            if not yaml_match:
                raise Exception("LLM did not return a valid YAML block.")
                
            yaml_content = yaml_match.group(1)
            
            # Validate that the output is valid YAML
            try:
                yaml.safe_load(yaml_content)
                logger.info("Successfully generated and validated workflow YAML.")
                return yaml_content
            except yaml.YAMLError as e:
                logger.error("Generated content is not valid YAML", error=str(e), content=yaml_content)
                raise Exception(f"Generated content is not valid YAML: {e}")

        except Exception as e:
            logger.error("Failed to generate workflow from LLM", exc_info=True)
            raise e # Re-raise for the caller to handle

# In a real app, the URL would come from config
generator = NLWorkflowGenerator(qpulse_url="http://quantumpulse-api.q-platform.svc.cluster.local:8000") 