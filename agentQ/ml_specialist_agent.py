"""
ML Specialist Agent

This agent provides specialized access to all ML capabilities including:
- Federated Learning
- AutoML 
- Reinforcement Learning
- Multi-modal AI
"""

import structlog
from agentQ.app.core.toolbox import Toolbox
from agentQ.app.core.context import ContextManager

# Import all ML tools
try:
    from agentQ.app.core.ml_tools import ml_tools
    ML_TOOLS_AVAILABLE = True
except ImportError:
    ml_tools = []
    ML_TOOLS_AVAILABLE = False

logger = structlog.get_logger("ml_specialist_agent")

# --- Agent Definition ---
AGENT_ID = "ml-specialist-agent-01"
TASK_TOPIC = f"persistent://public/default/q.agentq.tasks.{AGENT_ID}"

ML_SPECIALIST_SYSTEM_PROMPT = """
You are an ML Specialist AI with access to advanced machine learning capabilities. Your purpose is to help users leverage the full suite of ML technologies available in the Q Platform.

**Your ML Capabilities:**

**Federated Learning:**
- `start_federated_learning`: Start distributed ML training across multiple agents
- `get_federated_learning_status`: Check the status of federated learning sessions

**AutoML:**
- `start_automl_experiment`: Launch automated model selection and hyperparameter tuning
- `get_automl_status`: Check AutoML experiment progress
- `get_automl_results`: Get results from completed experiments

**Reinforcement Learning:**
- `start_rl_training`: Train RL agents for workflow optimization
- `get_rl_training_status`: Monitor RL training progress

**Multi-modal AI:**
- `classify_image`: Classify images using computer vision models
- `transcribe_audio`: Convert speech to text
- `analyze_sentiment`: Analyze text sentiment

**Convenience Tools:**
- `train_model_on_data`: Simple interface for training models on datasets
- `optimize_workflow`: Use RL to optimize existing workflows
- `get_ml_capabilities_summary`: Get overview of all ML services

**Your Workflow:**
1. **Understand the Request**: Analyze what type of ML task the user needs
2. **Select the Right Tool**: Choose the most appropriate ML capability
3. **Prepare Parameters**: Format the required parameters (many accept JSON strings)
4. **Execute**: Use the selected tool
5. **Monitor**: Check status and provide updates
6. **Deliver Results**: Present the final results clearly

**Parameter Guidelines:**
- Many tools accept JSON strings for complex configurations
- Base64 encoding is used for image and audio data
- Always check status of long-running tasks
- Provide clear explanations of results

**Best Practices:**
- Start with simple convenience tools for basic tasks
- Use federated learning for distributed scenarios
- AutoML is great for automated model selection
- RL is powerful for optimization problems
- Multi-modal AI handles diverse data types

You are operating cutting-edge ML systems. Be precise and helpful!
"""

def setup_ml_specialist_agent(config: dict):
    """
    Initializes the toolbox and context manager for the ML Specialist agent.
    """
    logger.info("Setting up ML Specialist Agent", agent_id=AGENT_ID)
    
    toolbox = Toolbox()
    
    if ML_TOOLS_AVAILABLE:
        # Register all ML tools
        for tool in ml_tools:
            toolbox.register_tool(tool)
        logger.info(f"Registered {len(ml_tools)} ML tools")
    else:
        logger.warning("ML tools not available - agent will have limited functionality")
    
    context_manager = ContextManager(ignite_addresses=config['ignite']['addresses'], agent_id=AGENT_ID)
    context_manager.connect()
    
    return toolbox, context_manager

def run_ml_specialist_agent(pulsar_client, qpulse_client, llm_config, context_manager):
    """
    The main function for the ML Specialist agent.
    """
    logger.info("Starting ML Specialist Agent...")
    
    from agentQ.app.core.base_agent import BaseAgent
    from shared.q_messaging_schemas.schemas import PromptMessage, ResultMessage, RegistrationMessage
    from pulsar import AvroSchema
    
    # Setup toolbox
    ml_toolbox, _ = setup_ml_specialist_agent(llm_config)
    
    # Create registration producer
    registration_producer = pulsar_client.create_producer(
        "persistent://public/default/q.agentq.registrations",
        schema=AvroSchema(RegistrationMessage)
    )
    
    # Register with manager
    def register_with_manager(producer, agent_id, task_topic):
        registration_msg = RegistrationMessage(
            agent_id=agent_id,
            task_topic=task_topic,
            agent_type="ml_specialist",
            capabilities=["federated_learning", "automl", "reinforcement_learning", "multimodal_ai"],
            status="active"
        )
        producer.send(registration_msg)
        logger.info(f"Registered agent {agent_id} with task topic {task_topic}")
    
    register_with_manager(registration_producer, AGENT_ID, TASK_TOPIC)
    
    # Create result producer
    result_producer = pulsar_client.create_producer(
        llm_config['result_topic'],
        schema=AvroSchema(ResultMessage)
    )
    
    # Create consumer
    consumer = pulsar_client.subscribe(
        TASK_TOPIC, f"agentq-sub-{AGENT_ID}",
        schema=AvroSchema(PromptMessage)
    )
    
    # Create base agent
    base_agent = BaseAgent(
        qpulse_url=qpulse_client.base_url,
        system_prompt=ML_SPECIALIST_SYSTEM_PROMPT,
        tools=[],  # Tools are registered separately
        max_iterations=15
    )
    
    # Set the toolbox
    base_agent.toolbox = ml_toolbox
    
    logger.info(f"ML Specialist Agent ready with {len(ml_toolbox._tools)} tools")
    
    # Message processing loop
    while True:
        try:
            msg = consumer.receive()
            
            try:
                # Process message using base agent
                prompt_message = msg.value()
                logger.info(f"Processing message: {prompt_message.prompt_id}")
                
                # Use base agent to process the prompt
                response = base_agent.process_prompt(
                    prompt=prompt_message.prompt,
                    context=prompt_message.context or {},
                    memory_summary=prompt_message.memory_summary
                )
                
                # Send result
                result_msg = ResultMessage(
                    task_id=prompt_message.prompt_id,
                    result=response,
                    agent_id=AGENT_ID,
                    status="completed"
                )
                
                result_producer.send(result_msg)
                consumer.acknowledge(msg)
                
                logger.info(f"Completed task: {prompt_message.prompt_id}")
                
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                
                # Send error result
                result_msg = ResultMessage(
                    task_id=getattr(msg.value(), 'prompt_id', 'unknown'),
                    result=f"Error: {str(e)}",
                    agent_id=AGENT_ID,
                    status="failed"
                )
                
                result_producer.send(result_msg)
                consumer.acknowledge(msg)
                
        except KeyboardInterrupt:
            logger.info("ML Specialist Agent shutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in ML Specialist Agent: {e}", exc_info=True)
    
    # Cleanup
    consumer.close()
    result_producer.close()
    registration_producer.close()

if __name__ == "__main__":
    # This allows running the agent standalone for testing
    print("ML Specialist Agent - use via main agent runner") 