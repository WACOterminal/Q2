"""
ML Tools for Agent Integration

This module provides tool wrappers for ML capabilities that can be integrated
with the existing agent toolbox system.
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List
from agentQ.app.core.toolbox import Tool

# Import ML tools from managerQ
try:
    from managerQ.app.core.ml_agent_tools import ml_agent_tools
except ImportError:
    # Fallback if managerQ is not available
    ml_agent_tools = None
    logging.warning("ML agent tools not available - ML capabilities will be disabled")

logger = logging.getLogger(__name__)

# ===== FEDERATED LEARNING TOOLS =====

def start_federated_learning_func(
    model_architecture: str,
    dataset_config: str,  # JSON string
    training_config: str,  # JSON string
    participating_agents: Optional[str] = None,  # JSON string of list
    aggregation_strategy: str = "federated_averaging",
    privacy_config: Optional[str] = None,  # JSON string
    config: Dict[str, Any] = None
) -> str:
    """Start federated learning session"""
    
    if ml_agent_tools is None:
        return "Error: ML capabilities not available"
    
    try:
        # Parse JSON strings
        dataset_config_dict = json.loads(dataset_config)
        training_config_dict = json.loads(training_config)
        participating_agents_list = json.loads(participating_agents) if participating_agents else None
        privacy_config_dict = json.loads(privacy_config) if privacy_config else None
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_agent_tools.start_federated_learning(
                    model_architecture=model_architecture,
                    dataset_config=dataset_config_dict,
                    training_config=training_config_dict,
                    participating_agents=participating_agents_list,
                    aggregation_strategy=aggregation_strategy,
                    privacy_config=privacy_config_dict
                )
            )
            return result
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to start federated learning: {e}")
        return f"Error: {str(e)}"

def get_federated_learning_status_func(session_id: str, config: Dict[str, Any] = None) -> str:
    """Get federated learning session status"""
    
    if ml_agent_tools is None:
        return "Error: ML capabilities not available"
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_agent_tools.get_federated_learning_status(session_id)
            )
            return result
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to get federated learning status: {e}")
        return f"Error: {str(e)}"

# ===== AUTOML TOOLS =====

def start_automl_experiment_func(
    experiment_name: str,
    model_type: str,
    dataset_config: str,  # JSON string
    optimization_objective: str = "accuracy",
    training_config: Optional[str] = None,  # JSON string
    n_trials: int = 100,
    timeout_hours: int = 24,
    config: Dict[str, Any] = None
) -> str:
    """Start AutoML experiment"""
    
    if ml_agent_tools is None:
        return "Error: ML capabilities not available"
    
    try:
        # Parse JSON strings
        dataset_config_dict = json.loads(dataset_config)
        training_config_dict = json.loads(training_config) if training_config else None
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_agent_tools.start_automl_experiment(
                    experiment_name=experiment_name,
                    model_type=model_type,
                    dataset_config=dataset_config_dict,
                    optimization_objective=optimization_objective,
                    training_config=training_config_dict,
                    n_trials=n_trials,
                    timeout_hours=timeout_hours
                )
            )
            return result
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to start AutoML experiment: {e}")
        return f"Error: {str(e)}"

def get_automl_status_func(experiment_id: str, config: Dict[str, Any] = None) -> str:
    """Get AutoML experiment status"""
    
    if ml_agent_tools is None:
        return "Error: ML capabilities not available"
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_agent_tools.get_automl_status(experiment_id)
            )
            return result
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to get AutoML status: {e}")
        return f"Error: {str(e)}"

def get_automl_results_func(experiment_id: str, config: Dict[str, Any] = None) -> str:
    """Get AutoML experiment results"""
    
    if ml_agent_tools is None:
        return "Error: ML capabilities not available"
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_agent_tools.get_automl_results(experiment_id)
            )
            return result
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to get AutoML results: {e}")
        return f"Error: {str(e)}"

# ===== REINFORCEMENT LEARNING TOOLS =====

def start_rl_training_func(
    agent_name: str,
    environment_type: str,
    algorithm: str = "ppo",
    training_config: Optional[str] = None,  # JSON string
    environment_config: Optional[str] = None,  # JSON string
    total_timesteps: int = 100000,
    config: Dict[str, Any] = None
) -> str:
    """Start RL training session"""
    
    if ml_agent_tools is None:
        return "Error: ML capabilities not available"
    
    try:
        # Parse JSON strings
        training_config_dict = json.loads(training_config) if training_config else None
        environment_config_dict = json.loads(environment_config) if environment_config else None
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_agent_tools.start_rl_training(
                    agent_name=agent_name,
                    environment_type=environment_type,
                    algorithm=algorithm,
                    training_config=training_config_dict,
                    environment_config=environment_config_dict,
                    total_timesteps=total_timesteps
                )
            )
            return result
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to start RL training: {e}")
        return f"Error: {str(e)}"

def get_rl_training_status_func(session_id: str, config: Dict[str, Any] = None) -> str:
    """Get RL training status"""
    
    if ml_agent_tools is None:
        return "Error: ML capabilities not available"
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_agent_tools.get_rl_training_status(session_id)
            )
            return result
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to get RL training status: {e}")
        return f"Error: {str(e)}"

# ===== MULTIMODAL AI TOOLS =====

def classify_image_func(
    image_data: str,  # Base64 encoded image
    agent_id: str = "default",
    config: Dict[str, Any] = None
) -> str:
    """Classify image using ML models"""
    
    if ml_agent_tools is None:
        return "Error: ML capabilities not available"
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_agent_tools.classify_image(image_data, agent_id)
            )
            return result
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to classify image: {e}")
        return f"Error: {str(e)}"

def transcribe_audio_func(
    audio_data: str,  # Base64 encoded audio
    agent_id: str = "default",
    config: Dict[str, Any] = None
) -> str:
    """Transcribe audio to text"""
    
    if ml_agent_tools is None:
        return "Error: ML capabilities not available"
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_agent_tools.transcribe_audio(audio_data, agent_id)
            )
            return result
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {e}")
        return f"Error: {str(e)}"

def analyze_sentiment_func(
    text: str,
    agent_id: str = "default",
    config: Dict[str, Any] = None
) -> str:
    """Analyze text sentiment"""
    
    if ml_agent_tools is None:
        return "Error: ML capabilities not available"
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_agent_tools.analyze_sentiment(text, agent_id)
            )
            return result
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to analyze sentiment: {e}")
        return f"Error: {str(e)}"

# ===== CONVENIENCE TOOLS =====

def train_model_on_data_func(
    experiment_name: str,
    dataset_config: str,  # JSON string
    model_type: str = "classification",
    optimization_objective: str = "accuracy",
    n_trials: int = 50,
    config: Dict[str, Any] = None
) -> str:
    """Train model on data using AutoML"""
    
    if ml_agent_tools is None:
        return "Error: ML capabilities not available"
    
    try:
        # Parse JSON string
        dataset_config_dict = json.loads(dataset_config)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_agent_tools.train_model_on_data(
                    experiment_name=experiment_name,
                    dataset_config=dataset_config_dict,
                    model_type=model_type,
                    optimization_objective=optimization_objective,
                    n_trials=n_trials
                )
            )
            return result
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        return f"Error: {str(e)}"

def optimize_workflow_func(
    agent_name: str,
    workflow_config: str,  # JSON string
    training_timesteps: int = 50000,
    config: Dict[str, Any] = None
) -> str:
    """Optimize workflow using reinforcement learning"""
    
    if ml_agent_tools is None:
        return "Error: ML capabilities not available"
    
    try:
        # Parse JSON string
        workflow_config_dict = json.loads(workflow_config)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_agent_tools.optimize_workflow(
                    agent_name=agent_name,
                    workflow_config=workflow_config_dict,
                    training_timesteps=training_timesteps
                )
            )
            return result
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to optimize workflow: {e}")
        return f"Error: {str(e)}"

def get_ml_capabilities_summary_func(config: Dict[str, Any] = None) -> str:
    """Get summary of all ML capabilities"""
    
    if ml_agent_tools is None:
        return "Error: ML capabilities not available"
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_agent_tools.get_ml_capabilities_summary()
            )
            return result
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to get ML capabilities summary: {e}")
        return f"Error: {str(e)}"

# ===== TOOL DEFINITIONS =====

# Federated Learning Tools
start_federated_learning_tool = Tool(
    name="start_federated_learning",
    description="Start a federated learning session for distributed ML training across agents. Parameters: model_architecture (str), dataset_config (JSON str), training_config (JSON str), participating_agents (JSON str list, optional), aggregation_strategy (str, optional), privacy_config (JSON str, optional)",
    func=start_federated_learning_func
)

get_federated_learning_status_tool = Tool(
    name="get_federated_learning_status",
    description="Get the status of a federated learning session. Parameters: session_id (str)",
    func=get_federated_learning_status_func
)

# AutoML Tools
start_automl_experiment_tool = Tool(
    name="start_automl_experiment",
    description="Start an AutoML experiment for automated model training. Parameters: experiment_name (str), model_type (str), dataset_config (JSON str), optimization_objective (str, optional), training_config (JSON str, optional), n_trials (int, optional), timeout_hours (int, optional)",
    func=start_automl_experiment_func
)

get_automl_status_tool = Tool(
    name="get_automl_status",
    description="Get the status of an AutoML experiment. Parameters: experiment_id (str)",
    func=get_automl_status_func
)

get_automl_results_tool = Tool(
    name="get_automl_results",
    description="Get the results of a completed AutoML experiment. Parameters: experiment_id (str)",
    func=get_automl_results_func
)

# Reinforcement Learning Tools
start_rl_training_tool = Tool(
    name="start_rl_training",
    description="Start reinforcement learning training for workflow optimization. Parameters: agent_name (str), environment_type (str), algorithm (str, optional), training_config (JSON str, optional), environment_config (JSON str, optional), total_timesteps (int, optional)",
    func=start_rl_training_func
)

get_rl_training_status_tool = Tool(
    name="get_rl_training_status",
    description="Get the status of an RL training session. Parameters: session_id (str)",
    func=get_rl_training_status_func
)

# Multimodal AI Tools
classify_image_tool = Tool(
    name="classify_image",
    description="Classify an image using computer vision models. Parameters: image_data (base64 encoded str), agent_id (str, optional)",
    func=classify_image_func
)

transcribe_audio_tool = Tool(
    name="transcribe_audio",
    description="Transcribe audio to text using speech recognition. Parameters: audio_data (base64 encoded str), agent_id (str, optional)",
    func=transcribe_audio_func
)

analyze_sentiment_tool = Tool(
    name="analyze_sentiment",
    description="Analyze the sentiment of text. Parameters: text (str), agent_id (str, optional)",
    func=analyze_sentiment_func
)

# Convenience Tools
train_model_on_data_tool = Tool(
    name="train_model_on_data",
    description="Train a machine learning model on provided data using AutoML. Parameters: experiment_name (str), dataset_config (JSON str), model_type (str, optional), optimization_objective (str, optional), n_trials (int, optional)",
    func=train_model_on_data_func
)

optimize_workflow_tool = Tool(
    name="optimize_workflow",
    description="Optimize a workflow using reinforcement learning. Parameters: agent_name (str), workflow_config (JSON str), training_timesteps (int, optional)",
    func=optimize_workflow_func
)

get_ml_capabilities_summary_tool = Tool(
    name="get_ml_capabilities_summary",
    description="Get a summary of all available ML capabilities and their current status. No parameters required.",
    func=get_ml_capabilities_summary_func
)

# Collection of all ML tools
ml_tools = [
    start_federated_learning_tool,
    get_federated_learning_status_tool,
    start_automl_experiment_tool,
    get_automl_status_tool,
    get_automl_results_tool,
    start_rl_training_tool,
    get_rl_training_status_tool,
    classify_image_tool,
    transcribe_audio_tool,
    analyze_sentiment_tool,
    train_model_on_data_tool,
    optimize_workflow_tool,
    get_ml_capabilities_summary_tool
]

# Convenience collections
federated_learning_tools = [
    start_federated_learning_tool,
    get_federated_learning_status_tool
]

automl_tools = [
    start_automl_experiment_tool,
    get_automl_status_tool,
    get_automl_results_tool
]

reinforcement_learning_tools = [
    start_rl_training_tool,
    get_rl_training_status_tool
]

multimodal_ai_tools = [
    classify_image_tool,
    transcribe_audio_tool,
    analyze_sentiment_tool
]

convenience_ml_tools = [
    train_model_on_data_tool,
    optimize_workflow_tool,
    get_ml_capabilities_summary_tool
] 