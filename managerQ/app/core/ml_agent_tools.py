"""
ML Agent Tools

This module provides tools for agents to use the advanced ML capabilities:
- Federated Learning tools
- AutoML tools 
- Reinforcement Learning tools
- Multi-modal AI tools
"""

import logging
from typing import Dict, Any, List, Optional
import json
import base64
from datetime import datetime

# ML Services
from managerQ.app.core.federated_learning_orchestrator import (
    federated_learning_orchestrator,
    AggregationStrategy
)
from managerQ.app.core.automl_service import (
    automl_service,
    ModelType,
    OptimizationObjective
)
from managerQ.app.core.reinforcement_learning_service import (
    rl_service,
    RLAlgorithm,
    RLEnvironmentType
)
from managerQ.app.core.multimodal_ai_service import (
    multimodal_ai_service,
    ModalityType,
    ProcessingTask
)
from managerQ.app.core.explainable_ai_service import (
    explainable_ai_service,
    ExplanationType,
    ExplanationMethod,
    ModelType as XAIModelType,
    DataType
)

logger = logging.getLogger(__name__)

class MLAgentTools:
    """ML capabilities tools for agents"""
    
    def __init__(self):
        self.tools = {
            # Federated Learning
            "start_federated_learning": self.start_federated_learning,
            "get_federated_learning_status": self.get_federated_learning_status,
            "list_federated_models": self.list_federated_models,
            
            # AutoML
            "start_automl_experiment": self.start_automl_experiment,
            "get_automl_status": self.get_automl_status,
            "get_automl_results": self.get_automl_results,
            "get_model_leaderboard": self.get_model_leaderboard,
            
            # Reinforcement Learning
            "start_rl_training": self.start_rl_training,
            "get_rl_training_status": self.get_rl_training_status,
            "deploy_rl_agent": self.deploy_rl_agent,
            "get_rl_agent_action": self.get_rl_agent_action,
            
            # Multi-modal AI
            "process_multimodal": self.process_multimodal,
            "get_multimodal_status": self.get_multimodal_status,
            "get_multimodal_result": self.get_multimodal_result,
            "store_multimodal_asset": self.store_multimodal_asset,
            
            # Convenience tools
            "classify_image": self.classify_image,
            "transcribe_audio": self.transcribe_audio,
            "analyze_sentiment": self.analyze_sentiment,
            "train_model_on_data": self.train_model_on_data,
            "optimize_workflow": self.optimize_workflow,
            
            # XAI tools
            "generate_shap_explanation": self.generate_shap_explanation,
            "generate_lime_explanation": self.generate_lime_explanation,
            "analyze_model_fairness": self.analyze_model_fairness,
            "generate_counterfactual_explanations": self.generate_counterfactual_explanations,
            "get_explanation_result": self.get_explanation_result
        }
    
    def get_tool(self, tool_name: str):
        """Get a specific tool by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all available tools"""
        return list(self.tools.keys())
    
    # ===== FEDERATED LEARNING TOOLS =====
    
    async def start_federated_learning(
        self,
        model_architecture: str,
        dataset_config: Dict[str, Any],
        training_config: Dict[str, Any],
        participating_agents: Optional[List[str]] = None,
        aggregation_strategy: str = "federated_averaging",
        privacy_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start federated learning session"""
        
        try:
            # Convert string to enum
            agg_strategy = AggregationStrategy(aggregation_strategy)
            
            session_id = await federated_learning_orchestrator.start_federated_learning_session(
                model_architecture=model_architecture,
                dataset_config=dataset_config,
                training_config=training_config,
                participating_agents=participating_agents,
                aggregation_strategy=agg_strategy,
                privacy_config=privacy_config
            )
            
            return f"Started federated learning session: {session_id}"
        
        except Exception as e:
            logger.error(f"Failed to start federated learning: {e}")
            return f"Error: {str(e)}"
    
    async def get_federated_learning_status(self, session_id: str) -> str:
        """Get federated learning session status"""
        
        try:
            status = await federated_learning_orchestrator.get_session_status(session_id)
            
            if status is None:
                return f"Session {session_id} not found"
            
            return json.dumps(status, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to get federated learning status: {e}")
            return f"Error: {str(e)}"
    
    async def list_federated_models(self, session_id: Optional[str] = None) -> str:
        """List federated learning models"""
        
        try:
            models = await federated_learning_orchestrator.list_model_versions(session_id)
            
            model_list = []
            for model in models:
                model_list.append({
                    "version_id": model.version_id,
                    "model_architecture": model.model_architecture,
                    "performance_metrics": model.performance_metrics,
                    "created_at": model.created_at,
                    "agent_contributions": model.agent_contributions
                })
            
            return json.dumps(model_list, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to list federated models: {e}")
            return f"Error: {str(e)}"
    
    # ===== AUTOML TOOLS =====
    
    async def start_automl_experiment(
        self,
        experiment_name: str,
        model_type: str,
        dataset_config: Dict[str, Any],
        optimization_objective: str = "accuracy",
        training_config: Optional[Dict[str, Any]] = None,
        n_trials: int = 100,
        timeout_hours: int = 24
    ) -> str:
        """Start AutoML experiment"""
        
        try:
            # Convert strings to enums
            model_type_enum = ModelType(model_type)
            objective_enum = OptimizationObjective(optimization_objective)
            
            experiment_id = await automl_service.start_automl_experiment(
                experiment_name=experiment_name,
                model_type=model_type_enum,
                dataset_config=dataset_config,
                optimization_objective=objective_enum,
                training_config=training_config,
                n_trials=n_trials,
                timeout_hours=timeout_hours
            )
            
            return f"Started AutoML experiment: {experiment_id}"
        
        except Exception as e:
            logger.error(f"Failed to start AutoML experiment: {e}")
            return f"Error: {str(e)}"
    
    async def get_automl_status(self, experiment_id: str) -> str:
        """Get AutoML experiment status"""
        
        try:
            status = await automl_service.get_experiment_status(experiment_id)
            
            if status is None:
                return f"Experiment {experiment_id} not found"
            
            return json.dumps(status, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to get AutoML status: {e}")
            return f"Error: {str(e)}"
    
    async def get_automl_results(self, experiment_id: str) -> str:
        """Get AutoML experiment results"""
        
        try:
            results = await automl_service.get_experiment_results(experiment_id)
            
            result_list = []
            for result in results:
                result_list.append({
                    "model_id": result.model_id,
                    "model_name": result.model_name,
                    "performance_metrics": result.performance_metrics,
                    "hyperparameters": result.hyperparameters,
                    "training_time": result.training_time
                })
            
            return json.dumps(result_list, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to get AutoML results: {e}")
            return f"Error: {str(e)}"
    
    async def get_model_leaderboard(self, model_type: Optional[str] = None) -> str:
        """Get model leaderboard"""
        
        try:
            leaderboard = await automl_service.get_model_leaderboard(model_type)
            return json.dumps(leaderboard, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to get model leaderboard: {e}")
            return f"Error: {str(e)}"
    
    # ===== REINFORCEMENT LEARNING TOOLS =====
    
    async def start_rl_training(
        self,
        agent_name: str,
        environment_type: str,
        algorithm: str = "ppo",
        training_config: Optional[Dict[str, Any]] = None,
        environment_config: Optional[Dict[str, Any]] = None,
        total_timesteps: int = 100000
    ) -> str:
        """Start RL training session"""
        
        try:
            # Convert strings to enums
            env_type_enum = RLEnvironmentType(environment_type)
            algorithm_enum = RLAlgorithm(algorithm)
            
            session_id = await rl_service.start_rl_training(
                agent_name=agent_name,
                environment_type=env_type_enum,
                algorithm=algorithm_enum,
                training_config=training_config,
                environment_config=environment_config,
                total_timesteps=total_timesteps
            )
            
            return f"Started RL training session: {session_id}"
        
        except Exception as e:
            logger.error(f"Failed to start RL training: {e}")
            return f"Error: {str(e)}"
    
    async def get_rl_training_status(self, session_id: str) -> str:
        """Get RL training status"""
        
        try:
            status = await rl_service.get_training_status(session_id)
            
            if status is None:
                return f"Training session {session_id} not found"
            
            return json.dumps(status, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to get RL training status: {e}")
            return f"Error: {str(e)}"
    
    async def deploy_rl_agent(self, agent_id: str, target_environment: str = "production") -> str:
        """Deploy trained RL agent"""
        
        try:
            success = await rl_service.deploy_rl_agent(agent_id, target_environment)
            
            if success:
                return f"Successfully deployed RL agent {agent_id} to {target_environment}"
            else:
                return f"Failed to deploy RL agent {agent_id}"
        
        except Exception as e:
            logger.error(f"Failed to deploy RL agent: {e}")
            return f"Error: {str(e)}"
    
    async def get_rl_agent_action(self, agent_id: str, state: Dict[str, Any]) -> str:
        """Get action from RL agent"""
        
        try:
            action = await rl_service.get_rl_agent_action(agent_id, state)
            
            if action is None:
                return f"RL agent {agent_id} not found or unavailable"
            
            return json.dumps(action, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to get RL agent action: {e}")
            return f"Error: {str(e)}"
    
    # ===== MULTIMODAL AI TOOLS =====
    
    async def process_multimodal(
        self,
        modality: str,
        task: str,
        input_data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        agent_id: str = "default"
    ) -> str:
        """Process multimodal request"""
        
        try:
            # Convert strings to enums
            modality_enum = ModalityType(modality)
            task_enum = ProcessingTask(task)
            
            request_id = await multimodal_ai_service.process_multimodal_request(
                agent_id=agent_id,
                modality=modality_enum,
                task=task_enum,
                input_data=input_data,
                parameters=parameters
            )
            
            return f"Queued multimodal request: {request_id}"
        
        except Exception as e:
            logger.error(f"Failed to process multimodal request: {e}")
            return f"Error: {str(e)}"
    
    async def get_multimodal_status(self, request_id: str) -> str:
        """Get multimodal request status"""
        
        try:
            status = await multimodal_ai_service.get_request_status(request_id)
            
            if status is None:
                return f"Request {request_id} not found"
            
            return json.dumps(status, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to get multimodal status: {e}")
            return f"Error: {str(e)}"
    
    async def get_multimodal_result(self, request_id: str) -> str:
        """Get multimodal request result"""
        
        try:
            result = await multimodal_ai_service.get_request_result(request_id)
            
            if result is None:
                return f"Request {request_id} not found or not completed"
            
            return json.dumps(result, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to get multimodal result: {e}")
            return f"Error: {str(e)}"
    
    async def store_multimodal_asset(
        self,
        modality: str,
        content_type: str,
        data: str,  # Base64 encoded data
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Store multimodal asset"""
        
        try:
            # Convert string to enum
            modality_enum = ModalityType(modality)
            
            # Decode base64 data
            decoded_data = base64.b64decode(data)
            
            asset_id = await multimodal_ai_service.store_multimodal_asset(
                modality=modality_enum,
                content_type=content_type,
                data=decoded_data,
                metadata=metadata,
                tags=tags
            )
            
            return f"Stored multimodal asset: {asset_id}"
        
        except Exception as e:
            logger.error(f"Failed to store multimodal asset: {e}")
            return f"Error: {str(e)}"
    
    # ===== CONVENIENCE TOOLS =====
    
    async def classify_image(self, image_data: str, agent_id: str = "default") -> str:
        """Classify image (convenience tool)"""
        
        try:
            request_id = await multimodal_ai_service.process_multimodal_request(
                agent_id=agent_id,
                modality=ModalityType.IMAGE,
                task=ProcessingTask.CLASSIFICATION,
                input_data={"base64": image_data}
            )
            
            return f"Image classification request: {request_id}"
        
        except Exception as e:
            logger.error(f"Failed to classify image: {e}")
            return f"Error: {str(e)}"
    
    async def transcribe_audio(self, audio_data: str, agent_id: str = "default") -> str:
        """Transcribe audio (convenience tool)"""
        
        try:
            request_id = await multimodal_ai_service.process_multimodal_request(
                agent_id=agent_id,
                modality=ModalityType.AUDIO,
                task=ProcessingTask.TRANSCRIPTION,
                input_data={"base64": audio_data}
            )
            
            return f"Audio transcription request: {request_id}"
        
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            return f"Error: {str(e)}"
    
    async def analyze_sentiment(self, text: str, agent_id: str = "default") -> str:
        """Analyze sentiment (convenience tool)"""
        
        try:
            request_id = await multimodal_ai_service.process_multimodal_request(
                agent_id=agent_id,
                modality=ModalityType.TEXT,
                task=ProcessingTask.CLASSIFICATION,
                input_data={"text": text}
            )
            
            return f"Sentiment analysis request: {request_id}"
        
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            return f"Error: {str(e)}"
    
    async def train_model_on_data(
        self,
        experiment_name: str,
        dataset_config: Dict[str, Any],
        model_type: str = "classification",
        optimization_objective: str = "accuracy",
        n_trials: int = 50
    ) -> str:
        """Train model on data (convenience tool)"""
        
        try:
            experiment_id = await automl_service.start_automl_experiment(
                experiment_name=experiment_name,
                model_type=ModelType(model_type),
                dataset_config=dataset_config,
                optimization_objective=OptimizationObjective(optimization_objective),
                n_trials=n_trials
            )
            
            return f"Started model training experiment: {experiment_id}"
        
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            return f"Error: {str(e)}"
    
    async def optimize_workflow(
        self,
        agent_name: str,
        workflow_config: Dict[str, Any],
        training_timesteps: int = 50000
    ) -> str:
        """Optimize workflow using RL (convenience tool)"""
        
        try:
            session_id = await rl_service.start_rl_training(
                agent_name=agent_name,
                environment_type=RLEnvironmentType.WORKFLOW_OPTIMIZATION,
                algorithm=RLAlgorithm.PPO,
                environment_config=workflow_config,
                total_timesteps=training_timesteps
            )
            
            return f"Started workflow optimization training: {session_id}"
        
        except Exception as e:
            logger.error(f"Failed to optimize workflow: {e}")
            return f"Error: {str(e)}"
    
    # ===== XAI TOOLS =====
    
    async def generate_shap_explanation(
        self,
        model_id: str,
        model_type: str,
        input_data: List[float],
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        background_data: Optional[List[List[float]]] = None
    ) -> str:
        """Generate SHAP explanation for a model prediction"""
        
        try:
            request_id = await explainable_ai_service.generate_explanation(
                model_id=model_id,
                model_type=XAIModelType(model_type),
                data_type=DataType.TABULAR,
                explanation_type=ExplanationType.LOCAL,
                explanation_method=ExplanationMethod.SHAP,
                input_data=input_data,
                feature_names=feature_names,
                class_names=class_names,
                background_data=background_data
            )
            
            return f"SHAP explanation request started: {request_id}"
        
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanation: {e}")
            return f"Error: {str(e)}"
    
    async def generate_lime_explanation(
        self,
        model_id: str,
        model_type: str,
        input_data: List[float],
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        background_data: Optional[List[List[float]]] = None
    ) -> str:
        """Generate LIME explanation for a model prediction"""
        
        try:
            request_id = await explainable_ai_service.generate_explanation(
                model_id=model_id,
                model_type=XAIModelType(model_type),
                data_type=DataType.TABULAR,
                explanation_type=ExplanationType.LOCAL,
                explanation_method=ExplanationMethod.LIME,
                input_data=input_data,
                feature_names=feature_names,
                class_names=class_names,
                background_data=background_data
            )
            
            return f"LIME explanation request started: {request_id}"
        
        except Exception as e:
            logger.error(f"Failed to generate LIME explanation: {e}")
            return f"Error: {str(e)}"
    
    async def analyze_model_fairness(
        self,
        model_id: str,
        model_type: str,
        test_data: Dict[str, List[Any]],
        protected_attributes: List[str],
        target_column: str,
        prediction_column: str
    ) -> str:
        """Analyze model fairness across protected attributes"""
        
        try:
            import pandas as pd
            
            # Convert to DataFrame
            test_df = pd.DataFrame(test_data)
            
            request_id = await explainable_ai_service.generate_fairness_analysis(
                model_id=model_id,
                model_type=XAIModelType(model_type),
                test_data=test_df,
                protected_attributes=protected_attributes,
                target_column=target_column,
                prediction_column=prediction_column
            )
            
            return f"Fairness analysis request started: {request_id}"
        
        except Exception as e:
            logger.error(f"Failed to analyze model fairness: {e}")
            return f"Error: {str(e)}"
    
    async def generate_counterfactual_explanations(
        self,
        model_id: str,
        model_type: str,
        input_data: List[float],
        feature_names: Optional[List[str]] = None,
        num_counterfactuals: int = 5
    ) -> str:
        """Generate counterfactual explanations for a prediction"""
        
        try:
            request_id = await explainable_ai_service.generate_counterfactual_explanations(
                model_id=model_id,
                model_type=XAIModelType(model_type),
                input_data=input_data,
                feature_names=feature_names,
                num_counterfactuals=num_counterfactuals
            )
            
            return f"Counterfactual explanations request started: {request_id}"
        
        except Exception as e:
            logger.error(f"Failed to generate counterfactual explanations: {e}")
            return f"Error: {str(e)}"
    
    async def get_explanation_result(
        self,
        request_id: str
    ) -> str:
        """Get the result of an explanation request"""
        
        try:
            result = await explainable_ai_service.get_explanation_result(request_id)
            
            if result:
                # Convert result to JSON-serializable format
                result_dict = {
                    "request_id": result.request_id,
                    "explanation_type": result.explanation_type.value,
                    "explanation_method": result.explanation_method.value,
                    "explanations": result.explanations,
                    "metrics": result.metrics,
                    "interpretation": result.interpretation,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "created_at": result.created_at.isoformat()
                }
                return json.dumps(result_dict, indent=2)
            else:
                return f"No result found for request ID: {request_id}"
        
        except Exception as e:
            logger.error(f"Failed to get explanation result: {e}")
            return f"Error: {str(e)}"
    
    # ===== INTEGRATION HELPERS =====
    
    async def get_ml_capabilities_summary(self) -> str:
        """Get summary of all ML capabilities"""
        
        try:
            # Get metrics from all services
            fl_metrics = await federated_learning_orchestrator.get_federated_learning_metrics()
            automl_metrics = await automl_service.get_automl_metrics()
            rl_metrics = await rl_service.get_rl_metrics()
            multimodal_metrics = await multimodal_ai_service.get_multimodal_metrics()
            
            summary = {
                "federated_learning": fl_metrics,
                "automl": automl_metrics,
                "reinforcement_learning": rl_metrics,
                "multimodal_ai": multimodal_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return json.dumps(summary, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to get ML capabilities summary: {e}")
            return f"Error: {str(e)}"
    
    async def collect_workflow_experience(
        self,
        workflow_id: str,
        agent_id: str,
        workflow_data: Dict[str, Any]
    ) -> str:
        """Collect workflow experience for RL training"""
        
        try:
            await rl_service.collect_workflow_experience(
                workflow_id=workflow_id,
                agent_id=agent_id,
                workflow_data=workflow_data
            )
            
            return f"Collected workflow experience for {workflow_id}"
        
        except Exception as e:
            logger.error(f"Failed to collect workflow experience: {e}")
            return f"Error: {str(e)}"

# Global instance
ml_agent_tools = MLAgentTools() 