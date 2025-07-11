"""
Explainable AI (XAI) Service

This service provides comprehensive explainability features for all ML services:
- SHAP (SHapley Additive exPlanations) for feature importance
- LIME (Local Interpretable Model-agnostic Explanations) for local explanations
- Counterfactual explanations for "what-if" scenarios
- Global model interpretability analysis
- Attention visualization for deep learning models
- Decision tree extraction for neural networks
- Fairness and bias analysis
- Explanation quality assessment
- Interactive explanation dashboards
- Integration with existing ML services
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import joblib
import base64
import io
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Explainability Libraries
import shap
import lime
import lime.lime_tabular
import lime.lime_text
import lime.lime_image
from lime.lime_base import LimeBase
import captum
from captum.attr import IntegratedGradients, GradientShap, DeepLift, Saliency
from captum.attr import LayerConductance, NeuronConductance, LayerActivation
from captum.attr import visualization as viz

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Image processing
from PIL import Image
import cv2

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.q_vectorstore_client.client import VectorStoreClient
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.vault_client import VaultClient

# ML Services integration
from .automl_service import AutoMLService, ModelCandidate
from .reinforcement_learning_service import ReinforcementLearningService
from .multimodal_ai_service import MultiModalAIService
from .model_registry_service import ModelRegistryService

logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    """Types of explanations"""
    GLOBAL = "global"                    # Global model behavior
    LOCAL = "local"                      # Individual prediction explanations
    COUNTERFACTUAL = "counterfactual"    # What-if scenarios
    FEATURE_IMPORTANCE = "feature_importance"  # Feature importance analysis
    ATTENTION = "attention"              # Attention maps for deep learning
    DECISION_TREE = "decision_tree"      # Decision tree extraction
    FAIRNESS = "fairness"                # Fairness and bias analysis
    ADVERSARIAL = "adversarial"          # Adversarial explanations

class ExplanationMethod(Enum):
    """Explanation methods"""
    SHAP = "shap"
    LIME = "lime"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRADIENT_SHAP = "gradient_shap"
    DEEP_LIFT = "deep_lift"
    SALIENCY = "saliency"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"

class ModelType(Enum):
    """Supported model types for explanation"""
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CUSTOM = "custom"

class DataType(Enum):
    """Data types for explanations"""
    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"

@dataclass
class ExplanationRequest:
    """Request for generating explanations"""
    request_id: str
    model_id: str
    model_type: ModelType
    data_type: DataType
    explanation_type: ExplanationType
    explanation_method: ExplanationMethod
    input_data: Any
    feature_names: Optional[List[str]] = None
    class_names: Optional[List[str]] = None
    background_data: Optional[Any] = None
    parameters: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class ExplanationResult:
    """Result of explanation generation"""
    request_id: str
    explanation_type: ExplanationType
    explanation_method: ExplanationMethod
    explanations: Dict[str, Any]
    visualizations: Dict[str, str]  # Base64 encoded images
    metrics: Dict[str, float]
    interpretation: str
    confidence: float
    processing_time: float
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class FairnessAnalysis:
    """Fairness analysis results"""
    protected_attributes: List[str]
    fairness_metrics: Dict[str, float]
    bias_detection: Dict[str, Any]
    recommendations: List[str]
    visualizations: Dict[str, str]

@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation result"""
    original_prediction: Any
    counterfactual_instances: List[Dict[str, Any]]
    feature_changes: List[Dict[str, Any]]
    confidence_changes: List[float]
    feasibility_scores: List[float]

class ExplainableAIService:
    """
    Explainable AI Service for Q Platform
    """
    
    def __init__(self, 
                 storage_path: str = "data/explanations",
                 models_path: str = "models/xai"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Request management
        self.active_requests: Dict[str, ExplanationRequest] = {}
        self.completed_explanations: Dict[str, ExplanationResult] = {}
        
        # Model registry
        self.model_registry = {}
        self.explainer_cache = {}
        
        # Service integrations
        self.automl_service = None
        self.rl_service = None
        self.multimodal_service = None
        self.model_registry_service = None
        
        # Configuration
        self.config = {
            "max_samples_shap": 1000,
            "max_samples_lime": 500,
            "num_features_lime": 10,
            "num_samples_lime": 5000,
            "batch_size": 32,
            "timeout_seconds": 300,
            "cache_explanations": True,
            "explanation_quality_threshold": 0.7
        }
        
        # Performance tracking
        self.xai_metrics = {
            "explanations_generated": 0,
            "local_explanations": 0,
            "global_explanations": 0,
            "counterfactual_explanations": 0,
            "fairness_analyses": 0,
            "average_processing_time": 0.0,
            "explanation_quality_scores": [],
            "user_satisfaction_scores": []
        }
        
        # Background tasks
        self.background_tasks: set = set()
        
        logger.info("Explainable AI Service initialized")
    
    async def initialize(self):
        """Initialize the XAI service"""
        logger.info("Initializing Explainable AI Service")
        
        # Initialize service integrations
        await self._initialize_service_integrations()
        
        # Load pre-trained explainers
        await self._load_explainer_cache()
        
        # Subscribe to model events
        await self._subscribe_to_model_events()
        
        # Start background tasks
        await self._start_background_tasks()
        
        logger.info("Explainable AI Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the XAI service"""
        logger.info("Shutting down Explainable AI Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Save explainer cache
        await self._save_explainer_cache()
        
        logger.info("Explainable AI Service shutdown complete")
    
    # ===== CORE EXPLANATION METHODS =====
    
    async def generate_explanation(
        self,
        model_id: str,
        model_type: ModelType,
        data_type: DataType,
        explanation_type: ExplanationType,
        explanation_method: ExplanationMethod,
        input_data: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        background_data: Optional[Any] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate explanation for a model prediction
        
        Args:
            model_id: ID of the model to explain
            model_type: Type of the model
            data_type: Type of input data
            explanation_type: Type of explanation to generate
            explanation_method: Method to use for explanation
            input_data: Input data for explanation
            feature_names: Names of features
            class_names: Names of classes
            background_data: Background data for explanation
            parameters: Additional parameters
            
        Returns:
            Request ID for the explanation
        """
        request_id = f"explain_{uuid.uuid4().hex[:12]}"
        
        # Create explanation request
        request = ExplanationRequest(
            request_id=request_id,
            model_id=model_id,
            model_type=model_type,
            data_type=data_type,
            explanation_type=explanation_type,
            explanation_method=explanation_method,
            input_data=input_data,
            feature_names=feature_names,
            class_names=class_names,
            background_data=background_data,
            parameters=parameters or {}
        )
        
        # Store request
        self.active_requests[request_id] = request
        
        # Process explanation asynchronously
        task = asyncio.create_task(self._process_explanation_request(request))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        logger.info(f"Started explanation generation: {request_id}")
        return request_id
    
    async def _process_explanation_request(self, request: ExplanationRequest):
        """Process an explanation request"""
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Processing explanation request: {request.request_id}")
            
            # Load model
            model = await self._load_model(request.model_id, request.model_type)
            
            # Generate explanation
            if request.explanation_method == ExplanationMethod.SHAP:
                result = await self._generate_shap_explanation(request, model)
            elif request.explanation_method == ExplanationMethod.LIME:
                result = await self._generate_lime_explanation(request, model)
            elif request.explanation_method == ExplanationMethod.INTEGRATED_GRADIENTS:
                result = await self._generate_integrated_gradients_explanation(request, model)
            elif request.explanation_method == ExplanationMethod.GRADIENT_SHAP:
                result = await self._generate_gradient_shap_explanation(request, model)
            elif request.explanation_method == ExplanationMethod.DEEP_LIFT:
                result = await self._generate_deep_lift_explanation(request, model)
            elif request.explanation_method == ExplanationMethod.SALIENCY:
                result = await self._generate_saliency_explanation(request, model)
            elif request.explanation_method == ExplanationMethod.PERMUTATION_IMPORTANCE:
                result = await self._generate_permutation_importance_explanation(request, model)
            elif request.explanation_method == ExplanationMethod.PARTIAL_DEPENDENCE:
                result = await self._generate_partial_dependence_explanation(request, model)
            else:
                raise ValueError(f"Unsupported explanation method: {request.explanation_method}")
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            result.processing_time = processing_time
            
            # Store result
            self.completed_explanations[request.request_id] = result
            
            # Remove from active requests
            del self.active_requests[request.request_id]
            
            # Update metrics
            self.xai_metrics["explanations_generated"] += 1
            self.xai_metrics["average_processing_time"] = (
                (self.xai_metrics["average_processing_time"] * (self.xai_metrics["explanations_generated"] - 1) + processing_time) 
                / self.xai_metrics["explanations_generated"]
            )
            
            if request.explanation_type == ExplanationType.LOCAL:
                self.xai_metrics["local_explanations"] += 1
            elif request.explanation_type == ExplanationType.GLOBAL:
                self.xai_metrics["global_explanations"] += 1
            elif request.explanation_type == ExplanationType.COUNTERFACTUAL:
                self.xai_metrics["counterfactual_explanations"] += 1
            elif request.explanation_type == ExplanationType.FAIRNESS:
                self.xai_metrics["fairness_analyses"] += 1
            
            # Publish explanation completion event
            await self._publish_explanation_event(request.request_id, "completed")
            
            logger.info(f"Explanation completed: {request.request_id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing explanation request {request.request_id}: {e}", exc_info=True)
            
            # Create error result
            error_result = ExplanationResult(
                request_id=request.request_id,
                explanation_type=request.explanation_type,
                explanation_method=request.explanation_method,
                explanations={"error": str(e)},
                visualizations={},
                metrics={},
                interpretation=f"Error generating explanation: {str(e)}",
                confidence=0.0,
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
            
            self.completed_explanations[request.request_id] = error_result
            
            # Remove from active requests
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            
            # Publish error event
            await self._publish_explanation_event(request.request_id, "error")
    
    # ===== SHAP EXPLANATIONS =====
    
    async def _generate_shap_explanation(self, request: ExplanationRequest, model) -> ExplanationResult:
        """Generate SHAP explanation"""
        
        logger.info(f"Generating SHAP explanation for {request.request_id}")
        
        try:
            # Get or create SHAP explainer
            explainer = await self._get_shap_explainer(request, model)
            
            # Generate SHAP values
            if request.data_type == DataType.TABULAR:
                shap_values = explainer.shap_values(request.input_data)
            elif request.data_type == DataType.IMAGE:
                shap_values = explainer.shap_values(request.input_data)
            elif request.data_type == DataType.TEXT:
                shap_values = explainer.shap_values([request.input_data])
            else:
                raise ValueError(f"Unsupported data type for SHAP: {request.data_type}")
            
            # Generate visualizations
            visualizations = await self._create_shap_visualizations(
                shap_values, request.input_data, request.feature_names, request.class_names
            )
            
            # Create interpretation
            interpretation = await self._interpret_shap_values(shap_values, request.feature_names)
            
            # Calculate metrics
            metrics = await self._calculate_shap_metrics(shap_values)
            
            return ExplanationResult(
                request_id=request.request_id,
                explanation_type=request.explanation_type,
                explanation_method=request.explanation_method,
                explanations={"shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values},
                visualizations=visualizations,
                metrics=metrics,
                interpretation=interpretation,
                confidence=metrics.get("explanation_quality", 0.8),
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}", exc_info=True)
            raise
    
    async def _get_shap_explainer(self, request: ExplanationRequest, model):
        """Get or create SHAP explainer"""
        
        cache_key = f"{request.model_id}_{request.explanation_method.value}_{request.data_type.value}"
        
        if cache_key in self.explainer_cache:
            return self.explainer_cache[cache_key]
        
        # Create appropriate SHAP explainer
        if request.data_type == DataType.TABULAR:
            if request.model_type == ModelType.SKLEARN:
                explainer = shap.TreeExplainer(model)
            elif request.model_type == ModelType.PYTORCH:
                explainer = shap.DeepExplainer(model, request.background_data)
            else:
                # Use KernelExplainer for generic models
                explainer = shap.KernelExplainer(model.predict, request.background_data)
        
        elif request.data_type == DataType.IMAGE:
            explainer = shap.DeepExplainer(model, request.background_data)
        
        elif request.data_type == DataType.TEXT:
            explainer = shap.Explainer(model, request.background_data)
        
        else:
            raise ValueError(f"Unsupported data type for SHAP: {request.data_type}")
        
        # Cache explainer
        self.explainer_cache[cache_key] = explainer
        
        return explainer
    
    async def _create_shap_visualizations(
        self,
        shap_values: np.ndarray,
        input_data: Any,
        feature_names: Optional[List[str]],
        class_names: Optional[List[str]]
    ) -> Dict[str, str]:
        """Create SHAP visualizations"""
        
        visualizations = {}
        
        try:
            # Waterfall plot
            fig, ax = plt.subplots(figsize=(10, 8))
            if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
                shap.waterfall_plot(shap_values[0], feature_names=feature_names, show=False)
            else:
                shap.waterfall_plot(shap_values, feature_names=feature_names, show=False)
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            visualizations['waterfall'] = image_base64
            plt.close()
            
            # Summary plot
            fig, ax = plt.subplots(figsize=(10, 8))
            if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
                shap.summary_plot(shap_values, input_data, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, input_data, feature_names=feature_names, show=False)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            visualizations['summary'] = image_base64
            plt.close()
            
            # Bar plot
            fig, ax = plt.subplots(figsize=(10, 8))
            if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
                shap.bar_plot(shap_values[0], feature_names=feature_names, show=False)
            else:
                shap.bar_plot(shap_values, feature_names=feature_names, show=False)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            visualizations['bar'] = image_base64
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating SHAP visualizations: {e}")
        
        return visualizations
    
    async def _interpret_shap_values(self, shap_values: np.ndarray, feature_names: Optional[List[str]]) -> str:
        """Interpret SHAP values"""
        
        try:
            if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
                # Multi-class case
                mean_importance = np.mean(np.abs(shap_values), axis=0)
            else:
                # Binary classification or regression
                mean_importance = np.abs(shap_values)
            
            # Get top features
            top_indices = np.argsort(mean_importance)[-5:][::-1]
            
            interpretation = "The most important features for this prediction are:\n"
            for i, idx in enumerate(top_indices):
                feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
                importance = mean_importance[idx]
                interpretation += f"{i+1}. {feature_name}: {importance:.4f}\n"
            
            return interpretation
            
        except Exception as e:
            logger.warning(f"Error interpreting SHAP values: {e}")
            return "Unable to generate interpretation due to an error."
    
    async def _calculate_shap_metrics(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate SHAP-based metrics"""
        
        try:
            metrics = {}
            
            # Feature importance distribution
            if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
                importance_values = np.abs(shap_values).mean(axis=0)
            else:
                importance_values = np.abs(shap_values)
            
            metrics['mean_importance'] = float(np.mean(importance_values))
            metrics['max_importance'] = float(np.max(importance_values))
            metrics['importance_std'] = float(np.std(importance_values))
            
            # Explanation quality (based on importance distribution)
            # Higher values indicate more concentrated importance
            metrics['explanation_quality'] = float(np.max(importance_values) / (np.sum(importance_values) + 1e-8))
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating SHAP metrics: {e}")
            return {}
    
    # ===== LIME EXPLANATIONS =====
    
    async def _generate_lime_explanation(self, request: ExplanationRequest, model) -> ExplanationResult:
        """Generate LIME explanation"""
        
        logger.info(f"Generating LIME explanation for {request.request_id}")
        
        try:
            # Create appropriate LIME explainer
            if request.data_type == DataType.TABULAR:
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    request.background_data,
                    feature_names=request.feature_names,
                    class_names=request.class_names,
                    mode='classification' if request.class_names else 'regression'
                )
                
                # Generate explanation
                explanation = explainer.explain_instance(
                    request.input_data,
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=self.config["num_features_lime"],
                    num_samples=self.config["num_samples_lime"]
                )
                
            elif request.data_type == DataType.TEXT:
                explainer = lime.lime_text.LimeTextExplainer(
                    class_names=request.class_names,
                    mode='classification' if request.class_names else 'regression'
                )
                
                # Generate explanation
                explanation = explainer.explain_instance(
                    request.input_data,
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=self.config["num_features_lime"],
                    num_samples=self.config["num_samples_lime"]
                )
                
            elif request.data_type == DataType.IMAGE:
                explainer = lime.lime_image.LimeImageExplainer()
                
                # Generate explanation
                explanation = explainer.explain_instance(
                    request.input_data,
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=self.config["num_features_lime"],
                    num_samples=self.config["num_samples_lime"]
                )
                
            else:
                raise ValueError(f"Unsupported data type for LIME: {request.data_type}")
            
            # Extract explanation data
            explanation_data = {
                'as_list': explanation.as_list(),
                'as_map': explanation.as_map(),
                'score': explanation.score,
                'local_exp': explanation.local_exp,
                'intercept': explanation.intercept
            }
            
            # Generate visualizations
            visualizations = await self._create_lime_visualizations(explanation, request.data_type)
            
            # Create interpretation
            interpretation = await self._interpret_lime_explanation(explanation)
            
            # Calculate metrics
            metrics = await self._calculate_lime_metrics(explanation)
            
            return ExplanationResult(
                request_id=request.request_id,
                explanation_type=request.explanation_type,
                explanation_method=request.explanation_method,
                explanations=explanation_data,
                visualizations=visualizations,
                metrics=metrics,
                interpretation=interpretation,
                confidence=metrics.get("explanation_quality", 0.8),
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}", exc_info=True)
            raise
    
    async def _create_lime_visualizations(self, explanation, data_type: DataType) -> Dict[str, str]:
        """Create LIME visualizations"""
        
        visualizations = {}
        
        try:
            if data_type == DataType.TABULAR:
                # Feature importance plot
                fig, ax = plt.subplots(figsize=(10, 8))
                explanation.show_in_notebook(show_table=True)
                
                # Create bar plot
                features, importances = zip(*explanation.as_list())
                y_pos = np.arange(len(features))
                
                colors = ['red' if imp < 0 else 'green' for imp in importances]
                bars = ax.barh(y_pos, importances, color=colors, alpha=0.7)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('Feature Importance')
                ax.set_title('LIME Feature Importance')
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, imp) in enumerate(zip(bars, importances)):
                    ax.text(bar.get_width() + 0.01 if imp >= 0 else bar.get_width() - 0.01,
                           bar.get_y() + bar.get_height()/2, 
                           f'{imp:.3f}', 
                           ha='left' if imp >= 0 else 'right', 
                           va='center')
                
                plt.tight_layout()
                
                # Save to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                visualizations['feature_importance'] = image_base64
                plt.close()
            
            elif data_type == DataType.IMAGE:
                # Get image explanation
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0],
                    positive_only=True,
                    num_features=10,
                    hide_rest=False
                )
                
                # Create subplot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Original image
                ax1.imshow(temp)
                ax1.set_title('Original Image')
                ax1.axis('off')
                
                # Explanation mask
                ax2.imshow(mask)
                ax2.set_title('LIME Explanation')
                ax2.axis('off')
                
                plt.tight_layout()
                
                # Save to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                visualizations['image_explanation'] = image_base64
                plt.close()
                
        except Exception as e:
            logger.warning(f"Error creating LIME visualizations: {e}")
        
        return visualizations
    
    async def _interpret_lime_explanation(self, explanation) -> str:
        """Interpret LIME explanation"""
        
        try:
            as_list = explanation.as_list()
            
            interpretation = "LIME Explanation:\n"
            interpretation += f"Model confidence: {explanation.score:.4f}\n\n"
            
            positive_features = [(feat, imp) for feat, imp in as_list if imp > 0]
            negative_features = [(feat, imp) for feat, imp in as_list if imp < 0]
            
            if positive_features:
                interpretation += "Features supporting the prediction:\n"
                for feat, imp in positive_features:
                    interpretation += f"  • {feat}: {imp:.4f}\n"
            
            if negative_features:
                interpretation += "\nFeatures opposing the prediction:\n"
                for feat, imp in negative_features:
                    interpretation += f"  • {feat}: {imp:.4f}\n"
            
            return interpretation
            
        except Exception as e:
            logger.warning(f"Error interpreting LIME explanation: {e}")
            return "Unable to generate interpretation due to an error."
    
    async def _calculate_lime_metrics(self, explanation) -> Dict[str, float]:
        """Calculate LIME-based metrics"""
        
        try:
            metrics = {}
            
            # Extract feature importances
            as_list = explanation.as_list()
            importances = [imp for _, imp in as_list]
            
            metrics['mean_importance'] = float(np.mean(np.abs(importances)))
            metrics['max_importance'] = float(np.max(np.abs(importances)))
            metrics['importance_std'] = float(np.std(importances))
            
            # Explanation quality
            metrics['explanation_quality'] = float(explanation.score)
            
            # Feature distribution
            positive_count = sum(1 for imp in importances if imp > 0)
            negative_count = sum(1 for imp in importances if imp < 0)
            metrics['positive_features'] = float(positive_count)
            metrics['negative_features'] = float(negative_count)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating LIME metrics: {e}")
            return {}
    
    # ===== CAPTUM EXPLANATIONS =====
    
    async def _generate_integrated_gradients_explanation(self, request: ExplanationRequest, model) -> ExplanationResult:
        """Generate Integrated Gradients explanation"""
        
        logger.info(f"Generating Integrated Gradients explanation for {request.request_id}")
        
        try:
            # Create Integrated Gradients explainer
            ig = IntegratedGradients(model)
            
            # Convert input to tensor
            input_tensor = torch.tensor(request.input_data, dtype=torch.float32, requires_grad=True)
            
            # Generate baseline
            baseline = torch.zeros_like(input_tensor)
            
            # Generate attributions
            attributions = ig.attribute(input_tensor, baseline, target=None)
            
            # Convert to numpy
            attributions_np = attributions.detach().numpy()
            
            # Generate visualizations
            visualizations = await self._create_captum_visualizations(
                attributions_np, request.input_data, request.feature_names, "Integrated Gradients"
            )
            
            # Create interpretation
            interpretation = await self._interpret_captum_attributions(attributions_np, request.feature_names)
            
            # Calculate metrics
            metrics = await self._calculate_captum_metrics(attributions_np)
            
            return ExplanationResult(
                request_id=request.request_id,
                explanation_type=request.explanation_type,
                explanation_method=request.explanation_method,
                explanations={"attributions": attributions_np.tolist()},
                visualizations=visualizations,
                metrics=metrics,
                interpretation=interpretation,
                confidence=metrics.get("explanation_quality", 0.8),
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Error generating Integrated Gradients explanation: {e}", exc_info=True)
            raise
    
    async def _generate_gradient_shap_explanation(self, request: ExplanationRequest, model) -> ExplanationResult:
        """Generate Gradient SHAP explanation"""
        
        logger.info(f"Generating Gradient SHAP explanation for {request.request_id}")
        
        try:
            # Create Gradient SHAP explainer
            gs = GradientShap(model)
            
            # Convert input to tensor
            input_tensor = torch.tensor(request.input_data, dtype=torch.float32, requires_grad=True)
            
            # Generate baseline
            baseline = torch.zeros_like(input_tensor)
            
            # Generate attributions
            attributions = gs.attribute(input_tensor, baseline, target=None)
            
            # Convert to numpy
            attributions_np = attributions.detach().numpy()
            
            # Generate visualizations
            visualizations = await self._create_captum_visualizations(
                attributions_np, request.input_data, request.feature_names, "Gradient SHAP"
            )
            
            # Create interpretation
            interpretation = await self._interpret_captum_attributions(attributions_np, request.feature_names)
            
            # Calculate metrics
            metrics = await self._calculate_captum_metrics(attributions_np)
            
            return ExplanationResult(
                request_id=request.request_id,
                explanation_type=request.explanation_type,
                explanation_method=request.explanation_method,
                explanations={"attributions": attributions_np.tolist()},
                visualizations=visualizations,
                metrics=metrics,
                interpretation=interpretation,
                confidence=metrics.get("explanation_quality", 0.8),
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Error generating Gradient SHAP explanation: {e}", exc_info=True)
            raise
    
    async def _generate_deep_lift_explanation(self, request: ExplanationRequest, model) -> ExplanationResult:
        """Generate DeepLift explanation"""
        
        logger.info(f"Generating DeepLift explanation for {request.request_id}")
        
        try:
            # Create DeepLift explainer
            dl = DeepLift(model)
            
            # Convert input to tensor
            input_tensor = torch.tensor(request.input_data, dtype=torch.float32, requires_grad=True)
            
            # Generate baseline
            baseline = torch.zeros_like(input_tensor)
            
            # Generate attributions
            attributions = dl.attribute(input_tensor, baseline, target=None)
            
            # Convert to numpy
            attributions_np = attributions.detach().numpy()
            
            # Generate visualizations
            visualizations = await self._create_captum_visualizations(
                attributions_np, request.input_data, request.feature_names, "DeepLift"
            )
            
            # Create interpretation
            interpretation = await self._interpret_captum_attributions(attributions_np, request.feature_names)
            
            # Calculate metrics
            metrics = await self._calculate_captum_metrics(attributions_np)
            
            return ExplanationResult(
                request_id=request.request_id,
                explanation_type=request.explanation_type,
                explanation_method=request.explanation_method,
                explanations={"attributions": attributions_np.tolist()},
                visualizations=visualizations,
                metrics=metrics,
                interpretation=interpretation,
                confidence=metrics.get("explanation_quality", 0.8),
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Error generating DeepLift explanation: {e}", exc_info=True)
            raise
    
    async def _generate_saliency_explanation(self, request: ExplanationRequest, model) -> ExplanationResult:
        """Generate Saliency explanation"""
        
        logger.info(f"Generating Saliency explanation for {request.request_id}")
        
        try:
            # Create Saliency explainer
            saliency = Saliency(model)
            
            # Convert input to tensor
            input_tensor = torch.tensor(request.input_data, dtype=torch.float32, requires_grad=True)
            
            # Generate attributions
            attributions = saliency.attribute(input_tensor, target=None)
            
            # Convert to numpy
            attributions_np = attributions.detach().numpy()
            
            # Generate visualizations
            visualizations = await self._create_captum_visualizations(
                attributions_np, request.input_data, request.feature_names, "Saliency"
            )
            
            # Create interpretation
            interpretation = await self._interpret_captum_attributions(attributions_np, request.feature_names)
            
            # Calculate metrics
            metrics = await self._calculate_captum_metrics(attributions_np)
            
            return ExplanationResult(
                request_id=request.request_id,
                explanation_type=request.explanation_type,
                explanation_method=request.explanation_method,
                explanations={"attributions": attributions_np.tolist()},
                visualizations=visualizations,
                metrics=metrics,
                interpretation=interpretation,
                confidence=metrics.get("explanation_quality", 0.8),
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Error generating Saliency explanation: {e}", exc_info=True)
            raise
    
    async def _create_captum_visualizations(
        self,
        attributions: np.ndarray,
        input_data: Any,
        feature_names: Optional[List[str]],
        method_name: str
    ) -> Dict[str, str]:
        """Create Captum visualizations"""
        
        visualizations = {}
        
        try:
            # Attribution bar plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get top features
            abs_attributions = np.abs(attributions)
            top_indices = np.argsort(abs_attributions)[-10:][::-1]
            
            top_attributions = attributions[top_indices]
            top_features = [feature_names[i] if feature_names else f"Feature {i}" for i in top_indices]
            
            # Create bar plot
            colors = ['red' if attr < 0 else 'green' for attr in top_attributions]
            bars = ax.barh(range(len(top_features)), top_attributions, color=colors, alpha=0.7)
            
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Attribution Score')
            ax.set_title(f'{method_name} Feature Attributions')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, attr) in enumerate(zip(bars, top_attributions)):
                ax.text(bar.get_width() + 0.01 if attr >= 0 else bar.get_width() - 0.01,
                       bar.get_y() + bar.get_height()/2,
                       f'{attr:.4f}',
                       ha='left' if attr >= 0 else 'right',
                       va='center')
            
            plt.tight_layout()
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            visualizations['attributions'] = image_base64
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating Captum visualizations: {e}")
        
        return visualizations
    
    async def _interpret_captum_attributions(self, attributions: np.ndarray, feature_names: Optional[List[str]]) -> str:
        """Interpret Captum attributions"""
        
        try:
            # Get top positive and negative attributions
            positive_indices = np.where(attributions > 0)[0]
            negative_indices = np.where(attributions < 0)[0]
            
            positive_attributions = attributions[positive_indices]
            negative_attributions = attributions[negative_indices]
            
            # Sort by absolute value
            top_positive = positive_indices[np.argsort(positive_attributions)[-3:][::-1]]
            top_negative = negative_indices[np.argsort(np.abs(negative_attributions))[-3:][::-1]]
            
            interpretation = "Attribution Analysis:\n\n"
            
            if len(top_positive) > 0:
                interpretation += "Top features contributing positively:\n"
                for idx in top_positive:
                    feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
                    attribution = attributions[idx]
                    interpretation += f"  • {feature_name}: {attribution:.4f}\n"
            
            if len(top_negative) > 0:
                interpretation += "\nTop features contributing negatively:\n"
                for idx in top_negative:
                    feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
                    attribution = attributions[idx]
                    interpretation += f"  • {feature_name}: {attribution:.4f}\n"
            
            return interpretation
            
        except Exception as e:
            logger.warning(f"Error interpreting Captum attributions: {e}")
            return "Unable to generate interpretation due to an error."
    
    async def _calculate_captum_metrics(self, attributions: np.ndarray) -> Dict[str, float]:
        """Calculate Captum-based metrics"""
        
        try:
            metrics = {}
            
            # Basic statistics
            metrics['mean_attribution'] = float(np.mean(attributions))
            metrics['max_attribution'] = float(np.max(attributions))
            metrics['min_attribution'] = float(np.min(attributions))
            metrics['attribution_std'] = float(np.std(attributions))
            
            # Positive/negative attribution balance
            positive_count = np.sum(attributions > 0)
            negative_count = np.sum(attributions < 0)
            metrics['positive_attributions'] = float(positive_count)
            metrics['negative_attributions'] = float(negative_count)
            
            # Attribution concentration
            abs_attributions = np.abs(attributions)
            total_attribution = np.sum(abs_attributions)
            max_attribution = np.max(abs_attributions)
            metrics['attribution_concentration'] = float(max_attribution / (total_attribution + 1e-8))
            
            # Explanation quality (based on attribution distribution)
            metrics['explanation_quality'] = float(1.0 - (np.std(abs_attributions) / (np.mean(abs_attributions) + 1e-8)))
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating Captum metrics: {e}")
            return {}
    
    # ===== PERMUTATION IMPORTANCE =====
    
    async def _generate_permutation_importance_explanation(self, request: ExplanationRequest, model) -> ExplanationResult:
        """Generate Permutation Importance explanation"""
        
        logger.info(f"Generating Permutation Importance explanation for {request.request_id}")
        
        try:
            from sklearn.inspection import permutation_importance
            
            # Calculate permutation importance
            result = permutation_importance(
                model, 
                request.input_data, 
                request.background_data if request.background_data is not None else np.ones(len(request.input_data)),
                n_repeats=10,
                random_state=42
            )
            
            # Extract importance scores
            importances = result.importances_mean
            std_importances = result.importances_std
            
            # Generate visualizations
            visualizations = await self._create_permutation_importance_visualizations(
                importances, std_importances, request.feature_names
            )
            
            # Create interpretation
            interpretation = await self._interpret_permutation_importance(importances, request.feature_names)
            
            # Calculate metrics
            metrics = await self._calculate_permutation_importance_metrics(importances, std_importances)
            
            return ExplanationResult(
                request_id=request.request_id,
                explanation_type=request.explanation_type,
                explanation_method=request.explanation_method,
                explanations={
                    "importances": importances.tolist(),
                    "std_importances": std_importances.tolist()
                },
                visualizations=visualizations,
                metrics=metrics,
                interpretation=interpretation,
                confidence=metrics.get("explanation_quality", 0.8),
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Error generating Permutation Importance explanation: {e}", exc_info=True)
            raise
    
    async def _create_permutation_importance_visualizations(
        self,
        importances: np.ndarray,
        std_importances: np.ndarray,
        feature_names: Optional[List[str]]
    ) -> Dict[str, str]:
        """Create Permutation Importance visualizations"""
        
        visualizations = {}
        
        try:
            # Importance bar plot with error bars
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get top features
            top_indices = np.argsort(importances)[-10:][::-1]
            top_importances = importances[top_indices]
            top_stds = std_importances[top_indices]
            top_features = [feature_names[i] if feature_names else f"Feature {i}" for i in top_indices]
            
            # Create bar plot
            bars = ax.barh(range(len(top_features)), top_importances, 
                          xerr=top_stds, capsize=5, alpha=0.7, color='steelblue')
            
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Permutation Importance')
            ax.set_title('Permutation Feature Importance')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, imp, std) in enumerate(zip(bars, top_importances, top_stds)):
                ax.text(bar.get_width() + std + 0.01,
                       bar.get_y() + bar.get_height()/2,
                       f'{imp:.4f}±{std:.4f}',
                       ha='left', va='center')
            
            plt.tight_layout()
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            visualizations['permutation_importance'] = image_base64
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating Permutation Importance visualizations: {e}")
        
        return visualizations
    
    async def _interpret_permutation_importance(self, importances: np.ndarray, feature_names: Optional[List[str]]) -> str:
        """Interpret Permutation Importance"""
        
        try:
            # Get top features
            top_indices = np.argsort(importances)[-5:][::-1]
            
            interpretation = "Permutation Importance Analysis:\n\n"
            interpretation += "Features ranked by importance (higher values indicate more important features):\n"
            
            for i, idx in enumerate(top_indices):
                feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
                importance = importances[idx]
                interpretation += f"{i+1}. {feature_name}: {importance:.4f}\n"
            
            interpretation += "\nNote: Permutation importance measures the decrease in model performance "
            interpretation += "when a feature's values are randomly shuffled."
            
            return interpretation
            
        except Exception as e:
            logger.warning(f"Error interpreting Permutation Importance: {e}")
            return "Unable to generate interpretation due to an error."
    
    async def _calculate_permutation_importance_metrics(self, importances: np.ndarray, std_importances: np.ndarray) -> Dict[str, float]:
        """Calculate Permutation Importance metrics"""
        
        try:
            metrics = {}
            
            # Basic statistics
            metrics['mean_importance'] = float(np.mean(importances))
            metrics['max_importance'] = float(np.max(importances))
            metrics['importance_std'] = float(np.std(importances))
            
            # Stability metrics
            metrics['mean_std'] = float(np.mean(std_importances))
            metrics['max_std'] = float(np.max(std_importances))
            
            # Reliability (inverse of coefficient of variation)
            cv = std_importances / (importances + 1e-8)
            metrics['reliability'] = float(1.0 / (np.mean(cv) + 1e-8))
            
            # Explanation quality
            metrics['explanation_quality'] = float(np.max(importances) / (np.sum(importances) + 1e-8))
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating Permutation Importance metrics: {e}")
            return {}
    
    # ===== PARTIAL DEPENDENCE =====
    
    async def _generate_partial_dependence_explanation(self, request: ExplanationRequest, model) -> ExplanationResult:
        """Generate Partial Dependence explanation"""
        
        logger.info(f"Generating Partial Dependence explanation for {request.request_id}")
        
        try:
            from sklearn.inspection import partial_dependence
            
            # Get feature indices for partial dependence
            feature_indices = list(range(min(5, len(request.feature_names or []))))
            
            # Calculate partial dependence
            pd_result = partial_dependence(
                model,
                request.background_data,
                features=feature_indices,
                kind='average'
            )
            
            # Extract partial dependence values
            partial_dependences = pd_result['values']
            grid_values = pd_result['grid_values']
            
            # Generate visualizations
            visualizations = await self._create_partial_dependence_visualizations(
                partial_dependences, grid_values, feature_indices, request.feature_names
            )
            
            # Create interpretation
            interpretation = await self._interpret_partial_dependence(
                partial_dependences, grid_values, feature_indices, request.feature_names
            )
            
            # Calculate metrics
            metrics = await self._calculate_partial_dependence_metrics(partial_dependences)
            
            return ExplanationResult(
                request_id=request.request_id,
                explanation_type=request.explanation_type,
                explanation_method=request.explanation_method,
                explanations={
                    "partial_dependences": [pd.tolist() for pd in partial_dependences],
                    "grid_values": [gv.tolist() for gv in grid_values]
                },
                visualizations=visualizations,
                metrics=metrics,
                interpretation=interpretation,
                confidence=metrics.get("explanation_quality", 0.8),
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Error generating Partial Dependence explanation: {e}", exc_info=True)
            raise
    
    async def _create_partial_dependence_visualizations(
        self,
        partial_dependences: List[np.ndarray],
        grid_values: List[np.ndarray],
        feature_indices: List[int],
        feature_names: Optional[List[str]]
    ) -> Dict[str, str]:
        """Create Partial Dependence visualizations"""
        
        visualizations = {}
        
        try:
            # Create subplots
            n_features = len(feature_indices)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_features == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (pd_values, grid_vals, feat_idx) in enumerate(zip(partial_dependences, grid_values, feature_indices)):
                ax = axes[i]
                
                # Plot partial dependence
                ax.plot(grid_vals, pd_values, linewidth=2, color='steelblue')
                ax.fill_between(grid_vals, pd_values, alpha=0.3, color='steelblue')
                
                # Formatting
                feature_name = feature_names[feat_idx] if feature_names else f"Feature {feat_idx}"
                ax.set_xlabel(feature_name)
                ax.set_ylabel('Partial Dependence')
                ax.set_title(f'Partial Dependence: {feature_name}')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            visualizations['partial_dependence'] = image_base64
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating Partial Dependence visualizations: {e}")
        
        return visualizations
    
    async def _interpret_partial_dependence(
        self,
        partial_dependences: List[np.ndarray],
        grid_values: List[np.ndarray],
        feature_indices: List[int],
        feature_names: Optional[List[str]]
    ) -> str:
        """Interpret Partial Dependence"""
        
        try:
            interpretation = "Partial Dependence Analysis:\n\n"
            
            for i, (pd_values, grid_vals, feat_idx) in enumerate(zip(partial_dependences, grid_values, feature_indices)):
                feature_name = feature_names[feat_idx] if feature_names else f"Feature {feat_idx}"
                
                # Find the range of partial dependence values
                min_pd = np.min(pd_values)
                max_pd = np.max(pd_values)
                range_pd = max_pd - min_pd
                
                interpretation += f"{feature_name}:\n"
                interpretation += f"  • Partial dependence range: {min_pd:.4f} to {max_pd:.4f}\n"
                interpretation += f"  • Impact magnitude: {range_pd:.4f}\n"
                
                # Find the value with maximum impact
                max_impact_idx = np.argmax(pd_values)
                max_impact_value = grid_vals[max_impact_idx]
                interpretation += f"  • Maximum impact at value: {max_impact_value:.4f}\n\n"
            
            interpretation += "Note: Partial dependence shows the expected marginal effect of a feature "
            interpretation += "on the model's predictions, averaged over all other features."
            
            return interpretation
            
        except Exception as e:
            logger.warning(f"Error interpreting Partial Dependence: {e}")
            return "Unable to generate interpretation due to an error."
    
    async def _calculate_partial_dependence_metrics(self, partial_dependences: List[np.ndarray]) -> Dict[str, float]:
        """Calculate Partial Dependence metrics"""
        
        try:
            metrics = {}
            
            # Calculate metrics for each feature
            ranges = []
            variances = []
            
            for pd_values in partial_dependences:
                ranges.append(np.max(pd_values) - np.min(pd_values))
                variances.append(np.var(pd_values))
            
            # Overall metrics
            metrics['mean_range'] = float(np.mean(ranges))
            metrics['max_range'] = float(np.max(ranges))
            metrics['mean_variance'] = float(np.mean(variances))
            metrics['max_variance'] = float(np.max(variances))
            
            # Explanation quality (based on feature impact)
            metrics['explanation_quality'] = float(np.max(ranges) / (np.sum(ranges) + 1e-8))
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating Partial Dependence metrics: {e}")
            return {}
    
    # ===== FAIRNESS ANALYSIS =====
    
    async def generate_fairness_analysis(
        self,
        model_id: str,
        model_type: ModelType,
        test_data: pd.DataFrame,
        protected_attributes: List[str],
        target_column: str,
        prediction_column: str
    ) -> str:
        """Generate fairness analysis"""
        
        request_id = f"fairness_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Generating fairness analysis: {request_id}")
        
        try:
            # Calculate fairness metrics
            fairness_metrics = await self._calculate_fairness_metrics(
                test_data, protected_attributes, target_column, prediction_column
            )
            
            # Detect bias
            bias_detection = await self._detect_bias(
                test_data, protected_attributes, target_column, prediction_column
            )
            
            # Generate recommendations
            recommendations = await self._generate_fairness_recommendations(
                fairness_metrics, bias_detection
            )
            
            # Create visualizations
            visualizations = await self._create_fairness_visualizations(
                test_data, protected_attributes, target_column, prediction_column
            )
            
            # Create fairness analysis result
            fairness_analysis = FairnessAnalysis(
                protected_attributes=protected_attributes,
                fairness_metrics=fairness_metrics,
                bias_detection=bias_detection,
                recommendations=recommendations,
                visualizations=visualizations
            )
            
            # Store result
            self.completed_explanations[request_id] = fairness_analysis
            
            # Update metrics
            self.xai_metrics["fairness_analyses"] += 1
            
            logger.info(f"Fairness analysis completed: {request_id}")
            
            return request_id
            
        except Exception as e:
            logger.error(f"Error generating fairness analysis: {e}", exc_info=True)
            raise
    
    async def _calculate_fairness_metrics(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target_column: str,
        prediction_column: str
    ) -> Dict[str, float]:
        """Calculate fairness metrics"""
        
        try:
            metrics = {}
            
            for attr in protected_attributes:
                unique_values = data[attr].unique()
                
                for val in unique_values:
                    group_data = data[data[attr] == val]
                    
                    # True positive rate
                    tp = len(group_data[(group_data[target_column] == 1) & (group_data[prediction_column] == 1)])
                    fn = len(group_data[(group_data[target_column] == 1) & (group_data[prediction_column] == 0)])
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    # False positive rate
                    fp = len(group_data[(group_data[target_column] == 0) & (group_data[prediction_column] == 1)])
                    tn = len(group_data[(group_data[target_column] == 0) & (group_data[prediction_column] == 0)])
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    # Precision
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    
                    # Recall
                    recall = tpr
                    
                    # F1 score
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # Store metrics
                    metrics[f"{attr}_{val}_tpr"] = tpr
                    metrics[f"{attr}_{val}_fpr"] = fpr
                    metrics[f"{attr}_{val}_precision"] = precision
                    metrics[f"{attr}_{val}_recall"] = recall
                    metrics[f"{attr}_{val}_f1"] = f1
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating fairness metrics: {e}")
            return {}
    
    async def _detect_bias(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target_column: str,
        prediction_column: str
    ) -> Dict[str, Any]:
        """Detect bias in predictions"""
        
        try:
            bias_detection = {}
            
            for attr in protected_attributes:
                unique_values = data[attr].unique()
                
                if len(unique_values) == 2:
                    # Binary protected attribute
                    val1, val2 = unique_values
                    
                    group1 = data[data[attr] == val1]
                    group2 = data[data[attr] == val2]
                    
                    # Demographic parity
                    pos_rate1 = len(group1[group1[prediction_column] == 1]) / len(group1)
                    pos_rate2 = len(group2[group2[prediction_column] == 1]) / len(group2)
                    demographic_parity = abs(pos_rate1 - pos_rate2)
                    
                    # Equalized odds
                    tpr1 = len(group1[(group1[target_column] == 1) & (group1[prediction_column] == 1)]) / len(group1[group1[target_column] == 1])
                    tpr2 = len(group2[(group2[target_column] == 1) & (group2[prediction_column] == 1)]) / len(group2[group2[target_column] == 1])
                    equalized_odds = abs(tpr1 - tpr2)
                    
                    bias_detection[attr] = {
                        "demographic_parity": demographic_parity,
                        "equalized_odds": equalized_odds,
                        "bias_detected": demographic_parity > 0.1 or equalized_odds > 0.1
                    }
            
            return bias_detection
            
        except Exception as e:
            logger.warning(f"Error detecting bias: {e}")
            return {}
    
    async def _generate_fairness_recommendations(
        self,
        fairness_metrics: Dict[str, float],
        bias_detection: Dict[str, Any]
    ) -> List[str]:
        """Generate fairness recommendations"""
        
        recommendations = []
        
        try:
            for attr, detection in bias_detection.items():
                if detection.get("bias_detected", False):
                    recommendations.append(f"Bias detected in {attr}. Consider:")
                    
                    if detection.get("demographic_parity", 0) > 0.1:
                        recommendations.append(f"  • Demographic parity violation in {attr}")
                        recommendations.append("  • Consider rebalancing training data")
                        recommendations.append("  • Apply fairness constraints during training")
                    
                    if detection.get("equalized_odds", 0) > 0.1:
                        recommendations.append(f"  • Equalized odds violation in {attr}")
                        recommendations.append("  • Consider post-processing techniques")
                        recommendations.append("  • Adjust decision thresholds per group")
            
            if not recommendations:
                recommendations.append("No significant bias detected in the analyzed protected attributes.")
                recommendations.append("Continue monitoring fairness metrics as data changes.")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error generating fairness recommendations: {e}")
            return ["Unable to generate recommendations due to an error."]
    
    async def _create_fairness_visualizations(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target_column: str,
        prediction_column: str
    ) -> Dict[str, str]:
        """Create fairness visualizations"""
        
        visualizations = {}
        
        try:
            for attr in protected_attributes:
                # Create fairness metrics bar plot
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                unique_values = data[attr].unique()
                
                # Calculate metrics for each group
                groups = []
                tprs = []
                fprs = []
                precisions = []
                recalls = []
                
                for val in unique_values:
                    group_data = data[data[attr] == val]
                    
                    # Calculate metrics
                    tp = len(group_data[(group_data[target_column] == 1) & (group_data[prediction_column] == 1)])
                    fn = len(group_data[(group_data[target_column] == 1) & (group_data[prediction_column] == 0)])
                    fp = len(group_data[(group_data[target_column] == 0) & (group_data[prediction_column] == 1)])
                    tn = len(group_data[(group_data[target_column] == 0) & (group_data[prediction_column] == 0)])
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tpr
                    
                    groups.append(str(val))
                    tprs.append(tpr)
                    fprs.append(fpr)
                    precisions.append(precision)
                    recalls.append(recall)
                
                # Plot metrics
                x = np.arange(len(groups))
                width = 0.35
                
                # TPR and FPR
                axes[0, 0].bar(x - width/2, tprs, width, label='TPR', alpha=0.8)
                axes[0, 0].bar(x + width/2, fprs, width, label='FPR', alpha=0.8)
                axes[0, 0].set_xlabel('Groups')
                axes[0, 0].set_ylabel('Rate')
                axes[0, 0].set_title(f'TPR and FPR by {attr}')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(groups)
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Precision and Recall
                axes[0, 1].bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
                axes[0, 1].bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
                axes[0, 1].set_xlabel('Groups')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].set_title(f'Precision and Recall by {attr}')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(groups)
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Prediction distribution
                for i, val in enumerate(unique_values):
                    group_data = data[data[attr] == val]
                    pred_dist = group_data[prediction_column].value_counts()
                    axes[1, 0].bar([f"{val}_0", f"{val}_1"], [pred_dist.get(0, 0), pred_dist.get(1, 0)], alpha=0.8)
                
                axes[1, 0].set_xlabel('Group_Prediction')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_title(f'Prediction Distribution by {attr}')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Confusion matrix comparison
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                
                # Save to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                visualizations[f'fairness_{attr}'] = image_base64
                plt.close()
                
        except Exception as e:
            logger.warning(f"Error creating fairness visualizations: {e}")
        
        return visualizations
    
    # ===== COUNTERFACTUAL EXPLANATIONS =====
    
    async def generate_counterfactual_explanations(
        self,
        model_id: str,
        model_type: ModelType,
        input_data: Any,
        feature_names: Optional[List[str]] = None,
        num_counterfactuals: int = 5
    ) -> str:
        """Generate counterfactual explanations"""
        
        request_id = f"counterfactual_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Generating counterfactual explanations: {request_id}")
        
        try:
            # Load model
            model = await self._load_model(model_id, model_type)
            
            # Generate counterfactuals
            counterfactuals = await self._generate_counterfactuals(
                model, input_data, feature_names, num_counterfactuals
            )
            
            # Create counterfactual explanation
            cf_explanation = CounterfactualExplanation(
                original_prediction=model.predict([input_data])[0],
                counterfactual_instances=counterfactuals["instances"],
                feature_changes=counterfactuals["changes"],
                confidence_changes=counterfactuals["confidence_changes"],
                feasibility_scores=counterfactuals["feasibility_scores"]
            )
            
            # Store result
            self.completed_explanations[request_id] = cf_explanation
            
            # Update metrics
            self.xai_metrics["counterfactual_explanations"] += 1
            
            logger.info(f"Counterfactual explanations completed: {request_id}")
            
            return request_id
            
        except Exception as e:
            logger.error(f"Error generating counterfactual explanations: {e}", exc_info=True)
            raise
    
    async def _generate_counterfactuals(
        self,
        model,
        input_data: Any,
        feature_names: Optional[List[str]],
        num_counterfactuals: int
    ) -> Dict[str, Any]:
        """Generate counterfactual instances"""
        
        try:
            # Simple counterfactual generation using random perturbations
            # In practice, you'd use more sophisticated methods like DiCE, FACE, etc.
            
            original_prediction = model.predict([input_data])[0]
            counterfactuals = {
                "instances": [],
                "changes": [],
                "confidence_changes": [],
                "feasibility_scores": []
            }
            
            for i in range(num_counterfactuals):
                # Create a perturbed version
                perturbed_data = input_data.copy()
                
                # Randomly select features to perturb
                num_features_to_change = np.random.randint(1, min(3, len(input_data)))
                features_to_change = np.random.choice(len(input_data), num_features_to_change, replace=False)
                
                changes = {}
                for feat_idx in features_to_change:
                    # Simple perturbation (add random noise)
                    original_value = input_data[feat_idx]
                    perturbation = np.random.normal(0, 0.1 * abs(original_value) if original_value != 0 else 0.1)
                    perturbed_data[feat_idx] = original_value + perturbation
                    
                    feature_name = feature_names[feat_idx] if feature_names else f"Feature_{feat_idx}"
                    changes[feature_name] = {
                        "original": original_value,
                        "counterfactual": perturbed_data[feat_idx],
                        "change": perturbation
                    }
                
                # Get prediction for perturbed data
                cf_prediction = model.predict([perturbed_data])[0]
                
                # Calculate confidence change (simplified)
                confidence_change = abs(cf_prediction - original_prediction)
                
                # Calculate feasibility score (simplified)
                feasibility_score = 1.0 / (1.0 + np.sum(np.abs(perturbed_data - input_data)))
                
                counterfactuals["instances"].append(perturbed_data.tolist())
                counterfactuals["changes"].append(changes)
                counterfactuals["confidence_changes"].append(confidence_change)
                counterfactuals["feasibility_scores"].append(feasibility_score)
            
            return counterfactuals
            
        except Exception as e:
            logger.warning(f"Error generating counterfactuals: {e}")
            return {"instances": [], "changes": [], "confidence_changes": [], "feasibility_scores": []}
    
    # ===== UTILITY METHODS =====
    
    async def _load_model(self, model_id: str, model_type: ModelType):
        """Load model from registry"""
        
        try:
            if model_id in self.model_registry:
                return self.model_registry[model_id]
            
            # Try to load from model registry service
            if self.model_registry_service:
                model_info = await self.model_registry_service.get_model_info(model_id)
                if model_info:
                    # Load model based on type
                    if model_type == ModelType.SKLEARN:
                        model = joblib.load(model_info["model_path"])
                    elif model_type == ModelType.PYTORCH:
                        model = torch.load(model_info["model_path"])
                    else:
                        raise ValueError(f"Unsupported model type: {model_type}")
                    
                    # Cache model
                    self.model_registry[model_id] = model
                    return model
            
            raise ValueError(f"Model {model_id} not found")
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    async def _initialize_service_integrations(self):
        """Initialize service integrations"""
        
        try:
            # Initialize service connections
            self.automl_service = AutoMLService()
            self.rl_service = ReinforcementLearningService()
            self.multimodal_service = MultiModalAIService()
            self.model_registry_service = ModelRegistryService()
            
            # Initialize services
            await self.automl_service.initialize()
            await self.rl_service.initialize()
            await self.multimodal_service.initialize()
            await self.model_registry_service.initialize()
            
            logger.info("Service integrations initialized successfully")
            
        except Exception as e:
            logger.warning(f"Error initializing service integrations: {e}")
    
    async def _load_explainer_cache(self):
        """Load explainer cache from storage"""
        
        try:
            cache_file = self.storage_path / "explainer_cache.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self.explainer_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.explainer_cache)} explainers from cache")
        except Exception as e:
            logger.warning(f"Error loading explainer cache: {e}")
    
    async def _save_explainer_cache(self):
        """Save explainer cache to storage"""
        
        try:
            cache_file = self.storage_path / "explainer_cache.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self.explainer_cache, f)
            logger.info(f"Saved {len(self.explainer_cache)} explainers to cache")
        except Exception as e:
            logger.warning(f"Error saving explainer cache: {e}")
    
    async def _subscribe_to_model_events(self):
        """Subscribe to model events"""
        
        try:
            # Subscribe to model deployment events
            await shared_pulsar_client.subscribe(
                "q.model.deployed",
                self._handle_model_deployment,
                subscription_name="xai_model_deployment"
            )
            
            # Subscribe to model update events
            await shared_pulsar_client.subscribe(
                "q.model.updated",
                self._handle_model_update,
                subscription_name="xai_model_update"
            )
            
        except Exception as e:
            logger.warning(f"Error subscribing to model events: {e}")
    
    async def _handle_model_deployment(self, event_data: Dict[str, Any]):
        """Handle model deployment events"""
        
        try:
            model_id = event_data.get("model_id")
            if model_id:
                # Clear cached explainers for this model
                keys_to_remove = [key for key in self.explainer_cache.keys() if key.startswith(model_id)]
                for key in keys_to_remove:
                    del self.explainer_cache[key]
                
                logger.info(f"Cleared explainer cache for model: {model_id}")
                
        except Exception as e:
            logger.warning(f"Error handling model deployment: {e}")
    
    async def _handle_model_update(self, event_data: Dict[str, Any]):
        """Handle model update events"""
        
        try:
            model_id = event_data.get("model_id")
            if model_id:
                # Clear cached explainers for this model
                keys_to_remove = [key for key in self.explainer_cache.keys() if key.startswith(model_id)]
                for key in keys_to_remove:
                    del self.explainer_cache[key]
                
                # Remove from model registry
                if model_id in self.model_registry:
                    del self.model_registry[model_id]
                
                logger.info(f"Cleared cache for updated model: {model_id}")
                
        except Exception as e:
            logger.warning(f"Error handling model update: {e}")
    
    async def _start_background_tasks(self):
        """Start background tasks"""
        
        try:
            # Start explanation quality monitoring
            task = asyncio.create_task(self._monitor_explanation_quality())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            # Start cache cleanup
            task = asyncio.create_task(self._cleanup_cache())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
        except Exception as e:
            logger.warning(f"Error starting background tasks: {e}")
    
    async def _monitor_explanation_quality(self):
        """Monitor explanation quality periodically"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Analyze explanation quality
                if self.xai_metrics["explanation_quality_scores"]:
                    avg_quality = np.mean(self.xai_metrics["explanation_quality_scores"])
                    
                    if avg_quality < self.config["explanation_quality_threshold"]:
                        logger.warning(f"Low explanation quality detected: {avg_quality:.3f}")
                        
                        # Publish quality alert
                        await self._publish_explanation_event("quality_alert", "low_quality")
                
            except Exception as e:
                logger.error(f"Error monitoring explanation quality: {e}")
    
    async def _cleanup_cache(self):
        """Cleanup old cache entries periodically"""
        
        while True:
            try:
                await asyncio.sleep(86400)  # Clean every day
                
                # Remove old completed explanations (older than 7 days)
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                
                keys_to_remove = []
                for key, result in self.completed_explanations.items():
                    if hasattr(result, 'created_at') and result.created_at < cutoff_time:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.completed_explanations[key]
                
                logger.info(f"Cleaned up {len(keys_to_remove)} old explanation results")
                
            except Exception as e:
                logger.error(f"Error cleaning up cache: {e}")
    
    async def _publish_explanation_event(self, request_id: str, event_type: str):
        """Publish explanation events"""
        
        try:
            event_data = {
                "request_id": request_id,
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "service": "explainable_ai"
            }
            
            await shared_pulsar_client.publish("q.xai.events", event_data)
            
        except Exception as e:
            logger.warning(f"Error publishing explanation event: {e}")
    
    # ===== API METHODS =====
    
    async def get_explanation_status(self, request_id: str) -> Dict[str, Any]:
        """Get explanation status"""
        
        if request_id in self.completed_explanations:
            return {
                "status": "completed",
                "request_id": request_id,
                "result": self.completed_explanations[request_id]
            }
        elif request_id in self.active_requests:
            return {
                "status": "processing",
                "request_id": request_id,
                "request": self.active_requests[request_id]
            }
        else:
            return {
                "status": "not_found",
                "request_id": request_id
            }
    
    async def get_explanation_result(self, request_id: str) -> Optional[ExplanationResult]:
        """Get explanation result"""
        
        return self.completed_explanations.get(request_id)
    
    async def list_explanations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent explanations"""
        
        explanations = []
        
        for request_id, result in list(self.completed_explanations.items())[-limit:]:
            explanations.append({
                "request_id": request_id,
                "explanation_type": result.explanation_type.value,
                "explanation_method": result.explanation_method.value,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "created_at": result.created_at.isoformat()
            })
        
        return explanations
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get XAI service metrics"""
        
        return {
            "metrics": self.xai_metrics,
            "active_requests": len(self.active_requests),
            "completed_explanations": len(self.completed_explanations),
            "cached_explainers": len(self.explainer_cache),
            "cached_models": len(self.model_registry)
        }

# Create global instance
explainable_ai_service = ExplainableAIService() 