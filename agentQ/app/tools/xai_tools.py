"""
XAI Tools for Agent Integration

This module provides explainability tools that agents can use to understand
and explain model behavior:
- SHAP and LIME explanations
- Fairness analysis
- Counterfactual explanations
- Model interpretability
"""

import logging
import json
import asyncio
import base64
from typing import Dict, Any, Optional, List
from agentQ.app.core.toolbox import Tool

# Import XAI service
try:
    from managerQ.app.core.explainable_ai_service import (
        explainable_ai_service,
        ExplanationType,
        ExplanationMethod,
        ModelType,
        DataType
    )
except ImportError:
    # Fallback if managerQ is not available
    explainable_ai_service = None
    logging.warning("XAI service not available - explainability features will be disabled")

logger = logging.getLogger(__name__)

# ===== EXPLANATION GENERATION TOOLS =====

def generate_shap_explanation_func(
    model_id: str,
    model_type: str,
    input_data: str,  # JSON string
    feature_names: Optional[str] = None,  # JSON string
    class_names: Optional[str] = None,  # JSON string
    background_data: Optional[str] = None,  # JSON string
    config: Dict[str, Any] = None
) -> str:
    """Generate SHAP explanation for a model prediction"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Parse JSON strings
        input_data_parsed = json.loads(input_data)
        feature_names_parsed = json.loads(feature_names) if feature_names else None
        class_names_parsed = json.loads(class_names) if class_names else None
        background_data_parsed = json.loads(background_data) if background_data else None
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            request_id = loop.run_until_complete(
                explainable_ai_service.generate_explanation(
                    model_id=model_id,
                    model_type=ModelType(model_type),
                    data_type=DataType.TABULAR,
                    explanation_type=ExplanationType.LOCAL,
                    explanation_method=ExplanationMethod.SHAP,
                    input_data=input_data_parsed,
                    feature_names=feature_names_parsed,
                    class_names=class_names_parsed,
                    background_data=background_data_parsed
                )
            )
            return f"SHAP explanation request started: {request_id}"
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to generate SHAP explanation: {e}")
        return f"Error: {str(e)}"

def generate_lime_explanation_func(
    model_id: str,
    model_type: str,
    input_data: str,  # JSON string
    feature_names: Optional[str] = None,  # JSON string
    class_names: Optional[str] = None,  # JSON string
    background_data: Optional[str] = None,  # JSON string
    config: Dict[str, Any] = None
) -> str:
    """Generate LIME explanation for a model prediction"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Parse JSON strings
        input_data_parsed = json.loads(input_data)
        feature_names_parsed = json.loads(feature_names) if feature_names else None
        class_names_parsed = json.loads(class_names) if class_names else None
        background_data_parsed = json.loads(background_data) if background_data else None
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            request_id = loop.run_until_complete(
                explainable_ai_service.generate_explanation(
                    model_id=model_id,
                    model_type=ModelType(model_type),
                    data_type=DataType.TABULAR,
                    explanation_type=ExplanationType.LOCAL,
                    explanation_method=ExplanationMethod.LIME,
                    input_data=input_data_parsed,
                    feature_names=feature_names_parsed,
                    class_names=class_names_parsed,
                    background_data=background_data_parsed
                )
            )
            return f"LIME explanation request started: {request_id}"
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to generate LIME explanation: {e}")
        return f"Error: {str(e)}"

def generate_integrated_gradients_explanation_func(
    model_id: str,
    model_type: str,
    input_data: str,  # JSON string
    feature_names: Optional[str] = None,  # JSON string
    class_names: Optional[str] = None,  # JSON string
    config: Dict[str, Any] = None
) -> str:
    """Generate Integrated Gradients explanation for a PyTorch model"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Parse JSON strings
        input_data_parsed = json.loads(input_data)
        feature_names_parsed = json.loads(feature_names) if feature_names else None
        class_names_parsed = json.loads(class_names) if class_names else None
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            request_id = loop.run_until_complete(
                explainable_ai_service.generate_explanation(
                    model_id=model_id,
                    model_type=ModelType(model_type),
                    data_type=DataType.TABULAR,
                    explanation_type=ExplanationType.LOCAL,
                    explanation_method=ExplanationMethod.INTEGRATED_GRADIENTS,
                    input_data=input_data_parsed,
                    feature_names=feature_names_parsed,
                    class_names=class_names_parsed
                )
            )
            return f"Integrated Gradients explanation request started: {request_id}"
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to generate Integrated Gradients explanation: {e}")
        return f"Error: {str(e)}"

def generate_permutation_importance_explanation_func(
    model_id: str,
    model_type: str,
    input_data: str,  # JSON string
    target_data: str,  # JSON string
    feature_names: Optional[str] = None,  # JSON string
    config: Dict[str, Any] = None
) -> str:
    """Generate Permutation Importance explanation for global model behavior"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Parse JSON strings
        input_data_parsed = json.loads(input_data)
        target_data_parsed = json.loads(target_data)
        feature_names_parsed = json.loads(feature_names) if feature_names else None
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            request_id = loop.run_until_complete(
                explainable_ai_service.generate_explanation(
                    model_id=model_id,
                    model_type=ModelType(model_type),
                    data_type=DataType.TABULAR,
                    explanation_type=ExplanationType.GLOBAL,
                    explanation_method=ExplanationMethod.PERMUTATION_IMPORTANCE,
                    input_data=input_data_parsed,
                    background_data=target_data_parsed,
                    feature_names=feature_names_parsed
                )
            )
            return f"Permutation Importance explanation request started: {request_id}"
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to generate Permutation Importance explanation: {e}")
        return f"Error: {str(e)}"

# ===== FAIRNESS ANALYSIS TOOLS =====

def analyze_model_fairness_func(
    model_id: str,
    model_type: str,
    test_data: str,  # JSON string (DataFrame as dict)
    protected_attributes: str,  # JSON string (list of column names)
    target_column: str,
    prediction_column: str,
    config: Dict[str, Any] = None
) -> str:
    """Analyze model fairness across protected attributes"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        import pandas as pd
        
        # Parse JSON strings
        test_data_dict = json.loads(test_data)
        protected_attributes_list = json.loads(protected_attributes)
        
        # Convert to DataFrame
        test_df = pd.DataFrame(test_data_dict)
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            request_id = loop.run_until_complete(
                explainable_ai_service.generate_fairness_analysis(
                    model_id=model_id,
                    model_type=ModelType(model_type),
                    test_data=test_df,
                    protected_attributes=protected_attributes_list,
                    target_column=target_column,
                    prediction_column=prediction_column
                )
            )
            return f"Fairness analysis request started: {request_id}"
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to analyze model fairness: {e}")
        return f"Error: {str(e)}"

def detect_bias_func(
    model_id: str,
    test_data: str,  # JSON string (DataFrame as dict)
    protected_attributes: str,  # JSON string (list of column names)
    target_column: str,
    prediction_column: str,
    config: Dict[str, Any] = None
) -> str:
    """Detect bias in model predictions"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        import pandas as pd
        
        # Parse JSON strings
        test_data_dict = json.loads(test_data)
        protected_attributes_list = json.loads(protected_attributes)
        
        # Convert to DataFrame
        test_df = pd.DataFrame(test_data_dict)
        
        # Simple bias detection
        bias_results = {}
        
        for attr in protected_attributes_list:
            unique_values = test_df[attr].unique()
            
            if len(unique_values) == 2:
                val1, val2 = unique_values
                
                group1 = test_df[test_df[attr] == val1]
                group2 = test_df[test_df[attr] == val2]
                
                # Calculate positive prediction rates
                pos_rate1 = len(group1[group1[prediction_column] == 1]) / len(group1) if len(group1) > 0 else 0
                pos_rate2 = len(group2[group2[prediction_column] == 1]) / len(group2) if len(group2) > 0 else 0
                
                # Demographic parity difference
                demographic_parity = abs(pos_rate1 - pos_rate2)
                
                bias_results[attr] = {
                    "demographic_parity": demographic_parity,
                    "bias_detected": demographic_parity > 0.1,
                    "group1_positive_rate": pos_rate1,
                    "group2_positive_rate": pos_rate2
                }
        
        return json.dumps(bias_results, indent=2)
    
    except Exception as e:
        logger.error(f"Failed to detect bias: {e}")
        return f"Error: {str(e)}"

# ===== COUNTERFACTUAL EXPLANATION TOOLS =====

def generate_counterfactual_explanations_func(
    model_id: str,
    model_type: str,
    input_data: str,  # JSON string
    feature_names: Optional[str] = None,  # JSON string
    num_counterfactuals: int = 5,
    config: Dict[str, Any] = None
) -> str:
    """Generate counterfactual explanations for a prediction"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Parse JSON strings
        input_data_parsed = json.loads(input_data)
        feature_names_parsed = json.loads(feature_names) if feature_names else None
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            request_id = loop.run_until_complete(
                explainable_ai_service.generate_counterfactual_explanations(
                    model_id=model_id,
                    model_type=ModelType(model_type),
                    input_data=input_data_parsed,
                    feature_names=feature_names_parsed,
                    num_counterfactuals=num_counterfactuals
                )
            )
            return f"Counterfactual explanations request started: {request_id}"
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to generate counterfactual explanations: {e}")
        return f"Error: {str(e)}"

def explain_what_if_scenario_func(
    model_id: str,
    model_type: str,
    input_data: str,  # JSON string
    feature_changes: str,  # JSON string (dict of feature_name: new_value)
    feature_names: Optional[str] = None,  # JSON string
    config: Dict[str, Any] = None
) -> str:
    """Explain what-if scenarios by changing specific features"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Parse JSON strings
        input_data_parsed = json.loads(input_data)
        feature_changes_parsed = json.loads(feature_changes)
        feature_names_parsed = json.loads(feature_names) if feature_names else None
        
        # Apply feature changes
        modified_data = input_data_parsed.copy()
        
        if feature_names_parsed:
            for feature_name, new_value in feature_changes_parsed.items():
                if feature_name in feature_names_parsed:
                    feature_idx = feature_names_parsed.index(feature_name)
                    modified_data[feature_idx] = new_value
        
        # Generate explanation for modified data
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            request_id = loop.run_until_complete(
                explainable_ai_service.generate_explanation(
                    model_id=model_id,
                    model_type=ModelType(model_type),
                    data_type=DataType.TABULAR,
                    explanation_type=ExplanationType.LOCAL,
                    explanation_method=ExplanationMethod.SHAP,
                    input_data=modified_data,
                    feature_names=feature_names_parsed
                )
            )
            
            return f"What-if scenario explanation request started: {request_id}. Changed features: {feature_changes_parsed}"
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to explain what-if scenario: {e}")
        return f"Error: {str(e)}"

# ===== EXPLANATION RETRIEVAL TOOLS =====

def get_explanation_status_func(
    request_id: str,
    config: Dict[str, Any] = None
) -> str:
    """Get the status of an explanation request"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            status = loop.run_until_complete(
                explainable_ai_service.get_explanation_status(request_id)
            )
            return json.dumps(status, indent=2, default=str)
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to get explanation status: {e}")
        return f"Error: {str(e)}"

def get_explanation_result_func(
    request_id: str,
    config: Dict[str, Any] = None
) -> str:
    """Get the result of a completed explanation request"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                explainable_ai_service.get_explanation_result(request_id)
            )
            
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
                    "created_at": result.created_at.isoformat(),
                    "visualizations": {k: f"base64_image_{len(v)}_bytes" for k, v in result.visualizations.items()}
                }
                return json.dumps(result_dict, indent=2)
            else:
                return f"No result found for request ID: {request_id}"
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to get explanation result: {e}")
        return f"Error: {str(e)}"

def get_explanation_visualization_func(
    request_id: str,
    visualization_type: str,
    config: Dict[str, Any] = None
) -> str:
    """Get a specific visualization from an explanation result"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                explainable_ai_service.get_explanation_result(request_id)
            )
            
            if result and visualization_type in result.visualizations:
                return result.visualizations[visualization_type]
            else:
                return f"Visualization '{visualization_type}' not found for request ID: {request_id}"
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to get explanation visualization: {e}")
        return f"Error: {str(e)}"

# ===== MODEL INTERPRETABILITY TOOLS =====

def analyze_model_interpretability_func(
    model_id: str,
    model_type: str,
    test_data: str,  # JSON string
    feature_names: Optional[str] = None,  # JSON string
    config: Dict[str, Any] = None
) -> str:
    """Analyze overall model interpretability"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Parse JSON strings
        test_data_parsed = json.loads(test_data)
        feature_names_parsed = json.loads(feature_names) if feature_names else None
        
        # Generate global explanation using permutation importance
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            request_id = loop.run_until_complete(
                explainable_ai_service.generate_explanation(
                    model_id=model_id,
                    model_type=ModelType(model_type),
                    data_type=DataType.TABULAR,
                    explanation_type=ExplanationType.GLOBAL,
                    explanation_method=ExplanationMethod.PERMUTATION_IMPORTANCE,
                    input_data=test_data_parsed,
                    feature_names=feature_names_parsed
                )
            )
            return f"Model interpretability analysis request started: {request_id}"
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to analyze model interpretability: {e}")
        return f"Error: {str(e)}"

def compare_model_explanations_func(
    model_id1: str,
    model_id2: str,
    model_type: str,
    input_data: str,  # JSON string
    feature_names: Optional[str] = None,  # JSON string
    config: Dict[str, Any] = None
) -> str:
    """Compare explanations between two models"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Parse JSON strings
        input_data_parsed = json.loads(input_data)
        feature_names_parsed = json.loads(feature_names) if feature_names else None
        
        # Generate explanations for both models
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            request_id1 = loop.run_until_complete(
                explainable_ai_service.generate_explanation(
                    model_id=model_id1,
                    model_type=ModelType(model_type),
                    data_type=DataType.TABULAR,
                    explanation_type=ExplanationType.LOCAL,
                    explanation_method=ExplanationMethod.SHAP,
                    input_data=input_data_parsed,
                    feature_names=feature_names_parsed
                )
            )
            
            request_id2 = loop.run_until_complete(
                explainable_ai_service.generate_explanation(
                    model_id=model_id2,
                    model_type=ModelType(model_type),
                    data_type=DataType.TABULAR,
                    explanation_type=ExplanationType.LOCAL,
                    explanation_method=ExplanationMethod.SHAP,
                    input_data=input_data_parsed,
                    feature_names=feature_names_parsed
                )
            )
            
            return f"Model comparison explanations started: Model 1: {request_id1}, Model 2: {request_id2}"
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to compare model explanations: {e}")
        return f"Error: {str(e)}"

# ===== UTILITY TOOLS =====

def list_recent_explanations_func(
    limit: int = 10,
    config: Dict[str, Any] = None
) -> str:
    """List recent explanations"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            explanations = loop.run_until_complete(
                explainable_ai_service.list_explanations(limit)
            )
            return json.dumps(explanations, indent=2)
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to list recent explanations: {e}")
        return f"Error: {str(e)}"

def get_xai_service_metrics_func(
    config: Dict[str, Any] = None
) -> str:
    """Get XAI service metrics and statistics"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            metrics = loop.run_until_complete(
                explainable_ai_service.get_service_metrics()
            )
            return json.dumps(metrics, indent=2)
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to get XAI service metrics: {e}")
        return f"Error: {str(e)}"

def validate_explanation_quality_func(
    request_id: str,
    quality_threshold: float = 0.7,
    config: Dict[str, Any] = None
) -> str:
    """Validate the quality of an explanation"""
    
    if explainable_ai_service is None:
        return "Error: XAI service not available"
    
    try:
        # Get explanation result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                explainable_ai_service.get_explanation_result(request_id)
            )
            
            if result:
                quality_score = result.confidence
                
                validation_result = {
                    "request_id": request_id,
                    "quality_score": quality_score,
                    "quality_threshold": quality_threshold,
                    "passes_quality_check": quality_score >= quality_threshold,
                    "explanation_method": result.explanation_method.value,
                    "processing_time": result.processing_time,
                    "metrics": result.metrics
                }
                
                return json.dumps(validation_result, indent=2)
            else:
                return f"No result found for request ID: {request_id}"
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Failed to validate explanation quality: {e}")
        return f"Error: {str(e)}"

# ===== TOOL DEFINITIONS =====

# SHAP Tools
generate_shap_explanation_tool = Tool(
    name="generate_shap_explanation",
    description="Generate SHAP explanation for a model prediction. Parameters: model_id (str), model_type (str: sklearn/pytorch/tensorflow), input_data (JSON str), feature_names (JSON str, optional), class_names (JSON str, optional), background_data (JSON str, optional)",
    func=generate_shap_explanation_func
)

# LIME Tools
generate_lime_explanation_tool = Tool(
    name="generate_lime_explanation",
    description="Generate LIME explanation for a model prediction. Parameters: model_id (str), model_type (str: sklearn/pytorch/tensorflow), input_data (JSON str), feature_names (JSON str, optional), class_names (JSON str, optional), background_data (JSON str, optional)",
    func=generate_lime_explanation_func
)

# Captum Tools
generate_integrated_gradients_explanation_tool = Tool(
    name="generate_integrated_gradients_explanation",
    description="Generate Integrated Gradients explanation for a PyTorch model. Parameters: model_id (str), model_type (str: pytorch), input_data (JSON str), feature_names (JSON str, optional), class_names (JSON str, optional)",
    func=generate_integrated_gradients_explanation_func
)

# Global Explanations
generate_permutation_importance_explanation_tool = Tool(
    name="generate_permutation_importance_explanation",
    description="Generate Permutation Importance explanation for global model behavior. Parameters: model_id (str), model_type (str), input_data (JSON str), target_data (JSON str), feature_names (JSON str, optional)",
    func=generate_permutation_importance_explanation_func
)

# Fairness Tools
analyze_model_fairness_tool = Tool(
    name="analyze_model_fairness",
    description="Analyze model fairness across protected attributes. Parameters: model_id (str), model_type (str), test_data (JSON str), protected_attributes (JSON str), target_column (str), prediction_column (str)",
    func=analyze_model_fairness_func
)

detect_bias_tool = Tool(
    name="detect_bias",
    description="Detect bias in model predictions. Parameters: model_id (str), test_data (JSON str), protected_attributes (JSON str), target_column (str), prediction_column (str)",
    func=detect_bias_func
)

# Counterfactual Tools
generate_counterfactual_explanations_tool = Tool(
    name="generate_counterfactual_explanations",
    description="Generate counterfactual explanations for a prediction. Parameters: model_id (str), model_type (str), input_data (JSON str), feature_names (JSON str, optional), num_counterfactuals (int, optional)",
    func=generate_counterfactual_explanations_func
)

explain_what_if_scenario_tool = Tool(
    name="explain_what_if_scenario",
    description="Explain what-if scenarios by changing specific features. Parameters: model_id (str), model_type (str), input_data (JSON str), feature_changes (JSON str), feature_names (JSON str, optional)",
    func=explain_what_if_scenario_func
)

# Result Retrieval Tools
get_explanation_status_tool = Tool(
    name="get_explanation_status",
    description="Get the status of an explanation request. Parameters: request_id (str)",
    func=get_explanation_status_func
)

get_explanation_result_tool = Tool(
    name="get_explanation_result",
    description="Get the result of a completed explanation request. Parameters: request_id (str)",
    func=get_explanation_result_func
)

get_explanation_visualization_tool = Tool(
    name="get_explanation_visualization",
    description="Get a specific visualization from an explanation result. Parameters: request_id (str), visualization_type (str)",
    func=get_explanation_visualization_func
)

# Interpretability Tools
analyze_model_interpretability_tool = Tool(
    name="analyze_model_interpretability",
    description="Analyze overall model interpretability. Parameters: model_id (str), model_type (str), test_data (JSON str), feature_names (JSON str, optional)",
    func=analyze_model_interpretability_func
)

compare_model_explanations_tool = Tool(
    name="compare_model_explanations",
    description="Compare explanations between two models. Parameters: model_id1 (str), model_id2 (str), model_type (str), input_data (JSON str), feature_names (JSON str, optional)",
    func=compare_model_explanations_func
)

# Utility Tools
list_recent_explanations_tool = Tool(
    name="list_recent_explanations",
    description="List recent explanations. Parameters: limit (int, optional)",
    func=list_recent_explanations_func
)

get_xai_service_metrics_tool = Tool(
    name="get_xai_service_metrics",
    description="Get XAI service metrics and statistics. No parameters required.",
    func=get_xai_service_metrics_func
)

validate_explanation_quality_tool = Tool(
    name="validate_explanation_quality",
    description="Validate the quality of an explanation. Parameters: request_id (str), quality_threshold (float, optional)",
    func=validate_explanation_quality_func
)

# Collection of all XAI tools
xai_tools = [
    generate_shap_explanation_tool,
    generate_lime_explanation_tool,
    generate_integrated_gradients_explanation_tool,
    generate_permutation_importance_explanation_tool,
    analyze_model_fairness_tool,
    detect_bias_tool,
    generate_counterfactual_explanations_tool,
    explain_what_if_scenario_tool,
    get_explanation_status_tool,
    get_explanation_result_tool,
    get_explanation_visualization_tool,
    analyze_model_interpretability_tool,
    compare_model_explanations_tool,
    list_recent_explanations_tool,
    get_xai_service_metrics_tool,
    validate_explanation_quality_tool
]

# Convenience collections
explanation_generation_tools = [
    generate_shap_explanation_tool,
    generate_lime_explanation_tool,
    generate_integrated_gradients_explanation_tool,
    generate_permutation_importance_explanation_tool
]

fairness_tools = [
    analyze_model_fairness_tool,
    detect_bias_tool
]

counterfactual_tools = [
    generate_counterfactual_explanations_tool,
    explain_what_if_scenario_tool
]

interpretability_tools = [
    analyze_model_interpretability_tool,
    compare_model_explanations_tool
]

utility_tools = [
    get_explanation_status_tool,
    get_explanation_result_tool,
    get_explanation_visualization_tool,
    list_recent_explanations_tool,
    get_xai_service_metrics_tool,
    validate_explanation_quality_tool
] 