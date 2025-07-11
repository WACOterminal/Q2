"""
Explainable AI API

This API provides endpoints for generating and retrieving model explanations:
- SHAP and LIME explanations
- Fairness analysis
- Counterfactual explanations
- Model interpretability analysis
- Explanation quality validation
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
import logging
import json
import base64
from io import BytesIO
from fastapi.responses import StreamingResponse

# Q Platform imports
from shared.auth.dependencies import get_current_user, UserClaims
from managerQ.app.core.explainable_ai_service import (
    explainable_ai_service,
    ExplanationType,
    ExplanationMethod,
    ModelType,
    DataType,
    ExplanationRequest,
    ExplanationResult
)

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

# ===== REQUEST MODELS =====

class ExplanationGenerationRequest(BaseModel):
    """Request for generating explanations"""
    model_id: str = Field(..., description="ID of the model to explain")
    model_type: ModelType = Field(..., description="Type of the model")
    data_type: DataType = Field(DataType.TABULAR, description="Type of input data")
    explanation_type: ExplanationType = Field(..., description="Type of explanation")
    explanation_method: ExplanationMethod = Field(..., description="Explanation method")
    input_data: List[float] = Field(..., description="Input data for explanation")
    feature_names: Optional[List[str]] = Field(None, description="Names of features")
    class_names: Optional[List[str]] = Field(None, description="Names of classes")
    background_data: Optional[List[List[float]]] = Field(None, description="Background data")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

class FairnessAnalysisRequest(BaseModel):
    """Request for fairness analysis"""
    model_id: str = Field(..., description="ID of the model to analyze")
    model_type: ModelType = Field(..., description="Type of the model")
    test_data: Dict[str, List[Any]] = Field(..., description="Test data as dictionary")
    protected_attributes: List[str] = Field(..., description="Protected attribute columns")
    target_column: str = Field(..., description="Target column name")
    prediction_column: str = Field(..., description="Prediction column name")

class CounterfactualRequest(BaseModel):
    """Request for counterfactual explanations"""
    model_id: str = Field(..., description="ID of the model")
    model_type: ModelType = Field(..., description="Type of the model")
    input_data: List[float] = Field(..., description="Input data")
    feature_names: Optional[List[str]] = Field(None, description="Names of features")
    num_counterfactuals: int = Field(5, description="Number of counterfactuals to generate")

class WhatIfScenarioRequest(BaseModel):
    """Request for what-if scenario analysis"""
    model_id: str = Field(..., description="ID of the model")
    model_type: ModelType = Field(..., description="Type of the model")
    input_data: List[float] = Field(..., description="Input data")
    feature_changes: Dict[str, float] = Field(..., description="Feature changes to apply")
    feature_names: Optional[List[str]] = Field(None, description="Names of features")

class ModelComparisonRequest(BaseModel):
    """Request for model comparison"""
    model_id1: str = Field(..., description="ID of first model")
    model_id2: str = Field(..., description="ID of second model")
    model_type: ModelType = Field(..., description="Type of the models")
    input_data: List[float] = Field(..., description="Input data")
    feature_names: Optional[List[str]] = Field(None, description="Names of features")
    explanation_method: ExplanationMethod = Field(ExplanationMethod.SHAP, description="Explanation method")

# ===== RESPONSE MODELS =====

class ExplanationStatusResponse(BaseModel):
    """Response for explanation status"""
    status: str = Field(..., description="Status of the explanation request")
    request_id: str = Field(..., description="Request ID")
    message: Optional[str] = Field(None, description="Status message")
    progress: Optional[float] = Field(None, description="Progress percentage")

class ExplanationResultResponse(BaseModel):
    """Response for explanation result"""
    request_id: str = Field(..., description="Request ID")
    explanation_type: str = Field(..., description="Type of explanation")
    explanation_method: str = Field(..., description="Explanation method")
    explanations: Dict[str, Any] = Field(..., description="Explanation data")
    metrics: Dict[str, float] = Field(..., description="Explanation metrics")
    interpretation: str = Field(..., description="Human-readable interpretation")
    confidence: float = Field(..., description="Confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    created_at: str = Field(..., description="Creation timestamp")
    has_visualizations: bool = Field(..., description="Whether visualizations are available")

class ServiceMetricsResponse(BaseModel):
    """Response for service metrics"""
    explanations_generated: int = Field(..., description="Total explanations generated")
    local_explanations: int = Field(..., description="Local explanations generated")
    global_explanations: int = Field(..., description="Global explanations generated")
    fairness_analyses: int = Field(..., description="Fairness analyses performed")
    counterfactual_explanations: int = Field(..., description="Counterfactual explanations generated")
    average_processing_time: float = Field(..., description="Average processing time")
    active_requests: int = Field(..., description="Active requests")
    completed_explanations: int = Field(..., description="Completed explanations")
    cached_explainers: int = Field(..., description="Cached explainers")

# ===== EXPLANATION GENERATION ENDPOINTS =====

@router.post("/generate", response_model=Dict[str, str])
async def generate_explanation(
    request: ExplanationGenerationRequest,
    background_tasks: BackgroundTasks,
    user: UserClaims = Depends(get_current_user)
):
    """Generate explanation for a model prediction"""
    
    try:
        logger.info(f"Generating {request.explanation_method.value} explanation for model {request.model_id}")
        
        # Generate explanation
        request_id = await explainable_ai_service.generate_explanation(
            model_id=request.model_id,
            model_type=request.model_type,
            data_type=request.data_type,
            explanation_type=request.explanation_type,
            explanation_method=request.explanation_method,
            input_data=request.input_data,
            feature_names=request.feature_names,
            class_names=request.class_names,
            background_data=request.background_data,
            parameters=request.parameters
        )
        
        return {
            "request_id": request_id,
            "status": "started",
            "message": f"{request.explanation_method.value} explanation generation started"
        }
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/shap", response_model=Dict[str, str])
async def generate_shap_explanation(
    model_id: str,
    model_type: ModelType,
    input_data: List[float],
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    background_data: Optional[List[List[float]]] = None,
    user: UserClaims = Depends(get_current_user)
):
    """Generate SHAP explanation (convenience endpoint)"""
    
    try:
        request_id = await explainable_ai_service.generate_explanation(
            model_id=model_id,
            model_type=model_type,
            data_type=DataType.TABULAR,
            explanation_type=ExplanationType.LOCAL,
            explanation_method=ExplanationMethod.SHAP,
            input_data=input_data,
            feature_names=feature_names,
            class_names=class_names,
            background_data=background_data
        )
        
        return {
            "request_id": request_id,
            "status": "started",
            "message": "SHAP explanation generation started"
        }
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lime", response_model=Dict[str, str])
async def generate_lime_explanation(
    model_id: str,
    model_type: ModelType,
    input_data: List[float],
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    background_data: Optional[List[List[float]]] = None,
    user: UserClaims = Depends(get_current_user)
):
    """Generate LIME explanation (convenience endpoint)"""
    
    try:
        request_id = await explainable_ai_service.generate_explanation(
            model_id=model_id,
            model_type=model_type,
            data_type=DataType.TABULAR,
            explanation_type=ExplanationType.LOCAL,
            explanation_method=ExplanationMethod.LIME,
            input_data=input_data,
            feature_names=feature_names,
            class_names=class_names,
            background_data=background_data
        )
        
        return {
            "request_id": request_id,
            "status": "started",
            "message": "LIME explanation generation started"
        }
        
    except Exception as e:
        logger.error(f"Error generating LIME explanation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/global", response_model=Dict[str, str])
async def generate_global_explanation(
    model_id: str,
    model_type: ModelType,
    test_data: List[List[float]],
    target_data: List[float],
    feature_names: Optional[List[str]] = None,
    explanation_method: ExplanationMethod = ExplanationMethod.PERMUTATION_IMPORTANCE,
    user: UserClaims = Depends(get_current_user)
):
    """Generate global explanation for model behavior"""
    
    try:
        request_id = await explainable_ai_service.generate_explanation(
            model_id=model_id,
            model_type=model_type,
            data_type=DataType.TABULAR,
            explanation_type=ExplanationType.GLOBAL,
            explanation_method=explanation_method,
            input_data=test_data,
            background_data=target_data,
            feature_names=feature_names
        )
        
        return {
            "request_id": request_id,
            "status": "started",
            "message": f"Global explanation generation started using {explanation_method.value}"
        }
        
    except Exception as e:
        logger.error(f"Error generating global explanation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ===== FAIRNESS ANALYSIS ENDPOINTS =====

@router.post("/fairness", response_model=Dict[str, str])
async def analyze_fairness(
    request: FairnessAnalysisRequest,
    user: UserClaims = Depends(get_current_user)
):
    """Analyze model fairness across protected attributes"""
    
    try:
        import pandas as pd
        
        # Convert request data to DataFrame
        test_df = pd.DataFrame(request.test_data)
        
        request_id = await explainable_ai_service.generate_fairness_analysis(
            model_id=request.model_id,
            model_type=request.model_type,
            test_data=test_df,
            protected_attributes=request.protected_attributes,
            target_column=request.target_column,
            prediction_column=request.prediction_column
        )
        
        return {
            "request_id": request_id,
            "status": "started",
            "message": "Fairness analysis started"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing fairness: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bias-detection", response_model=Dict[str, Any])
async def detect_bias(
    test_data: Dict[str, List[Any]],
    protected_attributes: List[str],
    target_column: str,
    prediction_column: str,
    user: UserClaims = Depends(get_current_user)
):
    """Detect bias in model predictions"""
    
    try:
        import pandas as pd
        
        # Convert to DataFrame
        test_df = pd.DataFrame(test_data)
        
        bias_results = {}
        
        for attr in protected_attributes:
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
                
                # True positive rates
                tpr1 = len(group1[(group1[target_column] == 1) & (group1[prediction_column] == 1)]) / len(group1[group1[target_column] == 1]) if len(group1[group1[target_column] == 1]) > 0 else 0
                tpr2 = len(group2[(group2[target_column] == 1) & (group2[prediction_column] == 1)]) / len(group2[group2[target_column] == 1]) if len(group2[group2[target_column] == 1]) > 0 else 0
                
                # Equalized odds difference
                equalized_odds = abs(tpr1 - tpr2)
                
                bias_results[attr] = {
                    "demographic_parity": demographic_parity,
                    "equalized_odds": equalized_odds,
                    "bias_detected": demographic_parity > 0.1 or equalized_odds > 0.1,
                    "group1_positive_rate": pos_rate1,
                    "group2_positive_rate": pos_rate2,
                    "group1_tpr": tpr1,
                    "group2_tpr": tpr2
                }
        
        return {
            "bias_analysis": bias_results,
            "summary": {
                "total_attributes_analyzed": len(protected_attributes),
                "attributes_with_bias": sum(1 for result in bias_results.values() if result.get("bias_detected", False)),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error detecting bias: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ===== COUNTERFACTUAL EXPLANATIONS =====

@router.post("/counterfactual", response_model=Dict[str, str])
async def generate_counterfactual_explanations(
    request: CounterfactualRequest,
    user: UserClaims = Depends(get_current_user)
):
    """Generate counterfactual explanations"""
    
    try:
        request_id = await explainable_ai_service.generate_counterfactual_explanations(
            model_id=request.model_id,
            model_type=request.model_type,
            input_data=request.input_data,
            feature_names=request.feature_names,
            num_counterfactuals=request.num_counterfactuals
        )
        
        return {
            "request_id": request_id,
            "status": "started",
            "message": "Counterfactual explanations generation started"
        }
        
    except Exception as e:
        logger.error(f"Error generating counterfactual explanations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/what-if", response_model=Dict[str, str])
async def analyze_what_if_scenario(
    request: WhatIfScenarioRequest,
    user: UserClaims = Depends(get_current_user)
):
    """Analyze what-if scenarios"""
    
    try:
        # Apply feature changes
        modified_data = request.input_data.copy()
        
        if request.feature_names:
            for feature_name, new_value in request.feature_changes.items():
                if feature_name in request.feature_names:
                    feature_idx = request.feature_names.index(feature_name)
                    modified_data[feature_idx] = new_value
        
        # Generate explanation for modified data
        request_id = await explainable_ai_service.generate_explanation(
            model_id=request.model_id,
            model_type=request.model_type,
            data_type=DataType.TABULAR,
            explanation_type=ExplanationType.LOCAL,
            explanation_method=ExplanationMethod.SHAP,
            input_data=modified_data,
            feature_names=request.feature_names
        )
        
        return {
            "request_id": request_id,
            "status": "started",
            "message": "What-if scenario analysis started",
            "changes_applied": request.feature_changes
        }
        
    except Exception as e:
        logger.error(f"Error analyzing what-if scenario: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ===== MODEL COMPARISON =====

@router.post("/compare", response_model=Dict[str, Any])
async def compare_models(
    request: ModelComparisonRequest,
    user: UserClaims = Depends(get_current_user)
):
    """Compare explanations between two models"""
    
    try:
        # Generate explanations for both models
        request_id1 = await explainable_ai_service.generate_explanation(
            model_id=request.model_id1,
            model_type=request.model_type,
            data_type=DataType.TABULAR,
            explanation_type=ExplanationType.LOCAL,
            explanation_method=request.explanation_method,
            input_data=request.input_data,
            feature_names=request.feature_names
        )
        
        request_id2 = await explainable_ai_service.generate_explanation(
            model_id=request.model_id2,
            model_type=request.model_type,
            data_type=DataType.TABULAR,
            explanation_type=ExplanationType.LOCAL,
            explanation_method=request.explanation_method,
            input_data=request.input_data,
            feature_names=request.feature_names
        )
        
        return {
            "comparison_id": f"comp_{request_id1}_{request_id2}",
            "model1_explanation_id": request_id1,
            "model2_explanation_id": request_id2,
            "status": "started",
            "message": "Model comparison started"
        }
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ===== RESULT RETRIEVAL ENDPOINTS =====

@router.get("/status/{request_id}", response_model=ExplanationStatusResponse)
async def get_explanation_status(
    request_id: str,
    user: UserClaims = Depends(get_current_user)
):
    """Get explanation status"""
    
    try:
        status = await explainable_ai_service.get_explanation_status(request_id)
        
        return ExplanationStatusResponse(
            status=status["status"],
            request_id=request_id,
            message=status.get("message", ""),
            progress=status.get("progress", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error getting explanation status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/result/{request_id}", response_model=ExplanationResultResponse)
async def get_explanation_result(
    request_id: str,
    user: UserClaims = Depends(get_current_user)
):
    """Get explanation result"""
    
    try:
        result = await explainable_ai_service.get_explanation_result(request_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Explanation result not found")
        
        return ExplanationResultResponse(
            request_id=result.request_id,
            explanation_type=result.explanation_type.value,
            explanation_method=result.explanation_method.value,
            explanations=result.explanations,
            metrics=result.metrics,
            interpretation=result.interpretation,
            confidence=result.confidence,
            processing_time=result.processing_time,
            created_at=result.created_at.isoformat(),
            has_visualizations=len(result.visualizations) > 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting explanation result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/visualization/{request_id}/{visualization_type}")
async def get_explanation_visualization(
    request_id: str,
    visualization_type: str,
    user: UserClaims = Depends(get_current_user)
):
    """Get explanation visualization"""
    
    try:
        result = await explainable_ai_service.get_explanation_result(request_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Explanation result not found")
        
        if visualization_type not in result.visualizations:
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        # Get base64 encoded image
        image_base64 = result.visualizations[visualization_type]
        
        # Decode base64
        image_data = base64.b64decode(image_base64)
        
        # Return image
        return StreamingResponse(
            BytesIO(image_data),
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={visualization_type}.png"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting explanation visualization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list", response_model=List[Dict[str, Any]])
async def list_explanations(
    limit: int = Query(50, ge=1, le=100),
    user: UserClaims = Depends(get_current_user)
):
    """List recent explanations"""
    
    try:
        explanations = await explainable_ai_service.list_explanations(limit)
        return explanations
        
    except Exception as e:
        logger.error(f"Error listing explanations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ===== QUALITY VALIDATION =====

@router.post("/validate/{request_id}", response_model=Dict[str, Any])
async def validate_explanation_quality(
    request_id: str,
    quality_threshold: float = Query(0.7, ge=0.0, le=1.0),
    user: UserClaims = Depends(get_current_user)
):
    """Validate explanation quality"""
    
    try:
        result = await explainable_ai_service.get_explanation_result(request_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Explanation result not found")
        
        quality_score = result.confidence
        
        validation_result = {
            "request_id": request_id,
            "quality_score": quality_score,
            "quality_threshold": quality_threshold,
            "passes_quality_check": quality_score >= quality_threshold,
            "explanation_method": result.explanation_method.value,
            "processing_time": result.processing_time,
            "metrics": result.metrics,
            "validation_timestamp": datetime.utcnow().isoformat()
        }
        
        return validation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating explanation quality: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ===== SERVICE INFORMATION =====

@router.get("/metrics", response_model=ServiceMetricsResponse)
async def get_service_metrics(
    user: UserClaims = Depends(get_current_user)
):
    """Get XAI service metrics"""
    
    try:
        metrics = await explainable_ai_service.get_service_metrics()
        
        return ServiceMetricsResponse(
            explanations_generated=metrics["metrics"]["explanations_generated"],
            local_explanations=metrics["metrics"]["local_explanations"],
            global_explanations=metrics["metrics"]["global_explanations"],
            fairness_analyses=metrics["metrics"]["fairness_analyses"],
            counterfactual_explanations=metrics["metrics"]["counterfactual_explanations"],
            average_processing_time=metrics["metrics"]["average_processing_time"],
            active_requests=metrics["active_requests"],
            completed_explanations=metrics["completed_explanations"],
            cached_explainers=metrics["cached_explainers"]
        )
        
    except Exception as e:
        logger.error(f"Error getting service metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    user: UserClaims = Depends(get_current_user)
):
    """Health check endpoint"""
    
    try:
        metrics = await explainable_ai_service.get_service_metrics()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "active_requests": metrics["active_requests"],
            "total_explanations": metrics["completed_explanations"],
            "service_version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        } 