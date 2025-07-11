"""
A/B Testing Framework Service

This service provides comprehensive A/B testing capabilities for the Q Platform:
- Experiment design and management
- Statistical analysis and significance testing
- Multi-variant testing (A/B/C/D/...)
- Real-time monitoring and alerts
- Integration with model deployment service
- Audience segmentation and targeting
- Performance metrics collection and analysis
- Automated experiment lifecycle management
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict, deque
import statistics
import math

# ML and statistical libraries
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    sklearn_available = True
except ImportError:
    sklearn_available = False
    logging.warning("Scikit-learn not available - advanced analytics will be limited")

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.vault_client import VaultClient
from .model_deployment_service import ModelDeploymentService

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Experiment status values"""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"

class VariantType(Enum):
    """Variant types"""
    CONTROL = "control"
    TREATMENT = "treatment"
    CHAMPION = "champion"
    CHALLENGER = "challenger"

class MetricType(Enum):
    """Metric types for statistical analysis"""
    CONVERSION_RATE = "conversion_rate"
    CLICK_THROUGH_RATE = "click_through_rate"
    REVENUE = "revenue"
    ENGAGEMENT = "engagement"
    RETENTION = "retention"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"

class StatisticalTest(Enum):
    """Statistical tests for significance"""
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    MANN_WHITNEY = "mann_whitney"
    FISHER_EXACT = "fisher_exact"
    BAYESIAN = "bayesian"

class TrafficAllocation(Enum):
    """Traffic allocation strategies"""
    EQUAL = "equal"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"
    THOMPSON_SAMPLING = "thompson_sampling"

@dataclass
class ExperimentVariant:
    """Experiment variant configuration"""
    variant_id: str
    name: str
    description: str
    variant_type: VariantType
    traffic_allocation: float
    configuration: Dict[str, Any]
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    deployment_id: Optional[str] = None
    
    def __post_init__(self):
        if self.configuration is None:
            self.configuration = {}

@dataclass
class ExperimentMetric:
    """Experiment metric definition"""
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    primary: bool = False
    higher_is_better: bool = True
    minimum_detectable_effect: float = 0.05
    statistical_test: StatisticalTest = StatisticalTest.T_TEST
    confidence_level: float = 0.95

@dataclass
class AudienceSegment:
    """Audience segment definition"""
    segment_id: str
    name: str
    description: str
    criteria: Dict[str, Any]
    size_estimate: Optional[int] = None
    
    def __post_init__(self):
        if self.criteria is None:
            self.criteria = {}

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    objective: str
    variants: List[ExperimentVariant]
    metrics: List[ExperimentMetric]
    audience_segments: List[AudienceSegment]
    traffic_allocation: TrafficAllocation
    start_date: datetime
    end_date: datetime
    sample_size: int
    confidence_level: float = 0.95
    statistical_power: float = 0.8
    early_stopping_enabled: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ExperimentResult:
    """Experiment result data"""
    experiment_id: str
    variant_id: str
    metric_id: str
    timestamp: datetime
    value: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

@dataclass
class StatisticalAnalysis:
    """Statistical analysis results"""
    metric_id: str
    variant_comparisons: List[Dict[str, Any]]
    statistical_significance: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    sample_size: int
    test_statistic: float
    degrees_of_freedom: Optional[int] = None
    bayesian_probability: Optional[float] = None

@dataclass
class ExperimentSummary:
    """Experiment summary and results"""
    experiment_id: str
    status: ExperimentStatus
    start_date: datetime
    end_date: Optional[datetime]
    total_participants: int
    variant_performance: Dict[str, Dict[str, float]]
    statistical_analyses: List[StatisticalAnalysis]
    winning_variant: Optional[str] = None
    confidence_score: float = 0.0
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

class ABTestingService:
    """
    Comprehensive A/B Testing Framework Service
    """
    
    def __init__(self, storage_path: str = "ab_testing"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Experiment management
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        self.experiment_summaries: Dict[str, ExperimentSummary] = {}
        
        # Real-time tracking
        self.active_experiments: Set[str] = set()
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {exp_id: variant_id}
        
        # Traffic allocation
        self.traffic_allocators: Dict[str, Any] = {}
        
        # Statistical analysis
        self.statistical_tests: Dict[StatisticalTest, Callable] = {
            StatisticalTest.T_TEST: self._t_test,
            StatisticalTest.CHI_SQUARE: self._chi_square_test,
            StatisticalTest.MANN_WHITNEY: self._mann_whitney_test,
            StatisticalTest.FISHER_EXACT: self._fisher_exact_test,
            StatisticalTest.BAYESIAN: self._bayesian_test
        }
        
        # Performance metrics
        self.performance_metrics = {
            "total_experiments": 0,
            "active_experiments": 0,
            "completed_experiments": 0,
            "total_participants": 0,
            "significant_results": 0,
            "average_experiment_duration": 0.0
        }
        
        # Service integrations
        self.model_deployment_service: Optional[ModelDeploymentService] = None
        self.vault_client = VaultClient()
        
        # Configuration
        self.config = {
            "min_sample_size": 100,
            "max_experiment_duration": 30,  # days
            "significance_threshold": 0.05,
            "minimum_effect_size": 0.02,
            "early_stopping_check_interval": 3600,  # seconds
            "max_variants_per_experiment": 10,
            "enable_real_time_monitoring": True,
            "enable_automated_analysis": True
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        logger.info("A/B Testing Service initialized")
    
    async def initialize(self):
        """Initialize the A/B testing service"""
        logger.info("Initializing A/B Testing Service")
        
        # Initialize service integrations
        await self._initialize_service_integrations()
        
        # Load existing experiments
        await self._load_experiments()
        
        # Start background monitoring
        await self._start_background_monitoring()
        
        # Subscribe to events
        await self._subscribe_to_events()
        
        logger.info("A/B Testing Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the A/B testing service"""
        logger.info("Shutting down A/B Testing Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Save experiment state
        await self._save_experiments()
        
        logger.info("A/B Testing Service shutdown complete")
    
    # ===== EXPERIMENT MANAGEMENT =====
    
    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B test experiment"""
        
        experiment_id = config.experiment_id or f"exp_{uuid.uuid4().hex[:12]}"
        
        try:
            # Validate experiment configuration
            await self._validate_experiment_config(config)
            
            # Calculate sample size if not provided
            if config.sample_size <= 0:
                config.sample_size = await self._calculate_sample_size(config)
            
            # Store experiment
            config.experiment_id = experiment_id
            self.experiments[experiment_id] = config
            
            # Initialize experiment tracking
            self.experiment_results[experiment_id] = []
            self.user_assignments[experiment_id] = {}
            
            # Create traffic allocator
            await self._create_traffic_allocator(experiment_id, config)
            
            # Create experiment summary
            summary = ExperimentSummary(
                experiment_id=experiment_id,
                status=ExperimentStatus.DRAFT,
                start_date=config.start_date,
                end_date=config.end_date,
                total_participants=0,
                variant_performance={},
                statistical_analyses=[]
            )
            self.experiment_summaries[experiment_id] = summary
            
            # Update metrics
            self.performance_metrics["total_experiments"] += 1
            
            # Publish event
            await self._publish_experiment_event(experiment_id, "created")
            
            logger.info(f"Created experiment: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            raise
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        try:
            # Check if experiment can be started
            if datetime.utcnow() < experiment.start_date:
                raise ValueError("Experiment start date is in the future")
            
            # Deploy model variants if needed
            await self._deploy_experiment_variants(experiment_id)
            
            # Start experiment
            self.active_experiments.add(experiment_id)
            summary = self.experiment_summaries[experiment_id]
            summary.status = ExperimentStatus.RUNNING
            
            # Update metrics
            self.performance_metrics["active_experiments"] += 1
            
            # Publish event
            await self._publish_experiment_event(experiment_id, "started")
            
            logger.info(f"Started experiment: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting experiment: {e}")
            return False
    
    async def stop_experiment(self, experiment_id: str, reason: str = "Manual stop") -> bool:
        """Stop an experiment"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        try:
            # Stop experiment
            self.active_experiments.discard(experiment_id)
            summary = self.experiment_summaries[experiment_id]
            summary.status = ExperimentStatus.COMPLETED
            summary.end_date = datetime.utcnow()
            
            # Perform final analysis
            await self._analyze_experiment_results(experiment_id)
            
            # Clean up deployments if needed
            await self._cleanup_experiment_deployments(experiment_id)
            
            # Update metrics
            self.performance_metrics["active_experiments"] -= 1
            self.performance_metrics["completed_experiments"] += 1
            
            # Publish event
            await self._publish_experiment_event(experiment_id, "stopped", {"reason": reason})
            
            logger.info(f"Stopped experiment: {experiment_id}, reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping experiment: {e}")
            return False
    
    # ===== TRAFFIC ALLOCATION =====
    
    async def assign_user_to_variant(
        self, 
        experiment_id: str, 
        user_id: str, 
        context: Dict[str, Any] = None
    ) -> Optional[str]:
        """Assign a user to an experiment variant"""
        
        if experiment_id not in self.experiments:
            return None
        
        if experiment_id not in self.active_experiments:
            return None
        
        # Check if user is already assigned
        if experiment_id in self.user_assignments and user_id in self.user_assignments[experiment_id]:
            return self.user_assignments[experiment_id][user_id]
        
        experiment = self.experiments[experiment_id]
        
        try:
            # Check audience segments
            if not await self._user_matches_audience(user_id, experiment.audience_segments, context):
                return None
            
            # Assign variant based on traffic allocation strategy
            variant_id = await self._allocate_traffic(experiment_id, user_id, context)
            
            # Store assignment
            if experiment_id not in self.user_assignments:
                self.user_assignments[experiment_id] = {}
            self.user_assignments[experiment_id][user_id] = variant_id
            
            # Update participant count
            summary = self.experiment_summaries[experiment_id]
            summary.total_participants += 1
            
            logger.debug(f"Assigned user {user_id} to variant {variant_id} in experiment {experiment_id}")
            return variant_id
            
        except Exception as e:
            logger.error(f"Error assigning user to variant: {e}")
            return None
    
    async def _allocate_traffic(self, experiment_id: str, user_id: str, context: Dict[str, Any]) -> str:
        """Allocate traffic to variants based on allocation strategy"""
        
        experiment = self.experiments[experiment_id]
        
        if experiment.traffic_allocation == TrafficAllocation.EQUAL:
            return await self._equal_allocation(experiment_id, user_id)
        elif experiment.traffic_allocation == TrafficAllocation.WEIGHTED:
            return await self._weighted_allocation(experiment_id, user_id)
        elif experiment.traffic_allocation == TrafficAllocation.ADAPTIVE:
            return await self._adaptive_allocation(experiment_id, user_id)
        elif experiment.traffic_allocation == TrafficAllocation.THOMPSON_SAMPLING:
            return await self._thompson_sampling_allocation(experiment_id, user_id)
        else:
            return await self._equal_allocation(experiment_id, user_id)
    
    async def _equal_allocation(self, experiment_id: str, user_id: str) -> str:
        """Allocate traffic equally among variants"""
        
        experiment = self.experiments[experiment_id]
        
        # Create a consistent hash for the user
        user_hash = int(hashlib.md5(f"{experiment_id}:{user_id}".encode()).hexdigest(), 16)
        
        # Allocate to variant based on hash
        variant_index = user_hash % len(experiment.variants)
        return experiment.variants[variant_index].variant_id
    
    async def _weighted_allocation(self, experiment_id: str, user_id: str) -> str:
        """Allocate traffic based on variant weights"""
        
        experiment = self.experiments[experiment_id]
        
        # Create a consistent hash for the user
        user_hash = int(hashlib.md5(f"{experiment_id}:{user_id}".encode()).hexdigest(), 16)
        random_value = (user_hash % 10000) / 10000.0  # 0-1 range
        
        # Find variant based on cumulative weights
        cumulative_weight = 0.0
        for variant in experiment.variants:
            cumulative_weight += variant.traffic_allocation / 100.0
            if random_value <= cumulative_weight:
                return variant.variant_id
        
        # Fallback to last variant
        return experiment.variants[-1].variant_id
    
    async def _adaptive_allocation(self, experiment_id: str, user_id: str) -> str:
        """Allocate traffic adaptively based on performance"""
        
        # This would implement adaptive allocation based on current performance
        # For now, fallback to weighted allocation
        return await self._weighted_allocation(experiment_id, user_id)
    
    async def _thompson_sampling_allocation(self, experiment_id: str, user_id: str) -> str:
        """Allocate traffic using Thompson sampling"""
        
        # This would implement Thompson sampling for multi-armed bandit
        # For now, fallback to weighted allocation
        return await self._weighted_allocation(experiment_id, user_id)
    
    # ===== RESULT TRACKING =====
    
    async def track_event(
        self,
        experiment_id: str,
        user_id: str,
        metric_id: str,
        value: float,
        context: Dict[str, Any] = None
    ) -> bool:
        """Track an event/metric for an experiment"""
        
        if experiment_id not in self.active_experiments:
            return False
        
        if experiment_id not in self.user_assignments:
            return False
        
        if user_id not in self.user_assignments[experiment_id]:
            return False
        
        try:
            variant_id = self.user_assignments[experiment_id][user_id]
            
            # Create result record
            result = ExperimentResult(
                experiment_id=experiment_id,
                variant_id=variant_id,
                metric_id=metric_id,
                timestamp=datetime.utcnow(),
                value=value,
                user_id=user_id,
                context=context or {}
            )
            
            # Store result
            self.experiment_results[experiment_id].append(result)
            
            # Update performance metrics
            self.performance_metrics["total_participants"] += 1
            
            logger.debug(f"Tracked event: {experiment_id}/{variant_id}/{metric_id} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking event: {e}")
            return False
    
    # ===== STATISTICAL ANALYSIS =====
    
    async def analyze_experiment(self, experiment_id: str) -> Optional[List[StatisticalAnalysis]]:
        """Perform statistical analysis on experiment results"""
        
        if experiment_id not in self.experiments:
            return None
        
        try:
            return await self._analyze_experiment_results(experiment_id)
            
        except Exception as e:
            logger.error(f"Error analyzing experiment: {e}")
            return None
    
    async def _analyze_experiment_results(self, experiment_id: str) -> List[StatisticalAnalysis]:
        """Analyze experiment results and compute statistical significance"""
        
        experiment = self.experiments[experiment_id]
        results = self.experiment_results[experiment_id]
        
        analyses = []
        
        # Group results by metric
        metric_results = defaultdict(lambda: defaultdict(list))
        for result in results:
            metric_results[result.metric_id][result.variant_id].append(result.value)
        
        # Analyze each metric
        for metric in experiment.metrics:
            if metric.metric_id not in metric_results:
                continue
            
            metric_data = metric_results[metric.metric_id]
            
            # Get control variant data
            control_variant = next((v for v in experiment.variants if v.variant_type == VariantType.CONTROL), None)
            if not control_variant or control_variant.variant_id not in metric_data:
                continue
            
            control_data = metric_data[control_variant.variant_id]
            
            # Compare each treatment variant with control
            variant_comparisons = []
            
            for variant in experiment.variants:
                if variant.variant_type == VariantType.CONTROL:
                    continue
                
                if variant.variant_id not in metric_data:
                    continue
                
                treatment_data = metric_data[variant.variant_id]
                
                # Perform statistical test
                test_result = await self._perform_statistical_test(
                    control_data, treatment_data, metric.statistical_test
                )
                
                variant_comparisons.append({
                    "variant_id": variant.variant_id,
                    "control_mean": np.mean(control_data),
                    "treatment_mean": np.mean(treatment_data),
                    "control_std": np.std(control_data),
                    "treatment_std": np.std(treatment_data),
                    "control_count": len(control_data),
                    "treatment_count": len(treatment_data),
                    "p_value": test_result["p_value"],
                    "test_statistic": test_result["test_statistic"],
                    "effect_size": test_result["effect_size"]
                })
            
            # Create analysis result
            if variant_comparisons:
                best_comparison = min(variant_comparisons, key=lambda x: x["p_value"])
                
                analysis = StatisticalAnalysis(
                    metric_id=metric.metric_id,
                    variant_comparisons=variant_comparisons,
                    statistical_significance=best_comparison["p_value"] < self.config["significance_threshold"],
                    p_value=best_comparison["p_value"],
                    confidence_interval=self._calculate_confidence_interval(control_data, metric.confidence_level),
                    effect_size=best_comparison["effect_size"],
                    power=self._calculate_power(control_data, treatment_data, metric.confidence_level),
                    sample_size=len(control_data) + len(treatment_data),
                    test_statistic=best_comparison["test_statistic"]
                )
                
                analyses.append(analysis)
        
        # Store analyses
        self.experiment_summaries[experiment_id].statistical_analyses = analyses
        
        # Determine winning variant
        await self._determine_winning_variant(experiment_id, analyses)
        
        return analyses
    
    async def _perform_statistical_test(
        self, 
        control_data: List[float], 
        treatment_data: List[float], 
        test_type: StatisticalTest
    ) -> Dict[str, float]:
        """Perform statistical test between control and treatment groups"""
        
        test_function = self.statistical_tests.get(test_type, self._t_test)
        return await test_function(control_data, treatment_data)
    
    async def _t_test(self, control_data: List[float], treatment_data: List[float]) -> Dict[str, float]:
        """Perform independent t-test"""
        
        try:
            control_array = np.array(control_data)
            treatment_array = np.array(treatment_data)
            
            statistic, p_value = stats.ttest_ind(control_array, treatment_array)
            
            # Calculate Cohen's d (effect size)
            pooled_std = np.sqrt(((len(control_array) - 1) * np.var(control_array) + 
                                 (len(treatment_array) - 1) * np.var(treatment_array)) / 
                                (len(control_array) + len(treatment_array) - 2))
            
            effect_size = (np.mean(treatment_array) - np.mean(control_array)) / pooled_std
            
            return {
                "test_statistic": float(statistic),
                "p_value": float(p_value),
                "effect_size": float(effect_size)
            }
            
        except Exception as e:
            logger.error(f"Error in t-test: {e}")
            return {"test_statistic": 0.0, "p_value": 1.0, "effect_size": 0.0}
    
    async def _chi_square_test(self, control_data: List[float], treatment_data: List[float]) -> Dict[str, float]:
        """Perform chi-square test for categorical data"""
        
        try:
            # Create contingency table
            control_success = sum(1 for x in control_data if x > 0)
            control_failure = len(control_data) - control_success
            treatment_success = sum(1 for x in treatment_data if x > 0)
            treatment_failure = len(treatment_data) - treatment_success
            
            contingency_table = np.array([[control_success, control_failure],
                                        [treatment_success, treatment_failure]])
            
            statistic, p_value, _, _ = stats.chi2_contingency(contingency_table)
            
            # Calculate Cramér's V (effect size)
            n = np.sum(contingency_table)
            effect_size = np.sqrt(statistic / (n * (min(contingency_table.shape) - 1)))
            
            return {
                "test_statistic": float(statistic),
                "p_value": float(p_value),
                "effect_size": float(effect_size)
            }
            
        except Exception as e:
            logger.error(f"Error in chi-square test: {e}")
            return {"test_statistic": 0.0, "p_value": 1.0, "effect_size": 0.0}
    
    async def _mann_whitney_test(self, control_data: List[float], treatment_data: List[float]) -> Dict[str, float]:
        """Perform Mann-Whitney U test (non-parametric)"""
        
        try:
            statistic, p_value = stats.mannwhitneyu(control_data, treatment_data, alternative='two-sided')
            
            # Calculate effect size (r)
            n1, n2 = len(control_data), len(treatment_data)
            effect_size = statistic / (n1 * n2)
            
            return {
                "test_statistic": float(statistic),
                "p_value": float(p_value),
                "effect_size": float(effect_size)
            }
            
        except Exception as e:
            logger.error(f"Error in Mann-Whitney test: {e}")
            return {"test_statistic": 0.0, "p_value": 1.0, "effect_size": 0.0}
    
    async def _fisher_exact_test(self, control_data: List[float], treatment_data: List[float]) -> Dict[str, float]:
        """Perform Fisher's exact test"""
        
        try:
            # Create contingency table
            control_success = sum(1 for x in control_data if x > 0)
            control_failure = len(control_data) - control_success
            treatment_success = sum(1 for x in treatment_data if x > 0)
            treatment_failure = len(treatment_data) - treatment_success
            
            contingency_table = [[control_success, control_failure],
                               [treatment_success, treatment_failure]]
            
            _, p_value = stats.fisher_exact(contingency_table)
            
            # Calculate odds ratio as effect size
            odds_ratio = (control_success * treatment_failure) / (control_failure * treatment_success)
            effect_size = np.log(odds_ratio) if odds_ratio > 0 else 0.0
            
            return {
                "test_statistic": float(odds_ratio),
                "p_value": float(p_value),
                "effect_size": float(effect_size)
            }
            
        except Exception as e:
            logger.error(f"Error in Fisher's exact test: {e}")
            return {"test_statistic": 0.0, "p_value": 1.0, "effect_size": 0.0}
    
    async def _bayesian_test(self, control_data: List[float], treatment_data: List[float]) -> Dict[str, float]:
        """Perform Bayesian A/B test"""
        
        # Simplified Bayesian analysis
        # In practice, this would use more sophisticated Bayesian methods
        try:
            control_mean = np.mean(control_data)
            treatment_mean = np.mean(treatment_data)
            
            # Use t-test as approximation
            t_result = await self._t_test(control_data, treatment_data)
            
            # Convert p-value to Bayesian probability (simplified)
            bayesian_probability = 1.0 - t_result["p_value"]
            
            return {
                "test_statistic": float(treatment_mean - control_mean),
                "p_value": t_result["p_value"],
                "effect_size": t_result["effect_size"],
                "bayesian_probability": float(bayesian_probability)
            }
            
        except Exception as e:
            logger.error(f"Error in Bayesian test: {e}")
            return {"test_statistic": 0.0, "p_value": 1.0, "effect_size": 0.0, "bayesian_probability": 0.5}
    
    def _calculate_confidence_interval(self, data: List[float], confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for the data"""
        
        try:
            alpha = 1.0 - confidence_level
            n = len(data)
            mean = np.mean(data)
            std_err = stats.sem(data)
            
            # Use t-distribution for small samples
            t_critical = stats.t.ppf(1 - alpha/2, n - 1)
            margin_of_error = t_critical * std_err
            
            return (mean - margin_of_error, mean + margin_of_error)
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return (0.0, 0.0)
    
    def _calculate_power(self, control_data: List[float], treatment_data: List[float], confidence_level: float) -> float:
        """Calculate statistical power"""
        
        try:
            # Simplified power calculation
            # In practice, this would use more sophisticated methods
            n1, n2 = len(control_data), len(treatment_data)
            
            if n1 < 2 or n2 < 2:
                return 0.0
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((n1 - 1) * np.var(control_data) + 
                                 (n2 - 1) * np.var(treatment_data)) / 
                                (n1 + n2 - 2))
            
            effect_size = abs(np.mean(treatment_data) - np.mean(control_data)) / pooled_std
            
            # Approximate power calculation
            alpha = 1.0 - confidence_level
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = effect_size * np.sqrt(n1 * n2 / (n1 + n2)) - z_alpha
            
            power = stats.norm.cdf(z_beta)
            
            return max(0.0, min(1.0, power))
            
        except Exception as e:
            logger.error(f"Error calculating power: {e}")
            return 0.0
    
    # ===== UTILITY METHODS =====
    
    async def _validate_experiment_config(self, config: ExperimentConfig):
        """Validate experiment configuration"""
        
        # Check variants
        if not config.variants:
            raise ValueError("Experiment must have at least one variant")
        
        if len(config.variants) > self.config["max_variants_per_experiment"]:
            raise ValueError(f"Maximum {self.config['max_variants_per_experiment']} variants allowed")
        
        # Check control variant
        control_variants = [v for v in config.variants if v.variant_type == VariantType.CONTROL]
        if len(control_variants) != 1:
            raise ValueError("Experiment must have exactly one control variant")
        
        # Check traffic allocation
        total_allocation = sum(v.traffic_allocation for v in config.variants)
        if abs(total_allocation - 100.0) > 0.01:
            raise ValueError("Traffic allocation must sum to 100%")
        
        # Check metrics
        if not config.metrics:
            raise ValueError("Experiment must have at least one metric")
        
        primary_metrics = [m for m in config.metrics if m.primary]
        if len(primary_metrics) != 1:
            raise ValueError("Experiment must have exactly one primary metric")
        
        # Check dates
        if config.start_date >= config.end_date:
            raise ValueError("Start date must be before end date")
        
        duration = (config.end_date - config.start_date).days
        if duration > self.config["max_experiment_duration"]:
            raise ValueError(f"Maximum experiment duration is {self.config['max_experiment_duration']} days")
    
    async def _calculate_sample_size(self, config: ExperimentConfig) -> int:
        """Calculate required sample size for the experiment"""
        
        # Get primary metric
        primary_metric = next(m for m in config.metrics if m.primary)
        
        # Calculate sample size using power analysis
        try:
            alpha = 1.0 - config.confidence_level
            beta = 1.0 - config.statistical_power
            effect_size = primary_metric.minimum_detectable_effect
            
            # Simplified sample size calculation
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(config.statistical_power)
            
            sample_size_per_variant = ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
            total_sample_size = int(sample_size_per_variant * len(config.variants))
            
            return max(total_sample_size, self.config["min_sample_size"])
            
        except Exception as e:
            logger.error(f"Error calculating sample size: {e}")
            return self.config["min_sample_size"]
    
    async def _user_matches_audience(
        self, 
        user_id: str, 
        segments: List[AudienceSegment], 
        context: Dict[str, Any]
    ) -> bool:
        """Check if user matches audience segmentation criteria"""
        
        if not segments:
            return True
        
        # Simple implementation - check if user context matches any segment
        for segment in segments:
            if await self._user_matches_segment(user_id, segment, context):
                return True
        
        return False
    
    async def _user_matches_segment(
        self, 
        user_id: str, 
        segment: AudienceSegment, 
        context: Dict[str, Any]
    ) -> bool:
        """Check if user matches a specific audience segment"""
        
        try:
            # Simple rule-based matching
            for criterion, expected_value in segment.criteria.items():
                if criterion not in context:
                    return False
                
                actual_value = context[criterion]
                
                # Handle different comparison types
                if isinstance(expected_value, dict):
                    if "min" in expected_value and actual_value < expected_value["min"]:
                        return False
                    if "max" in expected_value and actual_value > expected_value["max"]:
                        return False
                    if "in" in expected_value and actual_value not in expected_value["in"]:
                        return False
                else:
                    if actual_value != expected_value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error matching user to segment: {e}")
            return False
    
    async def _determine_winning_variant(self, experiment_id: str, analyses: List[StatisticalAnalysis]):
        """Determine the winning variant based on statistical analysis"""
        
        experiment = self.experiments[experiment_id]
        summary = self.experiment_summaries[experiment_id]
        
        # Get primary metric analysis
        primary_metric = next(m for m in experiment.metrics if m.primary)
        primary_analysis = next((a for a in analyses if a.metric_id == primary_metric.metric_id), None)
        
        if not primary_analysis or not primary_analysis.statistical_significance:
            summary.winning_variant = None
            summary.confidence_score = 0.0
            return
        
        # Find best performing variant
        best_variant = None
        best_performance = None
        
        for comparison in primary_analysis.variant_comparisons:
            variant_id = comparison["variant_id"]
            
            if primary_metric.higher_is_better:
                performance = comparison["treatment_mean"]
            else:
                performance = -comparison["treatment_mean"]
            
            if best_performance is None or performance > best_performance:
                best_performance = performance
                best_variant = variant_id
        
        summary.winning_variant = best_variant
        summary.confidence_score = 1.0 - primary_analysis.p_value
        
        # Generate recommendations
        if primary_analysis.statistical_significance:
            effect_size = primary_analysis.effect_size
            
            if abs(effect_size) > 0.2:
                summary.recommendations.append(f"Strong effect detected for {best_variant}")
            elif abs(effect_size) > 0.1:
                summary.recommendations.append(f"Moderate effect detected for {best_variant}")
            else:
                summary.recommendations.append(f"Small but significant effect detected for {best_variant}")
        
        # Update metrics
        if primary_analysis.statistical_significance:
            self.performance_metrics["significant_results"] += 1
    
    # ===== INTEGRATION METHODS =====
    
    async def _deploy_experiment_variants(self, experiment_id: str):
        """Deploy model variants for the experiment"""
        
        if not self.model_deployment_service:
            return
        
        experiment = self.experiments[experiment_id]
        
        for variant in experiment.variants:
            if variant.model_id and variant.model_version:
                # Deploy model variant
                # This would integrate with the model deployment service
                logger.info(f"Deploying model variant {variant.variant_id} for experiment {experiment_id}")
    
    async def _cleanup_experiment_deployments(self, experiment_id: str):
        """Clean up deployments for finished experiment"""
        
        if not self.model_deployment_service:
            return
        
        experiment = self.experiments[experiment_id]
        
        for variant in experiment.variants:
            if variant.deployment_id:
                # Clean up deployment
                logger.info(f"Cleaning up deployment {variant.deployment_id} for experiment {experiment_id}")
    
    async def _create_traffic_allocator(self, experiment_id: str, config: ExperimentConfig):
        """Create traffic allocator for the experiment"""
        
        # Create allocator based on strategy
        if config.traffic_allocation == TrafficAllocation.THOMPSON_SAMPLING:
            # Initialize Thompson sampling parameters
            self.traffic_allocators[experiment_id] = {
                "type": "thompson_sampling",
                "alpha": {v.variant_id: 1.0 for v in config.variants},
                "beta": {v.variant_id: 1.0 for v in config.variants}
            }
        else:
            # Default allocator
            self.traffic_allocators[experiment_id] = {
                "type": "static",
                "weights": {v.variant_id: v.traffic_allocation for v in config.variants}
            }
    
    # ===== BACKGROUND MONITORING =====
    
    async def _start_background_monitoring(self):
        """Start background monitoring tasks"""
        
        # Early stopping monitoring
        if self.config["enable_real_time_monitoring"]:
            task = asyncio.create_task(self._early_stopping_monitor())
            self.background_tasks.add(task)
        
        # Automated analysis
        if self.config["enable_automated_analysis"]:
            task = asyncio.create_task(self._automated_analysis_loop())
            self.background_tasks.add(task)
    
    async def _early_stopping_monitor(self):
        """Monitor experiments for early stopping conditions"""
        
        while True:
            try:
                await asyncio.sleep(self.config["early_stopping_check_interval"])
                
                for experiment_id in list(self.active_experiments):
                    experiment = self.experiments[experiment_id]
                    
                    if not experiment.early_stopping_enabled:
                        continue
                    
                    # Check if experiment should be stopped early
                    should_stop = await self._check_early_stopping_conditions(experiment_id)
                    
                    if should_stop:
                        await self.stop_experiment(experiment_id, "Early stopping triggered")
                
            except Exception as e:
                logger.error(f"Error in early stopping monitor: {e}")
    
    async def _automated_analysis_loop(self):
        """Automated analysis loop for active experiments"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                for experiment_id in list(self.active_experiments):
                    # Perform interim analysis
                    await self._analyze_experiment_results(experiment_id)
                
            except Exception as e:
                logger.error(f"Error in automated analysis loop: {e}")
    
    async def _check_early_stopping_conditions(self, experiment_id: str) -> bool:
        """Check if experiment should be stopped early"""
        
        try:
            # Perform interim analysis
            analyses = await self._analyze_experiment_results(experiment_id)
            
            if not analyses:
                return False
            
            # Check for statistical significance on primary metric
            experiment = self.experiments[experiment_id]
            primary_metric = next(m for m in experiment.metrics if m.primary)
            primary_analysis = next((a for a in analyses if a.metric_id == primary_metric.metric_id), None)
            
            if not primary_analysis:
                return False
            
            # Stop if statistically significant with sufficient power
            if primary_analysis.statistical_significance and primary_analysis.power > 0.8:
                return True
            
            # Stop if clearly no effect and sufficient data
            if primary_analysis.p_value > 0.8 and primary_analysis.sample_size > experiment.sample_size:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking early stopping conditions: {e}")
            return False
    
    # ===== PERSISTENCE =====
    
    async def _load_experiments(self):
        """Load experiments from storage"""
        
        try:
            experiments_file = self.storage_path / "experiments.json"
            if experiments_file.exists():
                with open(experiments_file, 'r') as f:
                    experiments_data = json.load(f)
                
                for exp_data in experiments_data:
                    # Reconstruct experiment objects
                    config = ExperimentConfig(**exp_data)
                    self.experiments[config.experiment_id] = config
                    
                    # Restore active experiments
                    summary_data = exp_data.get("summary", {})
                    if summary_data.get("status") == "running":
                        self.active_experiments.add(config.experiment_id)
            
            logger.info("Experiments loaded from storage")
            
        except Exception as e:
            logger.warning(f"Error loading experiments: {e}")
    
    async def _save_experiments(self):
        """Save experiments to storage"""
        
        try:
            experiments_data = []
            for experiment_id, config in self.experiments.items():
                exp_data = asdict(config)
                
                # Add summary data
                if experiment_id in self.experiment_summaries:
                    exp_data["summary"] = asdict(self.experiment_summaries[experiment_id])
                
                experiments_data.append(exp_data)
            
            experiments_file = self.storage_path / "experiments.json"
            with open(experiments_file, 'w') as f:
                json.dump(experiments_data, f, indent=2, default=str)
            
            logger.info("Experiments saved to storage")
            
        except Exception as e:
            logger.warning(f"Error saving experiments: {e}")
    
    # ===== EVENT HANDLING =====
    
    async def _initialize_service_integrations(self):
        """Initialize service integrations"""
        
        try:
            # Initialize model deployment service integration
            # This would be injected or discovered
            pass
            
        except Exception as e:
            logger.warning(f"Error initializing service integrations: {e}")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events"""
        
        try:
            await shared_pulsar_client.subscribe(
                "q.deployment.events",
                self._handle_deployment_event,
                subscription_name="ab_testing_deployment_events"
            )
            
        except Exception as e:
            logger.warning(f"Error subscribing to events: {e}")
    
    async def _handle_deployment_event(self, event_data: Dict[str, Any]):
        """Handle deployment events"""
        
        try:
            deployment_id = event_data.get("deployment_id")
            event_type = event_data.get("event_type")
            
            # Link deployment events to experiments
            for experiment_id, experiment in self.experiments.items():
                for variant in experiment.variants:
                    if variant.deployment_id == deployment_id:
                        logger.info(f"Deployment event {event_type} for experiment {experiment_id} variant {variant.variant_id}")
                        break
            
        except Exception as e:
            logger.error(f"Error handling deployment event: {e}")
    
    async def _publish_experiment_event(self, experiment_id: str, event_type: str, data: Dict[str, Any] = None):
        """Publish experiment event"""
        
        try:
            event_data = {
                "experiment_id": experiment_id,
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                **(data or {})
            }
            
            await shared_pulsar_client.publish("q.ab_testing.events", event_data)
            
        except Exception as e:
            logger.warning(f"Error publishing experiment event: {e}")
    
    # ===== API METHODS =====
    
    async def get_experiment_summary(self, experiment_id: str) -> Optional[ExperimentSummary]:
        """Get experiment summary"""
        
        return self.experiment_summaries.get(experiment_id)
    
    async def get_experiment_results(self, experiment_id: str) -> List[ExperimentResult]:
        """Get experiment results"""
        
        return self.experiment_results.get(experiment_id, [])
    
    async def get_active_experiments(self) -> List[str]:
        """Get list of active experiment IDs"""
        
        return list(self.active_experiments)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        return self.performance_metrics.copy()

# Global service instance
ab_testing_service = ABTestingService() 