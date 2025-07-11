"""
package ai.qagi.common.dto.automl;

AutoML Service

This service provides automated machine learning capabilities:
- Automated model selection and architecture search
- Hyperparameter optimization using Spark for distributed processing
- Feature engineering and selection
- Model ensembling and stacking
- Performance evaluation and comparison
- Integration with federated learning
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import joblib
from collections import defaultdict

# ML Libraries
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
import mlflow
import mlflow.pytorch
import mlflow.sklearn

# Spark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as SparkRandomForest
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.q_vectorstore_client.client import VectorStoreClient
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.vault_client import VaultClient

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"

class OptimizationObjective(Enum):
    """Optimization objectives"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1"
    ROC_AUC = "roc_auc"
    MSE = "mse"
    MAE = "mae"
    R2_SCORE = "r2"

class AutoMLStatus(Enum):
    """AutoML experiment status"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AutoMLExperiment:
    """AutoML experiment configuration"""
    experiment_id: str
    experiment_name: str
    model_type: ModelType
    optimization_objective: OptimizationObjective
    dataset_config: Dict[str, Any]
    training_config: Dict[str, Any]
    search_space: Dict[str, Any]
    status: AutoMLStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    best_model: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    trials_completed: int = 0
    total_trials: int = 100
    results_summary: Optional[Dict[str, Any]] = None

@dataclass
class ModelCandidate:
    """Model candidate in AutoML search"""
    model_id: str
    experiment_id: str
    model_type: str
    model_name: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    training_time: float
    model_size: int
    feature_importance: Optional[Dict[str, float]] = None
    cross_validation_scores: Optional[List[float]] = None
    created_at: datetime = None

@dataclass
class FeatureEngineeringConfig:
    """Feature engineering configuration"""
    auto_feature_selection: bool = True
    polynomial_features: bool = False
    interaction_features: bool = False
    scaling_method: str = "standard"
    dimensionality_reduction: Optional[str] = None
    feature_selection_method: str = "mutual_info"
    max_features: Optional[int] = None

class AutoMLService:
    """
    Automated Machine Learning Service
    """
    
    def __init__(self, 
                 model_storage_path: str = "models/automl",
                 spark_config: Optional[Dict[str, Any]] = None):
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.spark_config = spark_config or {}
        self.spark_session = None
        
        # Active experiments
        self.active_experiments: Dict[str, AutoMLExperiment] = {}
        self.experiment_results: Dict[str, List[ModelCandidate]] = defaultdict(list)
        
        # Model registry
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.model_leaderboard: Dict[str, List[ModelCandidate]] = defaultdict(list)
        
        # Optimization studies
        self.optuna_studies: Dict[str, optuna.Study] = {}
        
        # Configuration
        self.config = {
            "default_trials": 100,
            "cv_folds": 5,
            "test_size": 0.2,
            "random_state": 42,
            "n_jobs": -1,
            "timeout_hours": 24,
            "early_stopping_rounds": 10
        }
        
        # Performance tracking
        self.automl_metrics = {
            "experiments_completed": 0,
            "models_trained": 0,
            "best_accuracy_achieved": 0.0,
            "average_experiment_time": 0.0,
            "total_compute_time": 0.0
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # MLflow tracking
        self.mlflow_tracking_uri = None
        
    async def initialize(self):
        """Initialize the AutoML service"""
        logger.info("Initializing AutoML Service")
        
        # Initialize Spark session
        await self._initialize_spark_session()
        
        # Initialize MLflow
        await self._initialize_mlflow()
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        # Load existing experiments
        await self._load_experiments()
        
        # Start background tasks
        self.background_tasks.add(asyncio.create_task(self._experiment_monitor()))
        self.background_tasks.add(asyncio.create_task(self._performance_tracking()))
        
        logger.info("AutoML Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the AutoML service"""
        logger.info("Shutting down AutoML Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Save experiments
        await self._save_experiments()
        
        # Stop Spark session
        if self.spark_session:
            self.spark_session.stop()
        
        logger.info("AutoML Service shut down successfully")
    
    # ===== EXPERIMENT MANAGEMENT =====
    
    async def start_automl_experiment(
        self,
        experiment_name: str,
        model_type: ModelType,
        dataset_config: Dict[str, Any],
        optimization_objective: OptimizationObjective = OptimizationObjective.ACCURACY,
        training_config: Optional[Dict[str, Any]] = None,
        search_space: Optional[Dict[str, Any]] = None,
        n_trials: int = 100,
        timeout_hours: int = 24
    ) -> str:
        """
        Start a new AutoML experiment
        
        Args:
            experiment_name: Name for the experiment
            model_type: Type of model to train
            dataset_config: Dataset configuration
            optimization_objective: Objective to optimize
            training_config: Training configuration
            search_space: Hyperparameter search space
            n_trials: Number of trials to run
            timeout_hours: Timeout in hours
            
        Returns:
            Experiment ID
        """
        experiment_id = f"automl_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Starting AutoML experiment: {experiment_name}")
        
        # Create experiment
        experiment = AutoMLExperiment(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            model_type=model_type,
            optimization_objective=optimization_objective,
            dataset_config=dataset_config,
            training_config=training_config or {},
            search_space=search_space or self._get_default_search_space(model_type),
            status=AutoMLStatus.INITIALIZING,
            created_at=datetime.utcnow(),
            total_trials=n_trials
        )
        
        self.active_experiments[experiment_id] = experiment
        
        # Start experiment in background
        asyncio.create_task(self._run_automl_experiment(experiment))
        
        # Publish experiment started event
        await shared_pulsar_client.publish(
            "q.ml.automl.experiment.started",
            {
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "model_type": model_type.value,
                "optimization_objective": optimization_objective.value,
                "n_trials": n_trials,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return experiment_id
    
    async def _run_automl_experiment(self, experiment: AutoMLExperiment):
        """Run AutoML experiment"""
        
        try:
            experiment.status = AutoMLStatus.RUNNING
            experiment.started_at = datetime.utcnow()
            
            logger.info(f"Running AutoML experiment: {experiment.experiment_id}")
            
            # Load and prepare dataset
            X, y = await self._load_dataset(experiment.dataset_config)
            
            # Feature engineering
            X_processed = await self._perform_feature_engineering(X, experiment.training_config)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, 
                test_size=self.config["test_size"],
                random_state=self.config["random_state"]
            )
            
            # Create MLflow experiment
            mlflow.create_experiment(experiment.experiment_name, artifact_location=str(self.model_storage_path))
            
            # Run hyperparameter optimization
            if experiment.model_type == ModelType.NEURAL_NETWORK:
                await self._run_neural_network_optimization(experiment, X_train, y_train, X_test, y_test)
            else:
                await self._run_traditional_ml_optimization(experiment, X_train, y_train, X_test, y_test)
            
            # Complete experiment
            experiment.status = AutoMLStatus.COMPLETED
            experiment.completed_at = datetime.utcnow()
            
            # Update metrics
            self.automl_metrics["experiments_completed"] += 1
            
            # Generate results summary
            experiment.results_summary = await self._generate_results_summary(experiment)
            
            logger.info(f"Completed AutoML experiment: {experiment.experiment_id}")
            
            # Publish experiment completed event
            await shared_pulsar_client.publish(
                "q.ml.automl.experiment.completed",
                {
                    "experiment_id": experiment.experiment_id,
                    "best_score": experiment.best_score,
                    "trials_completed": experiment.trials_completed,
                    "results_summary": experiment.results_summary,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"AutoML experiment failed: {experiment.experiment_id}: {e}")
            experiment.status = AutoMLStatus.FAILED
            experiment.completed_at = datetime.utcnow()
    
    async def _run_traditional_ml_optimization(
        self,
        experiment: AutoMLExperiment,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ):
        """Run optimization for traditional ML models"""
        
        def objective(trial):
            # Select model type
            model_name = trial.suggest_categorical(
                'model_name', 
                ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'mlp']
            )
            
            # Get model and hyperparameters
            model, hyperparameters = self._get_model_and_params(model_name, trial, experiment.search_space)
            
            # Train model
            start_time = datetime.utcnow()
            model.fit(X_train, y_train)
            training_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Evaluate model
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if experiment.optimization_objective == OptimizationObjective.ACCURACY:
                score = accuracy_score(y_test, y_pred)
            elif experiment.optimization_objective == OptimizationObjective.PRECISION:
                score = precision_score(y_test, y_pred, average='weighted')
            elif experiment.optimization_objective == OptimizationObjective.RECALL:
                score = recall_score(y_test, y_pred, average='weighted')
            elif experiment.optimization_objective == OptimizationObjective.F1_SCORE:
                score = f1_score(y_test, y_pred, average='weighted')
            else:
                score = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=self.config["cv_folds"], 
                scoring=experiment.optimization_objective.value.lower()
            )
            
            # Create model candidate
            model_candidate = ModelCandidate(
                model_id=f"model_{uuid.uuid4().hex[:8]}",
                experiment_id=experiment.experiment_id,
                model_type=experiment.model_type.value,
                model_name=model_name,
                hyperparameters=hyperparameters,
                performance_metrics={
                    "test_score": score,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std()
                },
                training_time=training_time,
                model_size=len(pickle.dumps(model)),
                cross_validation_scores=cv_scores.tolist(),
                created_at=datetime.utcnow()
            )
            
            # Store model candidate
            self.experiment_results[experiment.experiment_id].append(model_candidate)
            
            # Update experiment
            experiment.trials_completed += 1
            if experiment.best_score is None or score > experiment.best_score:
                experiment.best_score = score
                experiment.best_model = {
                    "model_id": model_candidate.model_id,
                    "model_name": model_name,
                    "hyperparameters": hyperparameters,
                    "performance_metrics": model_candidate.performance_metrics
                }
                
                # Save best model
                model_path = self.model_storage_path / f"{model_candidate.model_id}.pkl"
                joblib.dump(model, model_path)
            
            # Log to MLflow
            with mlflow.start_run():
                mlflow.log_params(hyperparameters)
                mlflow.log_metrics({
                    "test_score": score,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "training_time": training_time
                })
                mlflow.sklearn.log_model(model, "model")
            
            return score
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name=f"automl_{experiment.experiment_id}"
        )
        
        self.optuna_studies[experiment.experiment_id] = study
        
        # Run optimization
        study.optimize(objective, n_trials=experiment.total_trials)
        
        logger.info(f"Completed {experiment.trials_completed} trials for experiment {experiment.experiment_id}")
    
    async def _run_neural_network_optimization(
        self,
        experiment: AutoMLExperiment,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ):
        """Run optimization for neural network models"""
        
        def objective(trial):
            # Neural network hyperparameters
            n_layers = trial.suggest_int('n_layers', 2, 5)
            layers = []
            
            input_size = X_train.shape[1]
            
            for i in range(n_layers):
                if i == 0:
                    layer_size = trial.suggest_int(f'layer_{i}_size', 64, 512)
                    layers.append(nn.Linear(input_size, layer_size))
                else:
                    prev_size = layers[-1].out_features
                    layer_size = trial.suggest_int(f'layer_{i}_size', 32, 256)
                    layers.append(nn.Linear(prev_size, layer_size))
                
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(trial.suggest_float(f'dropout_{i}', 0.1, 0.5)))
            
            # Output layer
            output_size = len(np.unique(y_train))
            layers.append(nn.Linear(layers[-3].out_features, output_size))
            
            # Create model
            model = nn.Sequential(*layers)
            
            # Training hyperparameters
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            epochs = trial.suggest_int('epochs', 10, 100)
            
            # Train model
            start_time = datetime.utcnow()
            trained_model, history = self._train_pytorch_model(
                model, X_train, y_train, X_test, y_test,
                lr=lr, batch_size=batch_size, epochs=epochs
            )
            training_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Evaluate model
            score = history['val_accuracy'][-1]
            
            # Create model candidate
            model_candidate = ModelCandidate(
                model_id=f"model_{uuid.uuid4().hex[:8]}",
                experiment_id=experiment.experiment_id,
                model_type=experiment.model_type.value,
                model_name="neural_network",
                hyperparameters={
                    'n_layers': n_layers,
                    'lr': lr,
                    'batch_size': batch_size,
                    'epochs': epochs
                },
                performance_metrics={
                    "test_score": score,
                    "final_loss": history['val_loss'][-1]
                },
                training_time=training_time,
                model_size=sum(p.numel() for p in trained_model.parameters()),
                created_at=datetime.utcnow()
            )
            
            # Store model candidate
            self.experiment_results[experiment.experiment_id].append(model_candidate)
            
            # Update experiment
            experiment.trials_completed += 1
            if experiment.best_score is None or score > experiment.best_score:
                experiment.best_score = score
                experiment.best_model = {
                    "model_id": model_candidate.model_id,
                    "model_name": "neural_network",
                    "hyperparameters": model_candidate.hyperparameters,
                    "performance_metrics": model_candidate.performance_metrics
                }
                
                # Save best model
                model_path = self.model_storage_path / f"{model_candidate.model_id}.pth"
                torch.save(trained_model.state_dict(), model_path)
            
            # Log to MLflow
            with mlflow.start_run():
                mlflow.log_params(model_candidate.hyperparameters)
                mlflow.log_metrics(model_candidate.performance_metrics)
                mlflow.pytorch.log_model(trained_model, "model")
            
            return score
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name=f"automl_nn_{experiment.experiment_id}"
        )
        
        self.optuna_studies[experiment.experiment_id] = study
        
        # Run optimization
        study.optimize(objective, n_trials=experiment.total_trials)
        
        logger.info(f"Completed {experiment.trials_completed} trials for experiment {experiment.experiment_id}")
    
    # ===== SPARK INTEGRATION =====
    
    async def run_spark_automl_experiment(
        self,
        experiment_id: str,
        dataset_path: str,
        feature_cols: List[str],
        label_col: str
    ):
        """Run AutoML experiment using Spark for distributed processing"""
        
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return
        
        # Load data
        df = self.spark_session.read.parquet(dataset_path)
        
        # Prepare features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        
        # String indexer for labels
        indexer = StringIndexer(inputCol=label_col, outputCol="label")
        
        # Model
        rf = SparkRandomForest(featuresCol="features", labelCol="label")
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, indexer, rf])
        
        # Parameter grid
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [10, 20, 50]) \
            .addGrid(rf.maxDepth, [5, 10, 20]) \
            .addGrid(rf.minInstancesPerNode, [1, 5, 10]) \
            .build()
        
        # Cross validator
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=3
        )
        
        # Fit model
        cvModel = crossval.fit(df)
        
        # Get best model
        bestModel = cvModel.bestModel
        
        logger.info(f"Completed Spark AutoML experiment: {experiment_id}")
        
        return bestModel
    
    # ===== UTILITY METHODS =====
    
    def _get_default_search_space(self, model_type: ModelType) -> Dict[str, Any]:
        """Get default search space for model type"""
        
        if model_type == ModelType.CLASSIFICATION:
            return {
                "random_forest": {
                    "n_estimators": (10, 200),
                    "max_depth": (3, 20),
                    "min_samples_split": (2, 20),
                    "min_samples_leaf": (1, 10)
                },
                "gradient_boosting": {
                    "n_estimators": (50, 200),
                    "learning_rate": (0.01, 0.3),
                    "max_depth": (3, 10),
                    "subsample": (0.8, 1.0)
                },
                "logistic_regression": {
                    "C": (0.001, 100.0),
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "lbfgs"]
                },
                "svm": {
                    "C": (0.001, 100.0),
                    "kernel": ["rbf", "poly", "sigmoid"],
                    "gamma": (0.001, 1.0)
                }
            }
        
        return {}
    
    def _get_model_and_params(
        self,
        model_name: str,
        trial: optuna.Trial,
        search_space: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Get model instance and hyperparameters"""
        
        model_space = search_space.get(model_name, {})
        hyperparameters = {}
        
        if model_name == "random_forest":
            hyperparameters = {
                "n_estimators": trial.suggest_int("n_estimators", *model_space.get("n_estimators", (10, 200))),
                "max_depth": trial.suggest_int("max_depth", *model_space.get("max_depth", (3, 20))),
                "min_samples_split": trial.suggest_int("min_samples_split", *model_space.get("min_samples_split", (2, 20))),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", *model_space.get("min_samples_leaf", (1, 10))),
                "random_state": self.config["random_state"]
            }
            model = RandomForestClassifier(**hyperparameters)
        
        elif model_name == "gradient_boosting":
            hyperparameters = {
                "n_estimators": trial.suggest_int("n_estimators", *model_space.get("n_estimators", (50, 200))),
                "learning_rate": trial.suggest_float("learning_rate", *model_space.get("learning_rate", (0.01, 0.3))),
                "max_depth": trial.suggest_int("max_depth", *model_space.get("max_depth", (3, 10))),
                "subsample": trial.suggest_float("subsample", *model_space.get("subsample", (0.8, 1.0))),
                "random_state": self.config["random_state"]
            }
            model = GradientBoostingClassifier(**hyperparameters)
        
        elif model_name == "logistic_regression":
            hyperparameters = {
                "C": trial.suggest_float("C", *model_space.get("C", (0.001, 100.0)), log=True),
                "penalty": trial.suggest_categorical("penalty", model_space.get("penalty", ["l1", "l2"])),
                "solver": trial.suggest_categorical("solver", model_space.get("solver", ["liblinear", "lbfgs"])),
                "random_state": self.config["random_state"],
                "max_iter": 1000
            }
            model = LogisticRegression(**hyperparameters)
        
        elif model_name == "svm":
            hyperparameters = {
                "C": trial.suggest_float("C", *model_space.get("C", (0.001, 100.0)), log=True),
                "kernel": trial.suggest_categorical("kernel", model_space.get("kernel", ["rbf", "poly", "sigmoid"])),
                "gamma": trial.suggest_float("gamma", *model_space.get("gamma", (0.001, 1.0)), log=True),
                "random_state": self.config["random_state"]
            }
            model = SVC(**hyperparameters)
        
        elif model_name == "mlp":
            hidden_layer_sizes = tuple([
                trial.suggest_int(f"hidden_layer_{i}", 50, 200) 
                for i in range(trial.suggest_int("n_hidden_layers", 1, 3))
            ])
            hyperparameters = {
                "hidden_layer_sizes": hidden_layer_sizes,
                "learning_rate_init": trial.suggest_float("learning_rate_init", 0.001, 0.1, log=True),
                "alpha": trial.suggest_float("alpha", 0.0001, 0.1, log=True),
                "max_iter": 1000,
                "random_state": self.config["random_state"]
            }
            model = MLPClassifier(**hyperparameters)
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        return model, hyperparameters
    
    def _train_pytorch_model(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        lr: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Train PyTorch model"""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            train_accuracy = 100 * correct / total
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = 100 * (val_predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss.item())
            history['val_accuracy'].append(val_accuracy)
        
        return model, history
    
    async def _load_dataset(self, dataset_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset based on configuration"""
        
        # This would load data from various sources
        # For now, return dummy data
        X = np.random.rand(1000, 10)
        y = np.random.randint(0, 2, 1000)
        
        return X, y
    
    async def _perform_feature_engineering(
        self,
        X: np.ndarray,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Perform feature engineering"""
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled
    
    async def _generate_results_summary(self, experiment: AutoMLExperiment) -> Dict[str, Any]:
        """Generate results summary for experiment"""
        
        results = self.experiment_results[experiment.experiment_id]
        
        if not results:
            return {}
        
        # Sort by performance
        results_sorted = sorted(results, key=lambda x: x.performance_metrics.get("test_score", 0), reverse=True)
        
        summary = {
            "best_model": {
                "model_id": results_sorted[0].model_id,
                "model_name": results_sorted[0].model_name,
                "score": results_sorted[0].performance_metrics.get("test_score", 0),
                "hyperparameters": results_sorted[0].hyperparameters
            },
            "top_models": [
                {
                    "model_id": result.model_id,
                    "model_name": result.model_name,
                    "score": result.performance_metrics.get("test_score", 0)
                }
                for result in results_sorted[:5]
            ],
            "performance_distribution": {
                "mean_score": np.mean([r.performance_metrics.get("test_score", 0) for r in results]),
                "std_score": np.std([r.performance_metrics.get("test_score", 0) for r in results]),
                "min_score": min([r.performance_metrics.get("test_score", 0) for r in results]),
                "max_score": max([r.performance_metrics.get("test_score", 0) for r in results])
            },
            "training_time": {
                "total_time": sum([r.training_time for r in results]),
                "average_time": np.mean([r.training_time for r in results])
            }
        }
        
        return summary
    
    async def _initialize_spark_session(self):
        """Initialize Spark session"""
        
        try:
            self.spark_session = SparkSession.builder \
                .appName("AutoML_Service") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            
            logger.info("Spark session initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Spark session: {e}")
    
    async def _initialize_mlflow(self):
        """Initialize MLflow tracking"""
        
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(str(self.model_storage_path / "mlruns"))
            
            logger.info("MLflow tracking initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics for AutoML"""
        
        topics = [
            "q.ml.automl.experiment.started",
            "q.ml.automl.experiment.completed",
            "q.ml.automl.model.trained",
            "q.ml.automl.results.updated"
        ]
        
        logger.info("AutoML Pulsar topics configured")
    
    async def _load_experiments(self):
        """Load existing experiments from storage"""
        
        experiments_file = self.model_storage_path / "experiments.json"
        if experiments_file.exists():
            try:
                with open(experiments_file, 'r') as f:
                    experiments_data = json.load(f)
                
                for exp_data in experiments_data:
                    experiment = AutoMLExperiment(**exp_data)
                    self.active_experiments[experiment.experiment_id] = experiment
                
                logger.info(f"Loaded {len(self.active_experiments)} experiments")
            except Exception as e:
                logger.error(f"Failed to load experiments: {e}")
    
    async def _save_experiments(self):
        """Save experiments to storage"""
        
        experiments_file = self.model_storage_path / "experiments.json"
        try:
            experiments_data = [asdict(exp) for exp in self.active_experiments.values()]
            
            with open(experiments_file, 'w') as f:
                json.dump(experiments_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.active_experiments)} experiments")
        except Exception as e:
            logger.error(f"Failed to save experiments: {e}")
    
    # ===== BACKGROUND TASKS =====
    
    async def _experiment_monitor(self):
        """Monitor active experiments"""
        
        while True:
            try:
                current_time = datetime.utcnow()
                
                for experiment_id, experiment in list(self.active_experiments.items()):
                    if experiment.status == AutoMLStatus.RUNNING:
                        # Check for timeout
                        if experiment.started_at and \
                           (current_time - experiment.started_at).total_seconds() > (self.config["timeout_hours"] * 3600):
                            
                            experiment.status = AutoMLStatus.CANCELLED
                            experiment.completed_at = current_time
                            
                            logger.warning(f"Experiment {experiment_id} timed out")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in experiment monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _performance_tracking(self):
        """Track AutoML performance metrics"""
        
        while True:
            try:
                # Update metrics
                completed_experiments = [
                    exp for exp in self.active_experiments.values()
                    if exp.status == AutoMLStatus.COMPLETED
                ]
                
                if completed_experiments:
                    self.automl_metrics["experiments_completed"] = len(completed_experiments)
                    
                    # Calculate average experiment time
                    experiment_times = [
                        (exp.completed_at - exp.started_at).total_seconds()
                        for exp in completed_experiments
                        if exp.started_at and exp.completed_at
                    ]
                    
                    if experiment_times:
                        self.automl_metrics["average_experiment_time"] = np.mean(experiment_times)
                
                # Count total models trained
                total_models = sum(len(results) for results in self.experiment_results.values())
                self.automl_metrics["models_trained"] = total_models
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(300)
    
    # ===== PUBLIC API METHODS =====
    
    async def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an AutoML experiment"""
        
        if experiment_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[experiment_id]
        results = self.experiment_results[experiment_id]
        
        return {
            "experiment_id": experiment_id,
            "experiment_name": experiment.experiment_name,
            "status": experiment.status.value,
            "model_type": experiment.model_type.value,
            "optimization_objective": experiment.optimization_objective.value,
            "trials_completed": experiment.trials_completed,
            "total_trials": experiment.total_trials,
            "best_score": experiment.best_score,
            "best_model": experiment.best_model,
            "created_at": experiment.created_at,
            "started_at": experiment.started_at,
            "completed_at": experiment.completed_at,
            "results_count": len(results)
        }
    
    async def get_experiment_results(self, experiment_id: str) -> List[ModelCandidate]:
        """Get results for an experiment"""
        
        return self.experiment_results.get(experiment_id, [])
    
    async def get_model_leaderboard(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get model leaderboard"""
        
        all_results = []
        for results in self.experiment_results.values():
            all_results.extend(results)
        
        if model_type:
            all_results = [r for r in all_results if r.model_type == model_type]
        
        # Sort by performance
        all_results.sort(key=lambda x: x.performance_metrics.get("test_score", 0), reverse=True)
        
        return [
            {
                "model_id": result.model_id,
                "model_name": result.model_name,
                "model_type": result.model_type,
                "score": result.performance_metrics.get("test_score", 0),
                "training_time": result.training_time,
                "created_at": result.created_at
            }
            for result in all_results[:50]  # Top 50
        ]
    
    async def get_automl_metrics(self) -> Dict[str, Any]:
        """Get AutoML service metrics"""
        
        return {
            "service_metrics": self.automl_metrics,
            "active_experiments": len([exp for exp in self.active_experiments.values() if exp.status == AutoMLStatus.RUNNING]),
            "completed_experiments": len([exp for exp in self.active_experiments.values() if exp.status == AutoMLStatus.COMPLETED]),
            "total_models": sum(len(results) for results in self.experiment_results.values())
        }

# Global instance
automl_service = AutoMLService() 