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
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import joblib
from collections import defaultdict
import os

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
from .model_registry_service import ModelRegistryService

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
    artifact_path: Optional[str] = None # Added field
    metadata: Optional[Dict[str, Any]] = None # Added field

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
                 spark_config: Optional[Dict[str, Any]] = None,
                 kg_client: Optional[KnowledgeGraphClient] = None): # Added kg_client parameter
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.spark_config = spark_config or {}
        self.spark_session = None
        
        # Active experiments
        self.active_experiments: Dict[str, AutoMLExperiment] = {}
        self.experiment_results: Dict[str, List[ModelCandidate]] = defaultdict(list)
        
        # Model registry integration
        self.model_registry_service = ModelRegistryService()
        
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
        
        # Knowledge Graph client
        self.kg_client = kg_client or KnowledgeGraphClient(base_url=os.getenv("KGQ_API_URL", "http://knowledgegraphq:8000")) # Initialize KG client

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

        # Initialize Knowledge Graph client
        # self.kg_client.initialize() is not awaited as there is no initialize method
        
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
        
        # Close Knowledge Graph client
        await self.kg_client.aclose() # Close KG client
        
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
        
        # Store experiment in Knowledge Graph
        await self._store_experiment_run_in_kg(experiment) # New: Store in KG

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
        """
        Run AutoML experiment
        """
        
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

            # Create lineage in Knowledge Graph for the experiment run
            # Assuming dataset_version_id and feature_ids can be extracted from dataset_config or inferred
            dataset_version_id = experiment.dataset_config.get("version_id") # Example: Extract from config
            feature_ids = experiment.dataset_config.get("feature_ids") # Example: Extract from config

            await self._create_experiment_lineage_in_kg(
                experiment_id=experiment.experiment_id,
                model_version_id=experiment.best_model.get("model_id") if experiment.best_model else None,
                dataset_version_id=dataset_version_id,
                feature_ids=feature_ids
            ) # New: Create lineage
            
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
        """
        Run optimization for traditional ML models
        """
        
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
                created_at=datetime.utcnow(),
                artifact_path=str(self.model_storage_path / f"{experiment.experiment_id}_{model_name}_{trial.number}.pkl"), # Populate artifact_path
                metadata={"trial_number": trial.number, "framework": "sklearn"}
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
                
                # Register model in registry
                await self._register_model_with_registry(
                    model_candidate, 
                    str(model_path), 
                    experiment.experiment_name,
                    "sklearn",
                    dataset_version_id=experiment.dataset_config.get("version_id"), # Pass dataset_version_id
                    feature_ids=experiment.dataset_config.get("feature_ids") # Pass feature_ids
                )
            
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
        """
        Run optimization for neural network models
        """
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train) # Assuming classification
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        class SimpleNN(nn.Module):
            def __init__(self, input_dim, n_layers, hidden_dim, output_dim):
                super(SimpleNN, self).__init__()
                layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
                for _ in range(n_layers - 1):
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(hidden_dim, output_dim))
                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        def objective(trial):
            # Neural network hyperparameters
            n_layers = trial.suggest_int('n_layers', 1, 3)
            hidden_dim = trial.suggest_int('hidden_dim', 32, 128, step=32)
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            epochs = trial.suggest_int('epochs', 5, 20)

            model = SimpleNN(X_train.shape[1], n_layers, hidden_dim, len(np.unique(y_train)))
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # Create DataLoader
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Training loop
            history = defaultdict(list)
            start_time = datetime.utcnow()
            for epoch in range(epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                # Evaluate on test set
                model.eval()
                with torch.no_grad():
                    outputs = model(X_test_tensor)
                    loss = criterion(outputs, y_test_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
                    history['val_loss'].append(loss.item())
                    history['val_accuracy'].append(accuracy)
            training_time = (datetime.utcnow() - start_time).total_seconds()

            # Evaluate model
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_tensor)
                _, predicted = torch.max(outputs.data, 1)
                score = accuracy_score(y_test_tensor.numpy(), predicted.numpy())

            # Create model candidate
            model_candidate = ModelCandidate(
                model_id=f"model_{uuid.uuid4().hex[:8]}",
                experiment_id=experiment.experiment_id,
                model_type=experiment.model_type.value,
                model_name="neural_network",
                hyperparameters={
                    'n_layers': n_layers,
                    'hidden_dim': hidden_dim,
                    'lr': lr,
                    'batch_size': batch_size,
                    'epochs': epochs
                },
                performance_metrics={
                    "test_score": score,
                    "final_loss": history['val_loss'][-1] if history['val_loss'] else None
                },
                training_time=training_time,
                model_size=sum(p.numel() for p in model.parameters()),
                created_at=datetime.utcnow(),
                artifact_path=str(self.model_storage_path / f"{experiment.experiment_id}_neural_network_{trial.number}.pth"), # Populate artifact_path
                metadata={"trial_number": trial.number, "framework": "pytorch"}
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
                torch.save(model.state_dict(), model_path)
                
                # Register model in registry
                await self._register_model_with_registry(
                    model_candidate, 
                    str(model_path), 
                    experiment.experiment_name,
                    "pytorch",
                    dataset_version_id=experiment.dataset_config.get("version_id"), # Pass dataset_version_id
                    feature_ids=experiment.dataset_config.get("feature_ids") # Pass feature_ids
                )
            
            # Log to MLflow
            with mlflow.start_run():
                mlflow.log_params(model_candidate.hyperparameters)
                mlflow.log_metrics(model_candidate.performance_metrics)
                mlflow.pytorch.log_model(model, "model")
            
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

    async def _get_default_search_space(self, model_type: ModelType) -> Dict[str, Any]:
        """Get default hyperparameter search space for a model type"""
        if model_type == ModelType.CLASSIFICATION:
            return {
                "random_forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_leaf": [1, 2, 4]
                },
                "gradient_boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                },
                "logistic_regression": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["liblinear", "lbfgs"]
                },
                "svm": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf"]
                },
                "mlp": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                    "activation": ["relu", "tanh"],
                    "alpha": [0.0001, 0.001, 0.01]
                }
            }
        elif model_type == ModelType.REGRESSION:
            return {
                "linear_regression": {},
                "random_forest_regressor": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None]
                }
            }
        elif model_type == ModelType.NEURAL_NETWORK:
            return {
                "n_layers": [1, 2, 3],
                "hidden_dim": [32, 64, 128],
                "lr": [1e-5, 1e-4, 1e-3, 1e-2],
                "batch_size": [32, 64, 128],
                "epochs": [5, 10, 15, 20]
            }
        return {}

    def _get_model_and_params(self, model_name: str, trial: optuna.Trial, search_space: Dict[str, Any]):
        """Get model instance and hyperparameters from Optuna trial"""
        hyperparameters = {}
        model = None

        if model_name == 'random_forest':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 5, 20) if trial.suggest_categorical('max_depth_none', [True, False]) else None
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=self.config["random_state"])
            hyperparameters = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf
            }
        elif model_name == 'gradient_boosting':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
            max_depth = trial.suggest_int('max_depth', 3, 7)
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=self.config["random_state"])
            hyperparameters = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth
            }
        elif model_name == 'logistic_regression':
            C = trial.suggest_loguniform('C', 1e-2, 1e2)
            solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
            model = LogisticRegression(C=C, solver=solver, random_state=self.config["random_state"], max_iter=1000)
            hyperparameters = {
                'C': C,
                'solver': solver
            }
        elif model_name == 'svm':
            C = trial.suggest_loguniform('C', 1e-2, 1e2)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
            model = SVC(C=C, kernel=kernel, random_state=self.config["random_state"], probability=True)
            hyperparameters = {
                'C': C,
                'kernel': kernel
            }
        elif model_name == 'mlp':
            hidden_layer_sizes_choice = trial.suggest_categorical('hidden_layer_sizes', ['(50,)', '(100,)', '(50, 50)'])
            hidden_layer_sizes = eval(hidden_layer_sizes_choice)
            activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
            alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-2)
            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha, random_state=self.config["random_state"], max_iter=1000)
            hyperparameters = {
                'hidden_layer_sizes': hidden_layer_sizes_choice,
                'activation': activation,
                'alpha': alpha
            }
        # Add other model types as needed
        return model, hyperparameters

    async def _load_dataset(self, dataset_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess dataset"""
        data_path = dataset_config.get("data_path")
        feature_columns = dataset_config.get("feature_columns", [])
        target_column = dataset_config.get("target_column")
        
        if not data_path or not target_column:
            raise ValueError("Dataset config must contain 'data_path' and 'target_column'")
        
        # Load data using Spark if available, otherwise Pandas
        if self.spark_session:
            df = self.spark_session.read.parquet(data_path) if data_path.endswith(".parquet") else self.spark_session.read.csv(data_path, header=True, inferSchema=True)
            df_pd = df.toPandas()
        else:
            if data_path.endswith(".csv"):
                df_pd = pd.read_csv(data_path)
            elif data_path.endswith(".parquet"):
                df_pd = pd.read_parquet(data_path)
            elif data_path.endswith(".json"):
                df_pd = pd.read_json(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        
        if feature_columns:
            X = df_pd[feature_columns].values
        else:
            X = df_pd.drop(columns=[target_column]).values

        y = df_pd[target_column].values
        
        # Impute missing values (simple imputation for demonstration)
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        return X, y

    async def _perform_feature_engineering(self, X: np.ndarray, training_config: Dict[str, Any]) -> np.ndarray:
        """Perform feature engineering based on training config"""
        
        pipeline_steps = []
        
        # Scaling
        if training_config.get("scaling") == "standard":
            pipeline_steps.append(("scaler", StandardScaler()))
        
        # PCA
        if training_config.get("pca_components", 0) > 0:
            pipeline_steps.append(("pca", PCA(n_components=training_config["pca_components"])))
        
        if pipeline_steps:
            pipeline = Pipeline(pipeline_steps)
            X_processed = pipeline.fit_transform(X)
        else:
            X_processed = X
            
        return X_processed

    async def _generate_results_summary(self, experiment: AutoMLExperiment) -> Dict[str, Any]:
        """Generate a summary of experiment results"""
        summary = {
            "experiment_id": experiment.experiment_id,
            "experiment_name": experiment.experiment_name,
            "status": experiment.status.value,
            "best_score": experiment.best_score,
            "best_model_id": experiment.best_model.get("model_id") if experiment.best_model else None,
            "trials_completed": experiment.trials_completed,
            "total_trials": experiment.total_trials,
            "start_time": experiment.started_at.isoformat() if experiment.started_at else None,
            "end_time": experiment.completed_at.isoformat() if experiment.completed_at else None,
            "duration_seconds": (experiment.completed_at - experiment.started_at).total_seconds() if experiment.started_at and experiment.completed_at else 0,
            "model_type": experiment.model_type.value,
            "optimization_objective": experiment.optimization_objective.value,
            "dataset_config": experiment.dataset_config,
            "training_config": experiment.training_config
        }

        # Add top N models to summary
        top_n = sorted(self.experiment_results[experiment.experiment_id],
                       key=lambda x: x.performance_metrics.get("test_score", 0), reverse=True)[:5]
        summary["top_models"] = [
            {
                "model_id": m.model_id,
                "model_name": m.model_name,
                "performance": m.performance_metrics.get("test_score", 0),
                "hyperparameters": m.hyperparameters
            } for m in top_n
        ]
        
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
        
        experiments_file = self.model_storage_path / "automl_experiments.json"
        if experiments_file.exists():
            try:
                with open(experiments_file, 'r') as f:
                    experiments_data = json.load(f)
                
                for exp_data in experiments_data:
                    # Convert string enums back to Enum objects
                    exp_data["model_type"] = ModelType(exp_data["model_type"])
                    exp_data["optimization_objective"] = OptimizationObjective(exp_data["optimization_objective"])
                    exp_data["status"] = AutoMLStatus(exp_data["status"])
                    
                    # Convert datetime strings to datetime objects
                    for k in ["created_at", "started_at", "completed_at"]:
                        if exp_data.get(k):
                            exp_data[k] = datetime.fromisoformat(exp_data[k])
                            
                    experiment = AutoMLExperiment(**exp_data)
                    self.active_experiments[experiment.experiment_id] = experiment
                
                logger.info(f"Loaded {len(self.active_experiments)} AutoML experiments")
            except Exception as e:
                logger.error(f"Failed to load AutoML experiments: {e}")
    
    async def _save_experiments(self):
        """Save experiments to storage"""
        
        experiments_file = self.model_storage_path / "automl_experiments.json"
        try:
            experiments_data = [
                asdict(exp) for exp in self.active_experiments.values()
            ]
            
            with open(experiments_file, 'w') as f:
                json.dump(experiments_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.active_experiments)} AutoML experiments")
        except Exception as e:
            logger.error(f"Failed to save AutoML experiments: {e}")
    
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
    
    async def _register_model_with_registry(
        self,
        model_candidate: ModelCandidate,
        artifact_path: str,
        experiment_name: str,
        model_framework: str,
        dataset_version_id: Optional[str] = None,
        feature_ids: Optional[List[str]] = None
    ):
        """
        Registers a model candidate with the central model registry service.
        """
        try:
            # Generate a consistent version_id for the model within the registry
            # This can be based on model_id and a timestamp
            version_id = f"{model_candidate.model_id}_{int(model_candidate.created_at.timestamp())}"

            await self.model_registry_service.register_model_version(
                model_name=f"{experiment_name}_{model_candidate.model_name}",
                version_id=version_id,
                artifact_path=artifact_path,
                metadata={
                    "experiment_id": model_candidate.experiment_id,
                    "model_type": model_candidate.model_type,
                    "model_framework": model_framework,
                    "hyperparameters": model_candidate.hyperparameters,
                    "performance_metrics": model_candidate.performance_metrics,
                    "training_time": model_candidate.training_time,
                    "model_size": model_candidate.model_size,
                    "cross_validation_scores": model_candidate.cross_validation_scores
                },
                is_active=False, # AutoML typically registers, deployment service activates
                dataset_version_id=dataset_version_id, # Pass dataset_version_id
                feature_ids=feature_ids # Pass feature_ids
            )
            
            # Publish model registration event
            await shared_pulsar_client.publish(
                "q.ml.automl.model.registered",
                {
                    "model_id": model_candidate.model_id,
                    "model_name": f"{experiment_name}_{model_candidate.model_name}",
                    "version_id": version_id,
                    "artifact_path": artifact_path,
                    "performance_score": model_candidate.performance_metrics.get("test_score", 0),
                    "experiment_id": model_candidate.experiment_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to register model {model_candidate.model_id}: {e}")

    # ===== KNOWLEDGE GRAPH INTEGRATION =====

    async def _store_experiment_run_in_kg(self, experiment: AutoMLExperiment):
        """Stores AutoML experiment run as a vertex in the Knowledge Graph."""
        try:
            vertex_data = {
                "experiment_id": experiment.experiment_id,
                "experiment_name": experiment.experiment_name,
                "model_type": experiment.model_type.value,
                "optimization_objective": experiment.optimization_objective.value,
                "dataset_config": experiment.dataset_config,
                "training_config": experiment.training_config,
                "search_space": experiment.search_space,
                "status": experiment.status.value,
                "created_at": experiment.created_at.isoformat(),
                "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
                "completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None,
                "best_score": experiment.best_score,
                "trials_completed": experiment.trials_completed,
                "total_trials": experiment.total_trials,
                "results_summary": experiment.results_summary
            }
            await self.kg_client.add_vertex(
                "ExperimentRun",
                experiment.experiment_id,
                vertex_data
            )
            logger.info(f"Stored ExperimentRun {experiment.experiment_id} in KG.")
        except Exception as e:
            logger.error(f"Failed to store ExperimentRun in KG: {e}", exc_info=True)

    async def _create_experiment_lineage_in_kg(
        self,
        experiment_id: str,
        model_version_id: Optional[str] = None,
        dataset_version_id: Optional[str] = None,
        feature_ids: Optional[List[str]] = None
    ):
        """Creates lineage relationships for an AutoML experiment run in the Knowledge Graph."""
        try:
            if model_version_id:
                await self.kg_client.add_edge(
                    "produced_model",
                    experiment_id,
                    model_version_id,
                    {"relationship": "produced_model"}
                )
                logger.info(f"Created 'produced_model' edge from {experiment_id} to {model_version_id}.")

            if dataset_version_id:
                await self.kg_client.add_edge(
                    "used_dataset",
                    experiment_id,
                    dataset_version_id,
                    {"relationship": "used_dataset"}
                )
                logger.info(f"Created 'used_dataset' edge from {experiment_id} to {dataset_version_id}.")

            if feature_ids:
                for feature_id in feature_ids:
                    await self.kg_client.add_edge(
                        "used_feature",
                        experiment_id,
                        feature_id,
                        {"relationship": "used_feature"}
                    )
                logger.info(f"Created 'used_feature' edge from {experiment_id} to {feature_id}.")

        except Exception as e:
            logger.error(f"Failed to create experiment lineage in KG: {e}", exc_info=True)

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

    async def get_automl_results(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get AutoML experiment results"""
        
        results = self.experiment_results.get(experiment_id, [])
        return [asdict(r) for r in results]

    async def get_model_leaderboard(self, model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """Get model leaderboard across all experiments"""
        
        all_models = []
        for experiment_id in self.experiment_results:
            all_models.extend(self.experiment_results[experiment_id])
        
        if model_type:
            all_models = [m for m in all_models if m.model_type == model_type.value]

        # Sort by best performance metric (e.g., test_score)
        leaderboard = sorted(all_models, key=lambda x: x.performance_metrics.get("test_score", float('-inf')), reverse=True)
        
        return [asdict(m) for m in leaderboard[:100]]

# Global instance
automl_service = AutoMLService() 