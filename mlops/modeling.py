"""
Model training and evaluation module for obesity classification.
Handles model training, hyperparameter optimization, and evaluation.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Professional model trainer for obesity classification."""
    
    def __init__(self, config: dict):
        """Initialize model trainer with configuration."""
        
        self.config = config
        self.model = None
        self.best_params = None
        self.cv_results = None
        
    def create_model(self, **params) -> RandomForestClassifier:
        """Create RandomForest model with given parameters."""
        
        # Default parameters from config
        default_params = self.config.get('parameters', {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        })
        
        # Override with provided parameters
        model_params = {**default_params, **params}
        
        logger.info(f"Creating RandomForest model with parameters: {model_params}")
        
        return RandomForestClassifier(**model_params)
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """Optimize hyperparameters using grid search."""
        
        logger.info("Starting hyperparameter optimization...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Base model
        base_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        cv_folds = self.config.get('cv_folds', 5)
        scoring = self.config.get('scoring', 'f1_macro')
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Store results
        self.best_params = grid_search.best_params_
        self.cv_results = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Hyperparameter optimization completed:")
        logger.info(f"  Best score: {grid_search.best_score_:.4f}")
        logger.info(f"  Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_, self.best_params
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   optimize: bool = True) -> RandomForestClassifier:
        """Train the model with optional hyperparameter optimization."""
        
        if optimize and self.config.get('hyperparameter_tuning', True):
            # Use hyperparameter optimization
            self.model, self.best_params = self.optimize_hyperparameters(X_train, y_train)
        else:
            # Use default/configured parameters
            self.model = self.create_model()
            logger.info("Training model with default parameters...")
            self.model.fit(X_train, y_train)
        
        logger.info("Model training completed successfully")
        
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate trained model on test data."""
        
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Detailed classification report
        class_report = classification_report(
            y_test, y_pred, 
            output_dict=True,
            zero_division=0
        )
        
        evaluation_results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': class_report
        }
        
        # Log results
        logger.info(f"Model evaluation results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1-macro: {f1_macro:.4f}")
        logger.info(f"  F1-weighted: {f1_weighted:.4f}")
        
        return evaluation_results
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on the model."""
        
        if self.model is None:
            raise ValueError("Model must be created before cross-validation")
        
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Define scoring metrics
        scoring_metrics = ['accuracy', 'f1_macro', 'f1_weighted']
        
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(
                self.model, X, y,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring=metric,
                n_jobs=-1
            )
            
            cv_results[metric] = {
                'scores': scores.tolist(),
                'mean': scores.mean(),
                'std': scores.std()
            }
            
            logger.info(f"  {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def get_feature_importance(self, feature_names: list) -> list:
        """Get feature importance from trained model."""
        
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # Create feature importance list
            feature_importance = [
                {'feature': name, 'importance': float(importance)}
                for name, importance in zip(feature_names, importances)
            ]
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            
            logger.info("Top 10 most important features:")
            for i, item in enumerate(feature_importance[:10]):
                logger.info(f"  {i+1}. {item['feature']}: {item['importance']:.4f}")
            
            return feature_importance
        else:
            logger.warning("Model does not support feature importance")
            return []
    
    def save_model(self, model_path: str = "models/model.pkl") -> None:
        """Save trained model to file."""
        
        if self.model is None:
            raise ValueError("No model to save")
        
        from pathlib import Path
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save additional info
        model_info = {
            'best_params': self.best_params,
            'cv_results': self.cv_results,
            'model_type': type(self.model).__name__
        }
        
        info_path = model_path.parent / "model_info.pkl"
        joblib.dump(model_info, info_path)
        
        logger.info(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: str = "models/model.pkl") -> RandomForestClassifier:
        """Load trained model from file."""
        
        from pathlib import Path
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        self.model = joblib.load(model_path)
        
        # Load additional info if available
        info_path = model_path.parent / "model_info.pkl"
        if info_path.exists():
            model_info = joblib.load(info_path)
            self.best_params = model_info.get('best_params')
            self.cv_results = model_info.get('cv_results')
        
        logger.info(f"Model loaded from: {model_path}")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained model."""
        
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities using trained model."""
        
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("Model does not support probability predictions")