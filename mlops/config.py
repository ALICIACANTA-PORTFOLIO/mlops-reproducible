"""
Configuration management for MLOps pipeline.
Centralized configuration handling with environment variables and YAML support.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for MLOps pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration with default values."""
        
        self.project_root = Path(__file__).parent.parent
        self.config_path = Path(config_path) if config_path else self.project_root / "params.yaml"
        
        # Default configuration
        self.defaults = {
            # Data configuration
            "data": {
                "raw_path": "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv",
                "processed_path": "data/processed/",
                "target_column": "NObeyesdad",
                "test_size": 0.2,
                "random_state": 42
            },
            
            # Model configuration  
            "model": {
                "name": "RandomForestClassifier",
                "parameters": {
                    "n_estimators": 200,
                    "max_depth": 15,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "random_state": 42,
                    "n_jobs": -1
                }
            },
            
            # Features configuration
            "features": {
                "selection_method": "mutual_info",
                "n_features": 10,
                "create_interactions": True,
                "apply_pca": False,
                "pca_components": 0.95
            },
            
            # Training configuration
            "training": {
                "cv_folds": 5,
                "scoring": "f1_macro",
                "hyperparameter_tuning": True
            },
            
            # MLflow configuration
            "mlflow": {
                "experiment_name": "obesity_classification",
                "model_name": "obesity_classifier",
                "tracking_uri": "sqlite:///mlruns.db"
            }
        }
        
        # Load configuration from file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with defaults fallback."""
        
        config = self.defaults.copy()
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config = self._deep_merge(config, file_config)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
        
        # Override with environment variables
        config = self._apply_env_overrides(config)
        
        return config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _apply_env_overrides(self, config: Dict) -> Dict:
        """Apply environment variable overrides."""
        
        # MLflow overrides
        if os.getenv('MLFLOW_TRACKING_URI'):
            config['mlflow']['tracking_uri'] = os.getenv('MLFLOW_TRACKING_URI')
        
        if os.getenv('MLFLOW_EXPERIMENT_NAME'):
            config['mlflow']['experiment_name'] = os.getenv('MLFLOW_EXPERIMENT_NAME')
        
        # Data path overrides
        if os.getenv('DATA_PATH'):
            config['data']['raw_path'] = os.getenv('DATA_PATH')
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save current configuration to YAML file."""
        
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data', {})
    
    @property 
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration.""" 
        return self.config.get('training', {})
    
    @property
    def mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return self.config.get('mlflow', {})