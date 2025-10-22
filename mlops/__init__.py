"""
MLOps module for obesity classification pipeline.
Professional implementation following software engineering best practices.
"""

__version__ = "1.0.0"
__author__ = "Alicia Cantarero"

# Main MLOps components
from .config import Config
from .dataset import DataProcessor
from .features import FeatureEngineer
from .modeling import ModelTrainer
from .train import train_model, predict_batch

__all__ = [
    "Config",
    "DataProcessor", 
    "FeatureEngineer",
    "ModelTrainer",
    "train_model",
    "predict_batch"
]