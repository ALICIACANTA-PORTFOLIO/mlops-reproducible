"""
Main training pipeline for obesity classification MLOps project.
Orchestrates the complete machine learning workflow.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import time

from .config import Config
from .dataset import DataProcessor
from .features import FeatureEngineer
from .modeling import ModelTrainer


def setup_logging(config: Config) -> None:
    """Setup logging configuration."""
    
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format', 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('mlops.log')
        ]
    )


def train_model(config_path: str = "params.yaml", experiment_name: str = "obesity_classification") -> Dict[str, Any]:
    """
    Main training function that orchestrates the complete ML workflow.
    
    Args:
        config_path: Path to configuration file
        experiment_name: Name for MLflow experiment
        
    Returns:
        Dictionary with training results and metrics
    """
    
    # Load configuration
    config = Config(config_path)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting obesity classification training pipeline...")
    logger.info(f"Configuration loaded from: {config_path}")
    
    # Initialize results dictionary
    results = {
        'experiment_name': experiment_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config_path': config_path
    }
    
    try:
        # Initialize MLflow (if configured)
        mlflow_tracking = config.get('mlflow.tracking', True)
        if mlflow_tracking:
            try:
                import mlflow
                import mlflow.sklearn
                
                # Set experiment
                mlflow.set_experiment(experiment_name)
                
                # Start MLflow run
                with mlflow.start_run():
                    results = _execute_training_pipeline(config, logger, results, mlflow_enabled=True)
                    
            except ImportError:
                logger.warning("MLflow not available, continuing without experiment tracking")
                results = _execute_training_pipeline(config, logger, results, mlflow_enabled=False)
        else:
            results = _execute_training_pipeline(config, logger, results, mlflow_enabled=False)
            
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        results['error'] = str(e)
        results['success'] = False
        raise
    
    return results


def _execute_training_pipeline(config: Config, logger: logging.Logger, 
                             results: Dict[str, Any], mlflow_enabled: bool = False) -> Dict[str, Any]:
    """Execute the complete training pipeline."""
    
    # 1. Data Processing
    logger.info("Step 1: Loading and processing data...")
    data_processor = DataProcessor(config.config)
    
    # Load data
    data_path = config.get('data.raw_path', 'data/raw/ObesityDataSet_raw_and_data_sinthetic.csv')
    df = data_processor.load_data(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Validate data
    validation_results = data_processor.validate_data(df)
    results['data_validation'] = validation_results
    
    # Preprocess data
    df_processed = data_processor.preprocess_data(df)
    logger.info(f"Processed dataset shape: {df_processed.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = data_processor.split_data(
        df_processed, 
        target_col=config.get('data.target_column', 'NObeyesdad'),
        test_size=config.get('data.test_size', 0.2)
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 2. Feature Engineering
    logger.info("Step 2: Feature engineering...")
    feature_engineer = FeatureEngineer(config.config)
    
    # Create features
    X_train_features = feature_engineer.create_features(X_train)
    X_test_features = feature_engineer.create_features(X_test)
    
    # Feature selection
    X_train_selected, selected_features = feature_engineer.select_features(
        X_train_features, y_train, 
        method=config.get('features.selection_method', 'mutual_info')
    )
    X_test_selected = feature_engineer.apply_feature_selection(X_test_features)
    
    logger.info(f"Selected {len(selected_features)} features: {selected_features}")
    results['selected_features'] = selected_features
    
    # 3. Model Training
    logger.info("Step 3: Training model...")
    model_trainer = ModelTrainer(config.config)
    
    # Train model
    model = model_trainer.train_model(
        X_train_selected, y_train,
        optimize=config.get('model.hyperparameter_tuning', True)
    )
    
    # 4. Model Evaluation
    logger.info("Step 4: Evaluating model...")
    
    # Test set evaluation
    evaluation_results = model_trainer.evaluate_model(X_test_selected, y_test)
    results['test_metrics'] = evaluation_results
    
    # Cross-validation
    cv_results = model_trainer.cross_validate(
        X_train_selected, y_train,
        cv_folds=config.get('model.cv_folds', 5)
    )
    results['cv_metrics'] = cv_results
    
    # Feature importance
    feature_importance = model_trainer.get_feature_importance(selected_features)
    results['feature_importance'] = feature_importance
    
    # 5. Save Model and Results
    logger.info("Step 5: Saving model and results...")
    
    # Save model
    model_path = config.get('model.save_path', 'models/model.pkl')
    model_trainer.save_model(model_path)
    results['model_path'] = model_path
    
    # Save results
    results_path = config.get('results.save_path', 'reports/training_results.json')
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    results['results_path'] = results_path
    
    # 6. MLflow Logging (if enabled)
    if mlflow_enabled:
        _log_to_mlflow(config, results, model_trainer, selected_features)
    
    # Summary
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
    logger.info(f"Test F1-macro: {evaluation_results['f1_macro']:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Results saved to: {results_path}")
    
    results['success'] = True
    return results


def _log_to_mlflow(config: Config, results: Dict[str, Any], 
                  model_trainer: ModelTrainer, selected_features: list) -> None:
    """Log results to MLflow."""
    
    import mlflow
    import mlflow.sklearn
    
    # Log parameters
    mlflow.log_params(config.get('model.parameters', {}))
    mlflow.log_param("n_features", len(selected_features))
    mlflow.log_param("feature_selection_method", config.get('features.selection_method', 'mutual_info'))
    
    # Log metrics
    test_metrics = results['test_metrics']
    mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
    mlflow.log_metric("test_f1_macro", test_metrics['f1_macro'])
    mlflow.log_metric("test_f1_weighted", test_metrics['f1_weighted'])
    
    # Log CV metrics
    cv_metrics = results['cv_metrics']
    mlflow.log_metric("cv_accuracy_mean", cv_metrics['accuracy']['mean'])
    mlflow.log_metric("cv_accuracy_std", cv_metrics['accuracy']['std'])
    mlflow.log_metric("cv_f1_macro_mean", cv_metrics['f1_macro']['mean'])
    mlflow.log_metric("cv_f1_macro_std", cv_metrics['f1_macro']['std'])
    
    # Log model
    mlflow.sklearn.log_model(
        model_trainer.model,
        "model",
        registered_model_name="obesity_classifier"
    )
    
    # Log artifacts
    if results.get('results_path'):
        mlflow.log_artifact(results['results_path'])


def predict_batch(model_path: str, data_path: str, output_path: str = None, 
                 config_path: str = "params.yaml") -> np.ndarray:
    """
    Make batch predictions on new data.
    
    Args:
        model_path: Path to saved model
        data_path: Path to input data
        output_path: Path to save predictions (optional)
        config_path: Path to configuration file
        
    Returns:
        Predictions array
    """
    
    # Load configuration
    config = Config(config_path)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Making batch predictions...")
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {data_path}")
    
    # Load model
    model_trainer = ModelTrainer(config.config)
    model_trainer.load_model(model_path)
    
    # Process data
    data_processor = DataProcessor(config.config)
    df = data_processor.load_data(data_path)
    df_processed = data_processor.preprocess_data(df)
    
    # Feature engineering
    feature_engineer = FeatureEngineer(config.config)
    X_features = feature_engineer.create_features(df_processed)
    X_selected = feature_engineer.apply_feature_selection(X_features)
    
    # Make predictions
    predictions = model_trainer.predict(X_selected)
    
    # Save predictions if output path provided
    if output_path:
        import pandas as pd
        
        pred_df = pd.DataFrame({
            'prediction': predictions,
            'prediction_proba_max': np.max(model_trainer.predict_proba(X_selected), axis=1)
        })
        
        pred_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to: {output_path}")
    
    logger.info(f"Made predictions for {len(predictions)} samples")
    
    return predictions


if __name__ == "__main__":
    """Command line interface for training."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Obesity Classification Training Pipeline")
    parser.add_argument("--config", default="params.yaml", help="Path to configuration file")
    parser.add_argument("--experiment", default="obesity_classification", help="MLflow experiment name")
    parser.add_argument("--predict", action="store_true", help="Run prediction mode")
    parser.add_argument("--model", help="Path to saved model (for prediction)")
    parser.add_argument("--data", help="Path to input data (for prediction)")
    parser.add_argument("--output", help="Path to save predictions")
    
    args = parser.parse_args()
    
    if args.predict:
        if not args.model or not args.data:
            raise ValueError("Model and data paths required for prediction")
        
        predictions = predict_batch(
            model_path=args.model,
            data_path=args.data,
            output_path=args.output,
            config_path=args.config
        )
        
        print(f"Predictions completed: {len(predictions)} samples")
    else:
        results = train_model(
            config_path=args.config,
            experiment_name=args.experiment
        )
        
        if results['success']:
            print("Training completed successfully!")
            print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
            print(f"Model saved to: {results['model_path']}")
        else:
            print(f"Training failed: {results.get('error', 'Unknown error')}")