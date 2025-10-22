#!/usr/bin/env python3
"""
Unified MLOps interface that bridges src/ (CLI modules) and mlops/ (Python API).
Allows users to choose between modular CLI approach or integrated Python API.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import the Python API
try:
    from mlops import train_model, predict_batch, Config
    MLOPS_API_AVAILABLE = True
except ImportError:
    MLOPS_API_AVAILABLE = False


class MLOpsRunner:
    """Unified runner for both src/ CLI and mlops/ Python API approaches."""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.src_dir = self.project_root / "src"
        
    def run_preprocessing(self, 
                         input_path: str,
                         output_path: str,
                         report_path: str = "reports/data_quality_report.json",
                         params_path: str = "params.yaml") -> int:
        """Run data preprocessing using src/data/preprocess.py"""
        
        cmd = [
            sys.executable, 
            str(self.src_dir / "data" / "preprocess.py"),
            "--inp", input_path,
            "--out", output_path,
            "--report", report_path,
            "--params", params_path
        ]
        
        print(f"🔄 Running preprocessing: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_feature_engineering(self,
                              input_path: str,
                              output_path: str,
                              params_path: str = "params.yaml") -> int:
        """Run feature engineering using src/data/make_features.py"""
        
        cmd = [
            sys.executable,
            str(self.src_dir / "data" / "make_features.py"),
            "--inp", input_path,
            "--out", output_path,
            "--params", params_path
        ]
        
        print(f"⚙️ Running feature engineering: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_training_cli(self,
                        data_path: str,
                        params_path: str = "params.yaml",
                        model_dir: str = "models",
                        metrics_path: str = "reports/metrics.json",
                        fig_cm_path: str = "reports/figures/confusion_matrix.png") -> int:
        """Run training using src/models/train.py"""
        
        cmd = [
            sys.executable,
            str(self.src_dir / "models" / "train.py"),
            "--data", data_path,
            "--params", params_path,
            "--model_dir", model_dir,
            "--metrics", metrics_path,
            "--fig_cm", fig_cm_path
        ]
        
        print(f"🤖 Running training (CLI): {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_training_api(self, params_path: str = "params.yaml", experiment_name: str = "obesity_classification") -> Dict[str, Any]:
        """Run training using mlops/ Python API"""
        
        if not MLOPS_API_AVAILABLE:
            raise ImportError("MLOps Python API not available. Install required dependencies.")
        
        print(f"🤖 Running training (Python API)...")
        return train_model(config_path=params_path, experiment_name=experiment_name)
    
    def run_evaluation(self,
                      data_path: str,
                      model_path: str = "models/mlflow_model",
                      eval_json: str = "reports/eval_metrics.json",
                      fig_cm: str = "reports/figures/confusion_matrix_eval.png",
                      fig_fi: str = "reports/figures/feature_importance.png") -> int:
        """Run evaluation using src/models/evaluate.py"""
        
        cmd = [
            sys.executable,
            str(self.src_dir / "models" / "evaluate.py"),
            "--data", data_path,
            "--model_path", model_path,
            "--eval_json", eval_json,
            "--fig_cm", fig_cm,
            "--fig_fi", fig_fi
        ]
        
        print(f"📊 Running evaluation: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_prediction_cli(self,
                          model_path: str = "models/mlflow_model",
                          features_csv: Optional[str] = None,
                          raw_csv: Optional[str] = None,
                          artifacts_dir: Optional[str] = None,
                          output_csv: str = "reports/predictions.csv") -> int:
        """Run predictions using src/models/predict.py"""
        
        cmd = [
            sys.executable,
            str(self.src_dir / "models" / "predict.py"),
            "--model_path", model_path,
            "--out_csv", output_csv
        ]
        
        if features_csv:
            cmd.extend(["--features_csv", features_csv])
        if raw_csv:
            cmd.extend(["--raw_csv", raw_csv])
        if artifacts_dir:
            cmd.extend(["--artifacts_dir", artifacts_dir])
        
        print(f"🔮 Running predictions (CLI): {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_prediction_api(self,
                          model_path: str,
                          data_path: str,
                          output_path: Optional[str] = None,
                          config_path: str = "params.yaml"):
        """Run predictions using mlops/ Python API"""
        
        if not MLOPS_API_AVAILABLE:
            raise ImportError("MLOps Python API not available. Install required dependencies.")
        
        print(f"🔮 Running predictions (Python API)...")
        return predict_batch(model_path=model_path, data_path=data_path, 
                           output_path=output_path, config_path=config_path)
    
    def run_complete_pipeline_cli(self, params_path: str = "params.yaml") -> bool:
        """Run complete pipeline using src/ CLI modules"""
        
        print("🚀 Starting complete MLOps pipeline (CLI mode)...")
        
        # Load config to get paths
        try:
            config = Config(params_path) if MLOPS_API_AVAILABLE else None
        except:
            config = None
        
        # Default paths (could be read from params.yaml if config available)
        raw_data = "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
        interim_data = "data/interim/obesity_clean.csv"
        processed_data = "data/processed/features.csv"
        
        steps = [
            ("Preprocessing", lambda: self.run_preprocessing(raw_data, interim_data, params_path=params_path)),
            ("Feature Engineering", lambda: self.run_feature_engineering(interim_data, processed_data, params_path=params_path)),
            ("Training", lambda: self.run_training_cli(processed_data, params_path=params_path)),
            ("Evaluation", lambda: self.run_evaluation(processed_data))
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*50}")
            print(f"Step: {step_name}")
            print(f"{'='*50}")
            
            try:
                result = step_func()
                if result != 0:
                    print(f"❌ {step_name} failed with exit code {result}")
                    return False
                print(f"✅ {step_name} completed successfully")
            except Exception as e:
                print(f"❌ {step_name} failed with error: {e}")
                return False
        
        print(f"\n{'='*50}")
        print("🎉 Complete pipeline executed successfully!")
        print(f"{'='*50}")
        return True
    
    def run_complete_pipeline_api(self, params_path: str = "params.yaml", experiment_name: str = "obesity_classification") -> Dict[str, Any]:
        """Run complete pipeline using mlops/ Python API"""
        
        if not MLOPS_API_AVAILABLE:
            raise ImportError("MLOps Python API not available. Install required dependencies.")
        
        print("🚀 Starting complete MLOps pipeline (API mode)...")
        return self.run_training_api(params_path, experiment_name)


def main():
    parser = argparse.ArgumentParser(description="Unified MLOps Interface - Choose between CLI and API approaches")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # CLI Pipeline commands
    cli_parser = subparsers.add_parser('cli', help='Run using src/ CLI modules (recommended for DVC)')
    cli_subparsers = cli_parser.add_subparsers(dest='cli_command')
    
    # CLI - Complete pipeline
    cli_complete = cli_subparsers.add_parser('pipeline', help='Run complete pipeline')
    cli_complete.add_argument('--params', default='params.yaml', help='Parameters file')
    
    # CLI - Individual steps
    cli_prep = cli_subparsers.add_parser('preprocess', help='Run preprocessing only')
    cli_prep.add_argument('--input', required=True, help='Input CSV path')
    cli_prep.add_argument('--output', required=True, help='Output CSV path') 
    cli_prep.add_argument('--params', default='params.yaml', help='Parameters file')
    
    cli_feat = cli_subparsers.add_parser('features', help='Run feature engineering only')
    cli_feat.add_argument('--input', required=True, help='Input CSV path')
    cli_feat.add_argument('--output', required=True, help='Output CSV path')
    cli_feat.add_argument('--params', default='params.yaml', help='Parameters file')
    
    cli_train = cli_subparsers.add_parser('train', help='Run training only')
    cli_train.add_argument('--data', required=True, help='Processed data CSV path')
    cli_train.add_argument('--params', default='params.yaml', help='Parameters file')
    
    cli_eval = cli_subparsers.add_parser('evaluate', help='Run evaluation only')
    cli_eval.add_argument('--data', required=True, help='Processed data CSV path')
    cli_eval.add_argument('--model', default='models/mlflow_model', help='Model path')
    
    cli_pred = cli_subparsers.add_parser('predict', help='Run predictions only')
    cli_pred.add_argument('--model', default='models/mlflow_model', help='Model path')
    cli_pred.add_argument('--features', help='Features CSV path')
    cli_pred.add_argument('--raw', help='Raw data CSV path')
    cli_pred.add_argument('--artifacts', help='Artifacts directory')
    cli_pred.add_argument('--output', default='reports/predictions.csv', help='Output CSV path')
    
    # API Pipeline commands  
    api_parser = subparsers.add_parser('api', help='Run using mlops/ Python API (recommended for interactive use)')
    api_subparsers = api_parser.add_subparsers(dest='api_command')
    
    # API - Complete pipeline
    api_complete = api_subparsers.add_parser('pipeline', help='Run complete pipeline')
    api_complete.add_argument('--params', default='params.yaml', help='Parameters file')
    api_complete.add_argument('--experiment', default='obesity_classification', help='MLflow experiment name')
    
    # API - Training
    api_train = api_subparsers.add_parser('train', help='Run training only')
    api_train.add_argument('--params', default='params.yaml', help='Parameters file')
    api_train.add_argument('--experiment', default='obesity_classification', help='MLflow experiment name')
    
    # API - Predictions
    api_pred = api_subparsers.add_parser('predict', help='Run predictions only') 
    api_pred.add_argument('--model', required=True, help='Model path')
    api_pred.add_argument('--data', required=True, help='Data CSV path')
    api_pred.add_argument('--output', help='Output CSV path')
    api_pred.add_argument('--params', default='params.yaml', help='Parameters file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    runner = MLOpsRunner()
    
    try:
        if args.command == 'cli':
            if args.cli_command == 'pipeline':
                success = runner.run_complete_pipeline_cli(args.params)
                sys.exit(0 if success else 1)
            elif args.cli_command == 'preprocess':
                result = runner.run_preprocessing(args.input, args.output, params_path=args.params)
                sys.exit(result)
            elif args.cli_command == 'features':
                result = runner.run_feature_engineering(args.input, args.output, params_path=args.params)
                sys.exit(result)
            elif args.cli_command == 'train':
                result = runner.run_training_cli(args.data, params_path=args.params)
                sys.exit(result)
            elif args.cli_command == 'evaluate':
                result = runner.run_evaluation(args.data, model_path=args.model)
                sys.exit(result)
            elif args.cli_command == 'predict':
                result = runner.run_prediction_cli(
                    model_path=args.model, features_csv=args.features, 
                    raw_csv=args.raw, artifacts_dir=args.artifacts, output_csv=args.output
                )
                sys.exit(result)
            else:
                cli_parser.print_help()
                
        elif args.command == 'api':
            if args.api_command == 'pipeline':
                results = runner.run_complete_pipeline_api(args.params, args.experiment)
                if results.get('success'):
                    print(f"✅ API Pipeline completed successfully!")
                    print(f"Accuracy: {results['test_metrics']['accuracy']:.3f}")
                    print(f"F1-macro: {results['test_metrics']['f1_macro']:.3f}")
                else:
                    print(f"❌ API Pipeline failed: {results.get('error', 'Unknown error')}")
                    sys.exit(1)
            elif args.api_command == 'train':
                results = runner.run_training_api(args.params, args.experiment)
                if results.get('success'):
                    print(f"✅ Training completed successfully!")
                    print(f"Model saved to: {results['model_path']}")
                else:
                    print(f"❌ Training failed: {results.get('error', 'Unknown error')}")
                    sys.exit(1)
            elif args.api_command == 'predict':
                predictions = runner.run_prediction_api(args.model, args.data, args.output, args.params)
                print(f"✅ Predictions completed: {len(predictions)} samples")
            else:
                api_parser.print_help()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()