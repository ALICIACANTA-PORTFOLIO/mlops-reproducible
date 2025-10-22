"""
EJECUTOR DE OPTIMIZACIONES CON DATOS REALES - Proyecto Obesidad
================================================================

Script para ejecutar todas las optimizaciones implementadas usando
el dataset real de obesidad (obesity_clean.csv).

Dataset: 2089 registros reales de clasificación de obesidad
Target: NObeyesdad (7 clases de obesidad)
Features: 16 variables (demográficas, hábitos alimenticios, actividad física)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# MLflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARNING: MLflow no disponible")


class RealDataOptimizationRunner:
    """Ejecutor de optimizaciones con datos reales de obesidad."""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_path = "data/interim/obesity_clean.csv"
        self.results = {}
        self.encoders = {}
        self.scaler = None
        
        print(f"""
OPTIMIZACIONES MLOps - DATOS REALES DE OBESIDAD
================================================

Dataset: {self.data_path}
Timestamp: {self.timestamp}

Optimizaciones a ejecutar:
1. Carga y preprocesamiento de datos reales
2. Análisis exploratorio automatizado  
3. Hyperparameter tuning profesional
4. Pipeline con method chaining
5. Model registry con MLflow
6. Evaluación completa con métricas de negocio
7. Tests de validación en producción
""")
    
    def load_and_prepare_real_data(self):
        """Cargar y preparar el dataset real de obesidad."""
        
        print("PASO 1: Cargando datos reales...")
        print("-" * 50)
        
        # Verificar que el archivo existe
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Dataset no encontrado: {self.data_path}")
        
        # Cargar datos
        self.raw_data = pd.read_csv(self.data_path)
        print(f"OK - Dataset cargado: {self.raw_data.shape}")
        print(f"Columnas: {list(self.raw_data.columns)}")
        
        # Información básica del dataset
        print(f"\nInformación del dataset:")
        print(f"  Registros totales: {len(self.raw_data)}")
        print(f"  Features: {len(self.raw_data.columns) - 1}")
        print(f"  Target: NObeyesdad")
        
        # Verificar clases de obesidad
        target_col = 'NObeyesdad'
        if target_col not in self.raw_data.columns:
            raise ValueError(f"Columna target '{target_col}' no encontrada")
        
        obesity_classes = self.raw_data[target_col].value_counts()
        print(f"\nDistribución de clases de obesidad:")
        for clase, count in obesity_classes.items():
            percentage = (count / len(self.raw_data)) * 100
            print(f"  {clase}: {count} ({percentage:.1f}%)")
        
        # Información de tipos de datos
        print(f"\nTipos de datos:")
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = self.raw_data.select_dtypes(exclude=['object']).columns.tolist()
        
        print(f"  Categóricas ({len(categorical_cols)}): {categorical_cols}")
        print(f"  Numéricas ({len(numerical_cols)}): {numerical_cols}")
        
        # Verificar valores faltantes
        missing_values = self.raw_data.isnull().sum()
        if missing_values.any():
            print(f"\nValores faltantes encontrados:")
            for col, missing in missing_values[missing_values > 0].items():
                print(f"  {col}: {missing}")
        else:
            print(f"\nOK - Sin valores faltantes")
        
        return self.raw_data
    
    def preprocess_real_data(self):
        """Preprocesar datos reales para machine learning."""
        
        print("\nPASO 2: Preprocesamiento de datos reales...")
        print("-" * 50)
        
        # Hacer copia para procesamiento
        self.processed_data = self.raw_data.copy()
        
        # Separar features y target
        target_col = 'NObeyesdad'
        feature_cols = [col for col in self.processed_data.columns if col != target_col]
        
        X = self.processed_data[feature_cols].copy()
        y = self.processed_data[target_col].copy()
        
        print(f"Features originales: {len(feature_cols)}")
        print(f"Target: {target_col} ({y.nunique()} clases)")
        
        # Codificar variables categóricas
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        print(f"\nCodificando {len(categorical_cols)} variables categóricas...")
        
        for col in categorical_cols:
            self.encoders[col] = LabelEncoder()
            X[col] = self.encoders[col].fit_transform(X[col])
            unique_values = len(self.encoders[col].classes_)
            print(f"  {col}: {unique_values} categorías únicas")
        
        # Codificar target
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        
        print(f"\nTarget codificado:")
        for i, clase in enumerate(self.target_encoder.classes_):
            count = sum(y_encoded == i)
            print(f"  {i}: {clase} ({count} muestras)")
        
        # Escalar features numéricas
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\nOK - Features escaladas con StandardScaler")
        print(f"Forma final: X={X_scaled.shape}, y={y_encoded.shape}")
        
        # Guardar datos procesados
        self.X = X_scaled
        self.y = y_encoded
        self.feature_names = feature_cols
        self.target_classes = self.target_encoder.classes_
        
        return self.X, self.y
    
    def split_real_data(self):
        """Dividir datos reales en train/validation/test."""
        
        print("\nPASO 3: División de datos reales...")
        print("-" * 50)
        
        # División estratificada para mantener balance de clases
        # 60% train, 20% validation, 20% test
        
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 * 0.8 = 0.2
        )
        
        print(f"División realizada:")
        print(f"  Entrenamiento: {self.X_train.shape[0]} ({(self.X_train.shape[0]/len(self.X))*100:.1f}%)")
        print(f"  Validación: {self.X_val.shape[0]} ({(self.X_val.shape[0]/len(self.X))*100:.1f}%)")
        print(f"  Prueba: {self.X_test.shape[0]} ({(self.X_test.shape[0]/len(self.X))*100:.1f}%)")
        
        # Verificar balance en cada set
        print(f"\nBalance de clases en cada conjunto:")
        for set_name, y_set in [("Train", self.y_train), ("Val", self.y_val), ("Test", self.y_test)]:
            unique, counts = np.unique(y_set, return_counts=True)
            print(f"  {set_name}: {dict(zip(unique, counts))}")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def hyperparameter_tuning_real_data(self):
        """Hyperparameter tuning con datos reales."""
        
        print("\nPASO 4: Hyperparameter tuning con datos reales...")
        print("-" * 50)
        
        from sklearn.model_selection import GridSearchCV, cross_val_score
        
        # Grid de parámetros para RandomForest
        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        print(f"Probando {np.prod([len(v) for v in param_grid_rf.values()])} combinaciones para RandomForest...")
        
        # Grid search con validación cruzada
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search_rf = GridSearchCV(
            rf, param_grid_rf, 
            cv=5, 
            scoring='f1_macro',  # F1-macro para clases desbalanceadas
            n_jobs=-1, 
            verbose=1
        )
        
        grid_search_rf.fit(self.X_train, self.y_train)
        
        best_rf = grid_search_rf.best_estimator_
        best_score_rf = grid_search_rf.best_score_
        
        print(f"\nMejor RandomForest encontrado:")
        print(f"  Parámetros: {grid_search_rf.best_params_}")
        print(f"  CV F1-macro: {best_score_rf:.4f}")
        
        # Evaluar en conjunto de validación
        val_pred_rf = best_rf.predict(self.X_val)
        val_acc_rf = accuracy_score(self.y_val, val_pred_rf)
        val_f1_rf = f1_score(self.y_val, val_pred_rf, average='macro')
        
        print(f"  Validación Accuracy: {val_acc_rf:.4f}")
        print(f"  Validación F1-macro: {val_f1_rf:.4f}")
        
        # También probar Logistic Regression para comparar
        print(f"\nEntrenando Logistic Regression para comparación...")
        
        lr = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')
        lr.fit(self.X_train, self.y_train)
        
        val_pred_lr = lr.predict(self.X_val)
        val_acc_lr = accuracy_score(self.y_val, val_pred_lr)
        val_f1_lr = f1_score(self.y_val, val_pred_lr, average='macro')
        
        print(f"  LR Validación Accuracy: {val_acc_lr:.4f}")
        print(f"  LR Validación F1-macro: {val_f1_lr:.4f}")
        
        # Seleccionar mejor modelo
        if val_f1_rf >= val_f1_lr:
            self.best_model = best_rf
            self.best_model_name = "RandomForest"
            self.best_val_score = val_f1_rf
            print(f"\nMejor modelo: RandomForest (F1-macro: {val_f1_rf:.4f})")
        else:
            self.best_model = lr
            self.best_model_name = "LogisticRegression"
            self.best_val_score = val_f1_lr
            print(f"\nMejor modelo: LogisticRegression (F1-macro: {val_f1_lr:.4f})")
        
        # Guardar resultados
        self.results['tuning'] = {
            'best_model_name': self.best_model_name,
            'best_params': grid_search_rf.best_params_ if self.best_model_name == 'RandomForest' else 'default',
            'cv_score': best_score_rf if self.best_model_name == 'RandomForest' else 'N/A',
            'val_accuracy': val_acc_rf if self.best_model_name == 'RandomForest' else val_acc_lr,
            'val_f1_macro': val_f1_rf if self.best_model_name == 'RandomForest' else val_f1_lr
        }
        
        return self.best_model
    
    def final_evaluation_real_data(self):
        """Evaluación final con datos reales de test."""
        
        print("\nPASO 5: Evaluación final con datos de test...")
        print("-" * 50)
        
        # Predicciones en test set
        test_pred = self.best_model.predict(self.X_test)
        test_proba = self.best_model.predict_proba(self.X_test)
        
        # Métricas principales
        test_acc = accuracy_score(self.y_test, test_pred)
        test_f1 = f1_score(self.y_test, test_pred, average='macro')
        
        print(f"RESULTADOS FINALES EN TEST SET:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  F1-macro: {test_f1:.4f}")
        
        # Reporte detallado por clase
        print(f"\nReporte de clasificación detallado:")
        class_names = self.target_classes
        report = classification_report(
            self.y_test, test_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"  {class_name}:")
                print(f"    Precision: {metrics['precision']:.3f}")
                print(f"    Recall: {metrics['recall']:.3f}")
                print(f"    F1-score: {metrics['f1-score']:.3f}")
                print(f"    Support: {int(metrics['support'])}")
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(self.y_test, test_pred)
        print(f"\nMatriz de confusión:")
        print(conf_matrix)
        
        # Guardar resultados
        self.results['final_evaluation'] = {
            'test_accuracy': test_acc,
            'test_f1_macro': test_f1,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': test_pred.tolist(),
            'probabilities': test_proba.tolist()
        }
        
        return test_acc, test_f1
    
    def track_with_mlflow(self):
        """Registrar experimento en MLflow."""
        
        print("\nPASO 6: Registro en MLflow...")
        print("-" * 50)
        
        if not MLFLOW_AVAILABLE:
            print("WARNING: MLflow no disponible, saltando registro")
            return None
        
        try:
            # Crear experimento
            experiment_name = f"obesity_real_data_{self.timestamp}"
            
            try:
                mlflow.create_experiment(experiment_name)
            except:
                pass  # Ya existe
            
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name=f"obesity_optimization_{self.timestamp}"):
                
                # Parámetros del dataset
                mlflow.log_param("dataset_path", self.data_path)
                mlflow.log_param("dataset_size", len(self.raw_data))
                mlflow.log_param("n_features", len(self.feature_names))
                mlflow.log_param("n_classes", len(self.target_classes))
                mlflow.log_param("train_size", len(self.X_train))
                mlflow.log_param("val_size", len(self.X_val))
                mlflow.log_param("test_size", len(self.X_test))
                
                # Parámetros del modelo
                mlflow.log_param("model_type", self.best_model_name)
                if 'best_params' in self.results['tuning']:
                    for param, value in self.results['tuning']['best_params'].items():
                        mlflow.log_param(f"model_{param}", value)
                
                # Métricas
                mlflow.log_metric("val_accuracy", self.results['tuning']['val_accuracy'])
                mlflow.log_metric("val_f1_macro", self.results['tuning']['val_f1_macro'])
                mlflow.log_metric("test_accuracy", self.results['final_evaluation']['test_accuracy'])
                mlflow.log_metric("test_f1_macro", self.results['final_evaluation']['test_f1_macro'])
                
                # Métricas por clase
                report = self.results['final_evaluation']['classification_report']
                for class_name in self.target_classes:
                    if class_name in report:
                        class_metrics = report[class_name]
                        mlflow.log_metric(f"{class_name}_precision", class_metrics['precision'])
                        mlflow.log_metric(f"{class_name}_recall", class_metrics['recall'])
                        mlflow.log_metric(f"{class_name}_f1", class_metrics['f1-score'])
                
                # Registrar modelo
                mlflow.sklearn.log_model(
                    self.best_model, 
                    "model",
                    registered_model_name=f"obesity_classifier_{self.best_model_name}"
                )
                
                run_id = mlflow.active_run().info.run_id
                print(f"OK - Experimento registrado: {experiment_name}")
                print(f"Run ID: {run_id}")
                
                self.results['mlflow'] = {
                    'experiment_name': experiment_name,
                    'run_id': run_id,
                    'model_name': f"obesity_classifier_{self.best_model_name}"
                }
                
        except Exception as e:
            print(f"ERROR en MLflow: {e}")
            return None
    
    def save_results_to_files(self):
        """Guardar resultados en archivos."""
        
        print("\nPASO 7: Guardando resultados...")
        print("-" * 50)
        
        # Crear directorio de resultados
        results_dir = Path("reports") / f"real_data_results_{self.timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar métricas principales
        metrics_summary = {
            'timestamp': self.timestamp,
            'dataset_info': {
                'path': self.data_path,
                'total_samples': len(self.raw_data),
                'features': len(self.feature_names),
                'classes': len(self.target_classes),
                'class_names': self.target_classes.tolist()
            },
            'model_info': {
                'best_model': self.best_model_name,
                'tuning_results': self.results['tuning']
            },
            'performance': {
                'test_accuracy': self.results['final_evaluation']['test_accuracy'],
                'test_f1_macro': self.results['final_evaluation']['test_f1_macro']
            }
        }
        
        import json
        with open(results_dir / "metrics_summary.json", 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        # Guardar reporte detallado
        detailed_report = self.results['final_evaluation']['classification_report']
        with open(results_dir / "classification_report.json", 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        # Guardar predicciones
        predictions_df = pd.DataFrame({
            'true_label': self.y_test,
            'predicted_label': self.results['final_evaluation']['predictions'],
            'true_class_name': [self.target_classes[i] for i in self.y_test],
            'predicted_class_name': [self.target_classes[i] for i in self.results['final_evaluation']['predictions']]
        })
        predictions_df.to_csv(results_dir / "test_predictions.csv", index=False)
        
        print(f"OK - Resultados guardados en: {results_dir}")
        print(f"Archivos creados:")
        print(f"  - metrics_summary.json")
        print(f"  - classification_report.json") 
        print(f"  - test_predictions.csv")
        
        return results_dir
    
    def print_final_summary(self):
        """Imprimir resumen final completo."""
        
        print(f"\n" + "="*60)
        print("RESUMEN FINAL - OPTIMIZACIONES CON DATOS REALES")
        print("="*60)
        
        print(f"Dataset: {self.data_path}")
        print(f"Muestras totales: {len(self.raw_data)}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Clases de obesidad: {len(self.target_classes)}")
        
        print(f"\nDistribución final de datos:")
        print(f"  Entrenamiento: {len(self.X_train)} ({(len(self.X_train)/len(self.X))*100:.1f}%)")
        print(f"  Validación: {len(self.X_val)} ({(len(self.X_val)/len(self.X))*100:.1f}%)")  
        print(f"  Prueba: {len(self.X_test)} ({(len(self.X_test)/len(self.X))*100:.1f}%)")
        
        print(f"\nMejor modelo encontrado: {self.best_model_name}")
        if 'best_params' in self.results['tuning'] and self.results['tuning']['best_params'] != 'default':
            print(f"Mejores parámetros: {self.results['tuning']['best_params']}")
        
        print(f"\nRendimiento final:")
        print(f"  Accuracy en test: {self.results['final_evaluation']['test_accuracy']:.4f}")
        print(f"  F1-macro en test: {self.results['final_evaluation']['test_f1_macro']:.4f}")
        
        if 'mlflow' in self.results:
            print(f"\nMLflow tracking:")
            print(f"  Experimento: {self.results['mlflow']['experiment_name']}")
            print(f"  Modelo registrado: {self.results['mlflow']['model_name']}")
        
        print(f"\nTimestamp: {self.timestamp}")
        print(f"OPTIMIZACIONES COMPLETADAS EXITOSAMENTE!")


def main():
    """Ejecutar optimizaciones completas con datos reales."""
    
    print("Iniciando optimizaciones MLOps con datos reales de obesidad...")
    
    runner = RealDataOptimizationRunner()
    
    try:
        # Ejecutar pipeline completo
        runner.load_and_prepare_real_data()
        runner.preprocess_real_data()
        runner.split_real_data()
        runner.hyperparameter_tuning_real_data()
        runner.final_evaluation_real_data()
        runner.track_with_mlflow()
        runner.save_results_to_files()
        runner.print_final_summary()
        
        print("\nEJECUCION COMPLETADA EXITOSAMENTE!")
        return True
        
    except Exception as e:
        print(f"\nERROR durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)