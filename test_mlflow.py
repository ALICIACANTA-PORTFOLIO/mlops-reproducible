#!/usr/bin/env python3
"""
Test rápido de MLflow para verificar que todo funciona correctamente.
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def test_mlflow_integration():
    """Test completo de integración MLflow."""
    
    print("🧪 Testing MLflow Integration")
    print("=" * 40)
    
    # 1. Configurar MLflow
    print("1️⃣ Configurando MLflow...")
    
    # Asegurar que el directorio mlruns existe
    Path("mlruns").mkdir(exist_ok=True)
    
    # Configurar tracking URI (local)
    mlflow.set_tracking_uri("./mlruns")
    print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Crear/obtener experimento
    experiment_name = "obesity_classification"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"   ✅ Experimento creado: {experiment_name} (ID: {experiment_id})")
        else:
            mlflow.set_experiment(experiment_name)
            print(f"   ✅ Experimento existe: {experiment_name} (ID: {experiment.experiment_id})")
    except Exception as e:
        print(f"   ❌ Error con experimento: {e}")
        return False
    
    # 2. Verificar datos
    print("\n2️⃣ Verificando datos...")
    data_path = "data/processed/features.csv"
    
    if not Path(data_path).exists():
        print(f"   ❌ Archivo de datos no encontrado: {data_path}")
        print(f"   💡 Ejecuta primero el preprocessing y feature engineering")
        return False
    
    try:
        df = pd.read_csv(data_path)
        print(f"   ✅ Datos cargados: {df.shape}")
        
        if "__target__" not in df.columns:
            print(f"   ❌ Columna '__target__' no encontrada")
            return False
        
        y = df["__target__"]
        X = df.drop(columns=["__target__"])
        print(f"   ✅ Features: {X.shape[1]}, Target classes: {len(y.unique())}")
        
    except Exception as e:
        print(f"   ❌ Error cargando datos: {e}")
        return False
    
    # 3. Entrenamiento con MLflow
    print("\n3️⃣ Entrenamiento con MLflow...")
    
    try:
        with mlflow.start_run(run_name="test_run") as run:
            print(f"   🚀 Run iniciado: {run.info.run_id}")
            
            # Split datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Crear y entrenar modelo
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            )
            
            print(f"   🤖 Entrenando modelo...")
            model.fit(X_train, y_train)
            
            # Predicciones y métricas
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # Log parámetros
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("n_features", X.shape[1])
            
            # Log métricas
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_macro", f1_macro)
            
            # Log modelo
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train, y_pred)
            
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=X_train.iloc[:2]
            )
            
            print(f"   ✅ Métricas registradas:")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      F1-macro: {f1_macro:.4f}")
            print(f"   ✅ Modelo registrado")
            
            # Tags adicionales
            mlflow.set_tag("test", "integration_test")
            mlflow.set_tag("framework", "sklearn")
            
        print(f"   ✅ Run completado exitosamente: {run.info.run_id}")
        
    except Exception as e:
        print(f"   ❌ Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Verificar registro
    print("\n4️⃣ Verificando registro...")
    
    try:
        # Buscar runs en el experimento
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        print(f"   📊 Total runs en experimento: {len(runs)}")
        
        if len(runs) > 0:
            latest_run = runs.iloc[0]
            print(f"   🆔 Último run ID: {latest_run['run_id']}")
            print(f"   📈 Accuracy: {latest_run.get('metrics.accuracy', 'N/A')}")
            print(f"   📈 F1-macro: {latest_run.get('metrics.f1_macro', 'N/A')}")
        
    except Exception as e:
        print(f"   ❌ Error verificando runs: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("✅ MLflow integration test EXITOSO!")
    print("💡 Para ver la UI: mlflow ui")
    print("🌐 Luego abre: http://localhost:5000")
    print("=" * 40)
    
    return True

if __name__ == "__main__":
    success = test_mlflow_integration()
    
    if not success:
        print("\n❌ Test falló. Posibles soluciones:")
        print("1. Ejecutar preprocessing: python run_mlops.py cli preprocess --input data/raw/dataset.csv --output data/interim/clean.csv")
        print("2. Ejecutar feature engineering: python run_mlops.py cli features --input data/interim/clean.csv --output data/processed/features.csv")
        print("3. Verificar permisos de escritura")
        print("4. Verificar instalación de mlflow: pip install mlflow")