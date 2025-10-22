#!/usr/bin/env python3
"""
Diagnóstico de MLflow - Verifica configuración y experimentos existentes.
"""

import mlflow
import pandas as pd
from pathlib import Path

def diagnose_mlflow():
    """Diagnóstica el estado actual de MLflow."""
    
    print("🔍 Diagnóstico de MLflow")
    print("=" * 50)
    
    # 1. Información básica
    print(f"📦 MLflow version: {mlflow.__version__}")
    print(f"📍 Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"📂 MLruns directory exists: {Path('mlruns').exists()}")
    
    # 2. Experimentos existentes
    print(f"\n🧪 Experimentos existentes:")
    try:
        experiments = mlflow.search_experiments()
        if experiments:
            for exp in experiments:
                print(f"  - {exp.name} (ID: {exp.experiment_id})")
                
                # Contar runs en este experimento
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                print(f"    Runs: {len(runs)}")
                if len(runs) > 0:
                    latest_run = runs.iloc[0]
                    print(f"    Latest run: {latest_run['run_id'][:8]}... ({latest_run['status']})")
        else:
            print("  ❌ No hay experimentos encontrados")
    except Exception as e:
        print(f"  ❌ Error al obtener experimentos: {e}")
    
    # 3. Verificar experimento específico
    print(f"\n🎯 Experimento 'obesity_classification':")
    try:
        exp = mlflow.get_experiment_by_name("obesity_classification")
        if exp:
            print(f"  ✅ Existe - ID: {exp.experiment_id}")
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            print(f"  📊 Runs totales: {len(runs)}")
            
            if len(runs) > 0:
                print(f"\n📈 Últimos 5 runs:")
                for i, (_, run) in enumerate(runs.head().iterrows()):
                    print(f"    {i+1}. {run['run_id'][:8]}... - Status: {run['status']}")
                    if 'metrics.accuracy' in run:
                        print(f"       Accuracy: {run['metrics.accuracy']:.4f}")
        else:
            print(f"  ❌ No existe")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # 4. Directorio mlruns
    print(f"\n📁 Contenido de mlruns/:")
    mlruns_path = Path("mlruns")
    if mlruns_path.exists():
        subdirs = [d for d in mlruns_path.iterdir() if d.is_dir()]
        print(f"  Subdirectorios: {len(subdirs)}")
        for subdir in sorted(subdirs)[:5]:  # Mostrar solo los primeros 5
            print(f"    - {subdir.name}")
    else:
        print(f"  ❌ Directorio mlruns no existe")
    
    # 5. Recomendar soluciones
    print(f"\n🔧 Posibles soluciones:")
    print(f"  1. Verificar que el entrenamiento se ejecute dentro de mlflow.start_run()")
    print(f"  2. Configurar MLflow tracking URI correctamente")
    print(f"  3. Verificar permisos de escritura en directorio mlruns")
    print(f"  4. Ejecutar: mlflow ui --port 5000")

def test_simple_logging():
    """Prueba simple de logging a MLflow."""
    
    print(f"\n🧪 Prueba de logging simple:")
    
    try:
        mlflow.set_experiment("test_experiment")
        
        with mlflow.start_run():
            # Log parámetros y métricas de prueba
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.95)
            
            run_id = mlflow.active_run().info.run_id
            print(f"  ✅ Run creado exitosamente: {run_id}")
            
        print(f"  ✅ Test de logging completado")
        
    except Exception as e:
        print(f"  ❌ Error en test de logging: {e}")

if __name__ == "__main__":
    diagnose_mlflow()
    test_simple_logging()
    
    print(f"\n" + "=" * 50)
    print(f"💡 Para ver la UI de MLflow, ejecuta: mlflow ui")
    print(f"🌐 Luego abre: http://localhost:5000")
    print(f"=" * 50)