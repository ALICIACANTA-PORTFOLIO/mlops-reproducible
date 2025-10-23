# src/models/train.py
import argparse, json
from pathlib import Path
import shutil
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn


def read_params(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg):
    mtype = (cfg["model"]["type"] or "random_forest").lower()
    if mtype == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg["model"].get("n_estimators", 300),
            max_depth=cfg["model"].get("max_depth", None),
            class_weight=cfg["model"].get("class_weight", None),
            random_state=cfg["split"].get("random_state", 52),
            n_jobs=-1
        )
    if mtype == "logistic_regression":
        return LogisticRegression(
            max_iter=cfg["model"].get("max_iter", 1000),
            class_weight=cfg["model"].get("class_weight", None),
            multi_class="auto",
            n_jobs=-1 if hasattr(LogisticRegression, "n_jobs") else None
        )
    raise ValueError(f"Modelo no soportado: {mtype}")


def plot_confusion_matrix(cm, labels, outpath):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def main(data_path, params_path, model_dir, metrics_path, fig_cm_path):
    P = read_params(params_path)
    
    # Configurar MLflow
    mlflow_config = P.get("mlflow", {})
    if mlflow_config.get("tracking_uri"):
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    
    experiment_name = mlflow_config.get("experiment_name", "obesity_classification")
    mlflow.set_experiment(experiment_name)
    
    print(f"🔬 MLflow experiment: {experiment_name}")
    print(f"📊 Tracking URI: {mlflow.get_tracking_uri()}")

    # Datos
    df = pd.read_csv(data_path)
    if "__target__" not in df.columns:
        raise ValueError("El dataset procesado debe contener la columna '__target__'")

    y = df["__target__"]
    X = df.drop(columns=["__target__"])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=P["split"]["test_size"],
        random_state=P["split"]["random_state"],
        stratify=y
    )

    model = build_model(P)

    with mlflow.start_run():
        # Agregar tags del experimento
        tags = mlflow_config.get("tags", {})
        for key, value in tags.items():
            mlflow.set_tag(key, value)
        mlflow.set_tag("data_size", len(df))
        mlflow.set_tag("n_features", len(X.columns))
        
        print(f"🚀 Starting MLflow run: {mlflow.active_run().info.run_id}")
        
        # Log de hiperparámetros detallados
        model_params = P.get("model", {})
        for k, v in model_params.items():
            mlflow.log_param(f"model__{k}", v)
        
        # Log parámetros de split
        split_params = P.get("split", {})
        for k, v in split_params.items():
            mlflow.log_param(f"split__{k}", v)
            
        # Log parámetros de features si existen
        if "features" in P:
            features_params = P.get("features", {})
            for k, v in features_params.items():
                if isinstance(v, (str, int, float, bool)):
                    mlflow.log_param(f"features__{k}", v)
                elif isinstance(v, list) and len(v) < 20:  # Solo listas pequeñas
                    mlflow.log_param(f"features__{k}", str(v)[:250])  # Truncar si es muy largo
        
        # Log configuración de validación si existe
        if "validation" in P:
            mlflow.log_param("outlier_action", P["validation"].get("outlier_action", "unknown"))
        
        # Log configuración de imputación si existe  
        if "imputation" in P:
            imp_params = P.get("imputation", {})
            for k, v in imp_params.items():
                if isinstance(v, (str, int, float, bool)):
                    mlflow.log_param(f"imputation__{k}", v)

        # Entrenar
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        # Métricas macro (multiclase)
        acc = accuracy_score(yte, pred)
        f1m = f1_score(yte, pred, average="macro")
        prm = precision_score(yte, pred, average="macro", zero_division=0)
        rcm = recall_score(yte, pred, average="macro", zero_division=0)

        # Log de métricas
        print(f"📊 Logging metrics to MLflow...")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1m)
        mlflow.log_metric("precision_macro", prm)
        mlflow.log_metric("recall_macro", rcm)
        
        print(f"✅ Metrics logged - Accuracy: {acc:.4f}, F1: {f1m:.4f}")

        # Reporte detallado
        report = classification_report(yte, pred, output_dict=True, zero_division=0)
        
        # Log métricas detalladas por clase
        print(f"📊 Logging detailed per-class metrics...")
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                # Limpiar nombre de clase para MLflow
                clean_name = class_name.replace(' ', '_').replace('-', '_')
                mlflow.log_metric(f"{clean_name}_precision", metrics['precision'])
                mlflow.log_metric(f"{clean_name}_recall", metrics['recall'])
                mlflow.log_metric(f"{clean_name}_f1", metrics['f1-score'])
                mlflow.log_metric(f"{clean_name}_support", metrics['support'])
        
        # Log métricas agregadas adicionales
        if 'weighted avg' in report:
            mlflow.log_metric("weighted_avg_precision", report['weighted avg']['precision'])
            mlflow.log_metric("weighted_avg_recall", report['weighted avg']['recall'])
            mlflow.log_metric("weighted_avg_f1", report['weighted avg']['f1-score'])
        
        # Log información del dataset
        mlflow.log_param("dataset_path", data_path)
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("n_classes", len(np.unique(y)))
        mlflow.log_param("class_distribution", dict(pd.Series(y).value_counts()))
        mlflow.log_param("train_size", len(Xtr))
        mlflow.log_param("test_size", len(Xte))

        # Guardar métricas a disco
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {"accuracy": acc, "f1_macro": f1m, "precision_macro": prm, "recall_macro": rcm,
                 "classification_report": report},
                f, indent=2, ensure_ascii=False
            )

        # Matriz de confusión (como artefacto)
        cm = confusion_matrix(yte, pred)
        labels = sorted(np.unique(y))
        plot_confusion_matrix(cm, labels, fig_cm_path)
        mlflow.log_artifact(fig_cm_path)
        
        # Log métricas adicionales de la matriz de confusión
        # Diagonal (verdaderos positivos)
        tp_total = np.diag(cm).sum()
        total_predictions = cm.sum()
        mlflow.log_metric("true_positives_total", int(tp_total))
        mlflow.log_metric("total_predictions", int(total_predictions))
        
        # Feature importance si el modelo lo soporta
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Log top 10 feature importances
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"feature_{i}" for i in range(len(importances))]
            importance_pairs = list(zip(feature_names, importances))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature_name, importance) in enumerate(importance_pairs[:10]):
                mlflow.log_metric(f"feature_importance_{i+1:02d}_{feature_name}", importance)
                
            # Log feature importance como artifact
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            top_features = importance_pairs[:15]
            features, importances_vals = zip(*top_features)
            plt.barh(range(len(top_features)), importances_vals[::-1])
            plt.yticks(range(len(top_features)), features[::-1])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importances')
            plt.tight_layout()
            
            importance_path = Path(fig_cm_path).parent / "feature_importance.png"
            importance_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(importance_path, dpi=200, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(str(importance_path))

        # Guardar y loggear el modelo
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        local_model_path = Path(model_dir) / "mlflow_model"
        
        # Si el directorio existe, eliminarlo primero para evitar conflictos
        if local_model_path.exists():
            shutil.rmtree(local_model_path)
            
        # Guardar modelo localmente
        print(f"💾 Saving model locally to: {local_model_path}")
        mlflow.sklearn.save_model(model, str(local_model_path))
        
        # Log modelo con signature e input_example para producción
        print(f"📤 Logging model to MLflow...")
        from mlflow.models.signature import infer_signature
        signature = infer_signature(Xtr, pred)
        input_example = Xtr.iloc[:1] if hasattr(Xtr, 'iloc') else Xtr[:1]
        
        # Nombre del modelo registrado
        registered_model_name = mlflow_config.get("registered_model_name", "obesity_classifier")
        
        # Log model con registro automático
        model_info = mlflow.sklearn.log_model(
            model, 
            "model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name
        )
        
        print(f"✅ Model logged to MLflow successfully!")
        print(f"🔗 Run ID: {mlflow.active_run().info.run_id}")
        print(f"📊 Experiment: {mlflow.get_experiment(mlflow.active_run().info.experiment_id).name}")
        
        # Model Registry: Gestión de versiones y aliases
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Obtener la última versión del modelo registrado
            # Buscar por nombre del modelo y run_id
            versions = client.search_model_versions(
                f"name='{registered_model_name}' and run_id='{mlflow.active_run().info.run_id}'"
            )
            
            if versions:
                model_version = versions[0].version
                
                print(f"\n🏷️  Model Registry Operations:")
                print(f"   Model Name: {registered_model_name}")
                print(f"   Version: {model_version}")
                
                # Agregar descripción al modelo
                client.update_model_version(
                    name=registered_model_name,
                    version=model_version,
                    description=f"RandomForest model trained on {len(df)} samples. "
                                f"Accuracy: {acc:.4f}, F1: {f1m:.4f}. "
                                f"Features: {len(X.columns)}. "
                                f"Run ID: {mlflow.active_run().info.run_id}"
                )
                
                # Transición automática a Staging si el modelo es bueno
                accuracy_threshold = mlflow_config.get("staging_threshold", 0.85)
                if acc >= accuracy_threshold:
                    client.transition_model_version_stage(
                        name=registered_model_name,
                        version=model_version,
                        stage="Staging",
                        archive_existing_versions=False
                    )
                    print(f"   ✅ Transitioned to STAGING (accuracy {acc:.4f} >= {accuracy_threshold})")
                    
                    # Agregar alias si es el mejor modelo hasta ahora
                    try:
                        # Obtener todas las versiones en Staging o Production
                        all_versions = client.search_model_versions(
                            f"name='{registered_model_name}'"
                        )
                        
                        # Filtrar por stage y obtener métricas
                        best_accuracy = 0.0
                        for v in all_versions:
                            if v.current_stage in ['Staging', 'Production']:
                                try:
                                    run = client.get_run(v.run_id)
                                    v_acc = run.data.metrics.get('accuracy', 0.0)
                                    if v_acc > best_accuracy:
                                        best_accuracy = v_acc
                                except:
                                    continue
                        
                        # Si este modelo es el mejor, asignar alias champion
                        if acc >= best_accuracy:
                            client.set_registered_model_alias(
                                registered_model_name,
                                "champion",
                                model_version
                            )
                            print(f"   🏆 Alias 'champion' assigned (best accuracy: {acc:.4f})")
                    except Exception as e:
                        print(f"   ⚠️  Could not assign alias: {e}")
                else:
                    print(f"   ⚠️  Model below staging threshold (accuracy {acc:.4f} < {accuracy_threshold})")
                    print(f"   📦 Model registered but not promoted to Staging")
                
                # Agregar tags adicionales al modelo registrado
                client.set_model_version_tag(
                    registered_model_name,
                    model_version,
                    "validation_status",
                    "passed" if acc >= accuracy_threshold else "needs_review"
                )
                client.set_model_version_tag(
                    registered_model_name,
                    model_version,
                    "model_type",
                    P["model"]["type"]
                )
                client.set_model_version_tag(
                    registered_model_name,
                    model_version,
                    "training_date",
                    pd.Timestamp.now().strftime("%Y-%m-%d")
                )
                
                print(f"   📋 Model version tags updated")
                
        except Exception as e:
            print(f"⚠️  Model Registry operations failed: {e}")
            print(f"   Model logged but not registered. Check MLflow configuration.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrenamiento de modelo + logging MLflow")
    ap.add_argument("--data", required=True, help="CSV de features procesadas (de make_features.py)")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--metrics", default="reports/metrics.json")
    ap.add_argument("--fig_cm", default="reports/figures/confusion_matrix.png")
    args = ap.parse_args()
    main(args.data, args.params, args.model_dir, args.metrics, args.fig_cm)
