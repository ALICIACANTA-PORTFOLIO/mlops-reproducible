# src/models/train.py
import argparse, json
from pathlib import Path
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
        # Log de hiperparámetros y split
        for k, v in P.get("model", {}).items():
            mlflow.log_param(f"model__{k}", v)
        mlflow.log_param("split__test_size", P["split"]["test_size"])
        mlflow.log_param("split__random_state", P["split"]["random_state"])

        # Entrenar
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        # Métricas macro (multiclase)
        acc = accuracy_score(yte, pred)
        f1m = f1_score(yte, pred, average="macro")
        prm = precision_score(yte, pred, average="macro", zero_division=0)
        rcm = recall_score(yte, pred, average="macro", zero_division=0)

        # Log de métricas
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1m)
        mlflow.log_metric("precision_macro", prm)
        mlflow.log_metric("recall_macro", rcm)

        # Reporte detallado
        report = classification_report(yte, pred, output_dict=True, zero_division=0)

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
        # Si guardaste mapping del target en features_meta.json, puedes mapear etiquetas legibles
        labels = sorted(np.unique(y))
        plot_confusion_matrix(cm, labels, fig_cm_path)
        mlflow.log_artifact(fig_cm_path)

        # Guardar y loggear el modelo
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        local_model_path = str(Path(model_dir) / "mlflow_model")
        mlflow.sklearn.save_model(model, local_model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrenamiento de modelo + logging MLflow")
    ap.add_argument("--data", required=True, help="CSV de features procesadas (de make_features.py)")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--metrics", default="reports/metrics.json")
    ap.add_argument("--fig_cm", default="reports/figures/confusion_matrix.png")
    args = ap.parse_args()
    main(args.data, args.params, args.model_dir, args.metrics, args.fig_cm)
