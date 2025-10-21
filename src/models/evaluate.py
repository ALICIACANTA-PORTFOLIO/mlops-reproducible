# src/models/evaluate.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow.pyfunc
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, labels, outpath):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_feature_importance(model, feature_names, outpath):
    if not hasattr(model, "feature_importances_"):
        return False
    fi = model.feature_importances_
    order = np.argsort(fi)[::-1][:20]  # top-20
    plt.figure(figsize=(7, 6))
    plt.barh(np.array(feature_names)[order][::-1], fi[order][::-1])
    plt.title("Feature Importance (Top 20)")
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
    return True


def main(processed_csv, model_path, eval_json, fig_cm, fig_fi):
    df = pd.read_csv(processed_csv)
    if "__target__" not in df.columns:
        raise ValueError("El dataset procesado debe contener '__target__'")

    y = df["__target__"]
    X = df.drop(columns=["__target__"])

    # Cargar modelo (guardado por train.py como MLflow sklearn model)
    model = mlflow.pyfunc.load_model(model_path)

    # Predicción
    pred = model.predict(X)

    # Reporte y CM
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    labels = sorted(np.unique(y))
    cm = confusion_matrix(y, pred)
    plot_confusion_matrix(cm, labels, fig_cm)

    # AUC macro-ovr si hay predict_proba (multi-clase)
    metrics = {"classification_report": report}
    try:
        # model._model_impl is MLflow PyFunc wrapper → acceso al estimador subyacente:
        clf = model._model_impl.python_model.model
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)
            auc = roc_auc_score(y, proba, multi_class="ovr")
            metrics["roc_auc_ovr"] = float(auc)
        # Feature Importance si existe
        if hasattr(clf, "feature_importances_"):
            ok = plot_feature_importance(clf, X.columns.tolist(), fig_fi)
            if ok:
                metrics["feature_importance"] = "saved"
            else:
                # Crear archivo vacío si no se pudo generar
                Path(fig_fi).parent.mkdir(parents=True, exist_ok=True)
                plt.figure(figsize=(1, 1))
                plt.text(0.5, 0.5, "Feature importance\nnot available", ha='center', va='center')
                plt.axis('off')
                plt.savefig(fig_fi, dpi=200)
                plt.close()
                metrics["feature_importance"] = "placeholder"
        else:
            # Crear archivo placeholder si el modelo no tiene feature_importances_
            Path(fig_fi).parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(1, 1))
            plt.text(0.5, 0.5, "Feature importance\nnot available", ha='center', va='center')
            plt.axis('off')
            plt.savefig(fig_fi, dpi=200)
            plt.close()
            metrics["feature_importance"] = "not_available"
    except Exception as e:
        # En caso de cualquier error, crear archivo placeholder
        Path(fig_fi).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(1, 1))
        plt.text(0.5, 0.5, f"Error generating\nfeature importance", ha='center', va='center')
        plt.axis('off')
        plt.savefig(fig_fi, dpi=200)
        plt.close()
        metrics["feature_importance"] = "error"

    Path(eval_json).parent.mkdir(parents=True, exist_ok=True)
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluación adicional del modelo + figuras")
    ap.add_argument("--data", required=True, help="CSV procesado con '__target__'")
    ap.add_argument("--model_path", default="models/mlflow_model", help="Ruta al modelo MLflow sklearn")
    ap.add_argument("--eval_json", default="reports/eval_metrics.json")
    ap.add_argument("--fig_cm", default="reports/figures/confusion_matrix_eval.png")
    ap.add_argument("--fig_fi", default="reports/figures/feature_importance.png")
    args = ap.parse_args()
    main(args.data, args.model_path, args.eval_json, args.fig_cm, args.fig_fi)
