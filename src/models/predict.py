# src/models/predict.py
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
import mlflow.pyfunc


def transform_with_artifacts(df_raw, artifacts_dir):
    """
    Transforma df_raw -> features usando los mismos artefactos
    (encoder/scaler/features_meta.json) guardados por make_features.py
    """
    meta = json.load(open(Path(artifacts_dir) / "features_meta.json", "r", encoding="utf-8"))

    enc_type = meta["encoder"]["type"]
    enc_cols = meta["encoder"]["columns"] or []
    sca_type = meta["scaler"]["type"]
    sca_cols = meta["scaler"]["columns"] or []

    # Encoder
    X_cat = pd.DataFrame(index=df_raw.index)
    if enc_type in ("onehot", "ordinal"):
        enc = joblib.load(Path(artifacts_dir) / "encoder.joblib")
        X_enc = enc.transform(df_raw[enc_cols].astype("string"))
        if enc_type == "onehot":
            fns = enc.get_feature_names_out(enc_cols)
            X_cat = pd.DataFrame(X_enc, columns=fns, index=df_raw.index)
        else:  # ordinal
            X_cat = pd.DataFrame(X_enc, columns=enc_cols, index=df_raw.index)
    elif enc_type == "label":
        # Si usaste label por columna en make_features, necesitarías guardar cada LabelEncoder
        # En esta guía asumimos onehot/ordinal para producción (recomendado).
        raise NotImplementedError("Modo 'label' no implementado para inferencia raw. Usa onehot/ordinal.")

    # Scaler
    X_num = pd.DataFrame(index=df_raw.index)
    if sca_type and sca_type != "none":
        scaler = joblib.load(Path(artifacts_dir) / "scaler.joblib")
        Xs = scaler.transform(df_raw[sca_cols])
        X_num = pd.DataFrame(Xs, columns=sca_cols, index=df_raw.index)
    else:
        X_num = df_raw[sca_cols].copy()

    # Ensamble
    parts = []
    if not X_num.empty: parts.append(X_num)
    if not X_cat.empty: parts.append(X_cat)
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df_raw.index)
    return X


def main(model_path, features_csv, raw_csv, artifacts_dir, out_csv):
    model = mlflow.pyfunc.load_model(model_path)

    if raw_csv:
        if not artifacts_dir:
            raise ValueError("Para raw_csv debes especificar --artifacts_dir (encoder/scaler/meta)")
        df_raw = pd.read_csv(raw_csv)
        X = transform_with_artifacts(df_raw, artifacts_dir)
        df_out = df_raw.copy()
    elif features_csv:
        df_feat = pd.read_csv(features_csv)
        if "__target__" in df_feat.columns:
            X = df_feat.drop(columns=["__target__"])
        else:
            X = df_feat
        df_out = df_feat.copy()
    else:
        raise ValueError("Debes proporcionar --features_csv o --raw_csv")

    preds = model.predict(X)
    df_out["prediction"] = preds

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_csv, index=False)
    else:
        print(df_out.head())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predicción batch/online con modelo MLflow")
    ap.add_argument("--model_path", default="models/mlflow_model", help="Ruta al modelo MLflow (sklearn flavor)")
    ap.add_argument("--features_csv", help="CSV de features ya procesadas (salida de make_features.py)")
    ap.add_argument("--raw_csv", help="CSV con datos en crudo (requiere artifacts_dir)")
    ap.add_argument("--artifacts_dir", help="Carpeta con encoder/scaler/features_meta.json")
    ap.add_argument("--out_csv", default="reports/predictions.csv", help="Salida CSV con predicciones")
    args = ap.parse_args()
    main(args.model_path, args.features_csv, args.raw_csv, args.artifacts_dir, args.out_csv)
