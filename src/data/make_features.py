# Generación de features a partir de un CSV "interim": codificación de categóricas y escalado numérico.
# - Toma listas de columnas desde params.yaml (categorical, numerical, target).
# - Encoder: onehot (default) | ordinal | label
# - Scaler: standard (default) | minmax | robust | none
# - Guarda CSV procesado y opcionalmente los artefactos (encoder/scaler) y el mapeo del target.

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib


def _read_params(path: Optional[str]) -> Dict[str, Any]:
    default = {
        "data": {
            "interim": "data/interim/obesity_clean.csv",
            "processed": "data/processed/features.csv",
            "target": "NObeyesdad"
        },
        "features": {
            "categorical": [
                "Gender", "family_history_with_overweight", "FAVC", "CAEC",
                "SMOKE", "SCC", "CALC", "MTRANS"
            ],
            "numerical": [
                "Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"
            ],
            "encoder": "onehot",        # onehot | ordinal | label
            "scaler": "standard",       # standard | minmax | robust | none
            "save_artifacts": True,     # guardar encoder/scaler/mapeo target
            "artifacts_dir": "models/features"  # carpeta para artefactos
        }
    }
    if not path:
        return default
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    # merge superficial
    def merge(dst, src):
        for k, v in src.items():
            if k not in dst:
                dst[k] = v
            elif isinstance(dst.get(k), dict) and isinstance(v, dict):
                merge(dst[k], v)
        return dst
    return merge(y, default)


def _build_encoder(name: str, categories: Optional[List[List[str]]] = None):
    name = (name or "onehot").lower()
    if name == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if name == "ordinal":
        # si se pasan categorías, las respeta; si no, aprende del fit
        return OrdinalEncoder(categories=categories)
    if name == "label":
        # LabelEncoder será por columna; lo manejaremos aparte
        return "label"
    raise ValueError(f"Encoder no soportado: {name}")


def _build_scaler(name: str):
    name = (name or "standard").lower()
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    if name == "none":
        return None
    raise ValueError(f"Scaler no soportado: {name}")


def encode_categoricals(df: pd.DataFrame,
                        cat_cols: List[str],
                        encoder_cfg: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Retorna (df_cat_encoded, encoder_info)
    - Si onehot/ordinal: usa un único encoder multi-columna
    - Si label: usa LabelEncoder por columna
    """
    info: Dict[str, Any] = {"type": encoder_cfg, "columns": cat_cols}
    if not cat_cols:
        return pd.DataFrame(index=df.index), info

    if encoder_cfg == "label":
        # LabelEncoder por columna (devuelve DataFrame con mismas columnas)
        encoders = {}
        out = pd.DataFrame(index=df.index)
        for c in cat_cols:
            le = LabelEncoder()
            vals = df[c].astype("string").fillna("UNKNOWN")
            out[c] = le.fit_transform(vals)
            encoders[c] = {
                "classes_": list(le.classes_)
            }
        info["label_encoders"] = encoders
        return out, info

    if encoder_cfg == "onehot":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X = enc.fit_transform(df[cat_cols].astype("string"))
        feature_names = enc.get_feature_names_out(cat_cols)
        out = pd.DataFrame(X, columns=feature_names, index=df.index)
        info["onehot"] = {
            "feature_names_out": feature_names.tolist(),
            "categories_": [list(c) for c in enc.categories_]
        }
        info["_sk_encoder"] = enc  # para persistir con joblib si se desea
        return out, info

    if encoder_cfg == "ordinal":
        enc = OrdinalEncoder()
        X = enc.fit_transform(df[cat_cols].astype("string"))
        out = pd.DataFrame(X, columns=cat_cols, index=df.index)
        info["ordinal"] = {
            "categories_": [list(c) for c in enc.categories_]
        }
        info["_sk_encoder"] = enc
        return out, info

    raise ValueError(f"Encoder no soportado: {encoder_cfg}")


def scale_numericals(df: pd.DataFrame,
                     num_cols: List[str],
                     scaler_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Retorna (df_num_scaled, scaler_info)
    """
    info: Dict[str, Any] = {"type": scaler_name, "columns": num_cols}
    if not num_cols:
        return pd.DataFrame(index=df.index), info
    scaler = _build_scaler(scaler_name)
    if scaler is None:
        return df[num_cols].copy(), info
    X = scaler.fit_transform(df[num_cols])
    out = pd.DataFrame(X, columns=num_cols, index=df.index)
    info["_sk_scaler"] = scaler
    return out, info


def map_target(df: pd.DataFrame, target_col: Optional[str]) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Opcionalmente codifica target -> entero y devuelve mapping.
    Si target no está presente, retorna (None, {}).
    """
    if not target_col or target_col not in df.columns:
        return pd.Series(index=df.index, dtype="float64"), {}
    y_raw = df[target_col].astype("string")
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    mapping = {
        "classes_": list(le.classes_),
        "mapping": {cls: int(i) for i, cls in enumerate(le.classes_)}
    }
    return pd.Series(y, index=df.index, name="__target__"), mapping


def save_artifacts(artifacts_dir: str,
                   encoder_info: Dict[str, Any],
                   scaler_info: Dict[str, Any],
                   target_info: Dict[str, Any]) -> None:
    d = Path(artifacts_dir)
    d.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "encoder": {"type": encoder_info.get("type"), "columns": encoder_info.get("columns")},
        "scaler":  {"type": scaler_info.get("type"),  "columns": scaler_info.get("columns")},
        "target":  {"classes_": target_info.get("classes_"), "mapping": target_info.get("mapping")}
    }

    # Persistir objetos sklearn si existen
    if "_sk_encoder" in encoder_info and encoder_info["_sk_encoder"] is not None:
        joblib.dump(encoder_info["_sk_encoder"], d / "encoder.joblib")
        # quita objeto no serializable del meta
        encoder_info = {k: v for k, v in encoder_info.items() if k != "_sk_encoder"}
    if "_sk_scaler" in scaler_info and scaler_info["_sk_scaler"] is not None:
        joblib.dump(scaler_info["_sk_scaler"], d / "scaler.joblib")
        scaler_info = {k: v for k, v in scaler_info.items() if k != "_sk_scaler"}

    # Guardar metadatos legibles
    meta["encoder"].update({k: v for k, v in encoder_info.items() if not k.startswith("_sk_")})
    meta["scaler"].update({k: v for k, v in scaler_info.items() if not k.startswith("_sk_")})

    with open(d / "features_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main(inp: str, out: str, params_path: Optional[str]) -> None:
    cfg = _read_params(params_path)
    F = cfg["features"]
    target_col = cfg["data"].get("target")

    df = pd.read_csv(inp)

    cat_cols = [c for c in F.get("categorical", []) if c in df.columns]
    num_cols = [c for c in F.get("numerical", []) if c in df.columns]

    # Construcción de target
    y_series, target_info = map_target(df, target_col)

    # Categóricas
    encoder_name = (F.get("encoder") or "onehot").lower()
    X_cat, encoder_info = encode_categoricals(df, cat_cols, encoder_name)

    # Numéricas
    scaler_name = (F.get("scaler") or "standard").lower()
    X_num, scaler_info = scale_numericals(df, num_cols, scaler_name)

    # Concatenar features
    parts = []
    if not X_num.empty:
        parts.append(X_num)
    if not X_cat.empty:
        parts.append(X_cat)
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)

    # Anexar target __target__ si existe
    if not y_series.empty:
        X["__target__"] = y_series.values

    # Guardar CSV de salida
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(out, index=False)

    # Guardar artefactos si se solicita
    if F.get("save_artifacts", True):
        save_artifacts(F.get("artifacts_dir", "models/features"), encoder_info, scaler_info, target_info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make features: codificación de categóricas y escalado numérico")
    ap.add_argument("--inp", required=True, help="Ruta del CSV de entrada (interim)")
    ap.add_argument("--out", required=True, help="Ruta del CSV de salida (processed)")
    ap.add_argument("--params", default="params.yaml", help="Ruta a params.yaml")
    args = ap.parse_args()

    main(inp=args.inp, out=args.out, params_path=args.params)
