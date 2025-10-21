# Limpieza y validación de rangos para el dataset de obesidad (UCI)
# - Quita duplicados
# - Estándariza strings/categóricas
# - Valida y recorta (clip) rangos numéricos configurables
# - Imputa faltantes (numérico: mediana; categórico: moda) salvo que se configure otra cosa
# - Genera un reporte de calidad (JSON) con conteos y acciones aplicadas

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml


def _read_params(path: str | None) -> Dict[str, Any]:
    """
    Lee params.yaml si existe; si no, devuelve estructura por defecto.
    """
    default = {
        "data": {
            "raw": "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv",
            "interim": "data/interim/obesity_clean.csv",
            "target": "NObeyesdad"
        },
        "columns": {
            # Ajusta estas listas según tu dataset/notebooks
            "categorical": [
                "Gender", "family_history_with_overweight", "FAVC", "CAEC",
                "SMOKE", "SCC", "CALC", "MTRANS"
            ],
            "numerical": [
                "Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"
            ]
        },
        "validation": {
            # Rangos realistas: AJUSTA a tu criterio/proyecto si difieren
            "ranges": {
                "Age":   {"min": 10,   "max": 90},
                "Height":{"min": 1.2,  "max": 2.2},   # en metros
                "Weight":{"min": 30,   "max": 250},   # en kg
                "FCVC":  {"min": 1,    "max": 3},     # Frecuencia de consumo de vegetales
                "NCP":   {"min": 1,    "max": 6},     # # comidas principales/día
                "CH2O":  {"min": 1,    "max": 3},     # vasos de agua
                "FAF":   {"min": 0,    "max": 3},     # actividad física
                "TUE":   {"min": 0,    "max": 2}      # tiempo usando tecnología
            },
            # Acción sobre outliers: "clip" (recortar al rango), "drop" (eliminar filas), "flag" (marcar)
            "outlier_action": "clip"
        },
        "imputation": {
            # Estrategias por tipo
            "numeric_strategy": "median",     # median | mean | zero | none
            "categorical_strategy": "mode",   # mode | constant | none
            "categorical_fill_value": "UNKNOWN"
        },
        "normalization": {
            # Normalización ligera de categóricas: recorte, lower, reemplazo de espacios múltiples
            "strip_strings": True,
            "lowercase": False,   # pon True si quieres estandarizar a minúsculas
            "collapse_spaces": True
        }
    }

    if path is None:
        return default

    params_path = Path(path)
    if not params_path.exists():
        return default

    with open(params_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    # Mezcla superficial: y tiene prioridad; default rellena faltantes
    def merge(d, u):
        for k, v in u.items():
            if k not in d:
                d[k] = v
            elif isinstance(v, dict) and isinstance(d.get(k), dict):
                merge(d[k], v)
        return d

    return merge(y, default)


def standardize_strings(df: pd.DataFrame, cols: List[str], cfg: Dict[str, Any]) -> pd.DataFrame:
    """Normalización ligera de cadenas para consistencia."""
    if not cols:
        return df
    strip = cfg.get("strip_strings", True)
    lower = cfg.get("lowercase", False)
    collapse = cfg.get("collapse_spaces", True)

    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string")
            if strip:
                df[c] = df[c].str.strip()
            if collapse:
                df[c] = df[c].str.replace(r"\s+", " ", regex=True)
            if lower:
                df[c] = df[c].str.lower()
    return df


def validate_and_handle_outliers(df: pd.DataFrame, ranges: Dict[str, Dict[str, float]], action: str,
                                 report: Dict[str, Any]) -> pd.DataFrame:
    """
    Valida rangos numéricos y actúa sobre outliers:
    - clip: recorta al rango
    - drop: elimina filas con outliers
    - flag: agrega columnas *_is_outlier (True/False)
    """
    outlier_counts = {}
    for col, bounds in (ranges or {}).items():
        if col not in df.columns:
            continue
        col_min = bounds.get("min", -np.inf)
        col_max = bounds.get("max", np.inf)
        mask_low = df[col] < col_min
        mask_high = df[col] > col_max
        n_low = int(mask_low.sum())
        n_high = int(mask_high.sum())
        outlier_counts[col] = {"below_min": n_low, "above_max": n_high}

        if action == "clip":
            df[col] = df[col].clip(lower=col_min, upper=col_max)
        elif action == "drop":
            df = df[~(mask_low | mask_high)].copy()
        elif action == "flag":
            df[f"{col}__is_outlier_low"] = mask_low
            df[f"{col}__is_outlier_high"] = mask_high
        else:
            # acción no reconocida -> no hacer nada, pero reportar
            pass

    report["outliers"] = {
        "action": action,
        "counts": outlier_counts
    }
    return df


def impute_missing(df: pd.DataFrame,
                   numerical: List[str],
                   categorical: List[str],
                   imp_cfg: Dict[str, Any],
                   report: Dict[str, Any]) -> pd.DataFrame:
    """Imputación simple y configurable por tipo de variable."""
    num_strategy = imp_cfg.get("numeric_strategy", "median")
    cat_strategy = imp_cfg.get("categorical_strategy", "mode")
    cat_fill = imp_cfg.get("categorical_fill_value", "UNKNOWN")

    imputations = {"numerical": {}, "categorical": {}}

    # Numéricos
    for c in numerical:
        if c not in df.columns:
            continue
        n_missing = int(df[c].isna().sum())
        if n_missing == 0 or num_strategy == "none":
            continue

        if num_strategy == "median":
            val = df[c].median()
        elif num_strategy == "mean":
            val = df[c].mean()
        elif num_strategy == "zero":
            val = 0.0
        else:
            # fallback seguro
            val = df[c].median()

        df[c] = df[c].fillna(val)
        imputations["numerical"][c] = {"strategy": num_strategy, "value": float(val), "n_imputed": n_missing}

    # Categóricas
    for c in categorical:
        if c not in df.columns:
            continue
        n_missing = int(df[c].isna().sum())
        if n_missing == 0 or cat_strategy == "none":
            continue

        if cat_strategy == "mode":
            # si hay empate de moda, pandas devuelve la primera
            val = df[c].mode(dropna=True)
            val = val.iloc[0] if len(val) else cat_fill
        elif cat_strategy == "constant":
            val = cat_fill
        else:
            val = cat_fill

        df[c] = df[c].fillna(val)
        imputations["categorical"][c] = {"strategy": cat_strategy, "value": str(val), "n_imputed": n_missing}

    report["imputation"] = imputations
    return df


def main(inp: str, out: str, report_path: str | None, params_path: str | None) -> None:
    cfg = _read_params(params_path)

    # --- Carga ---
    df = pd.read_csv(inp)
    report: Dict[str, Any] = {"input_path": inp, "n_rows_in": int(df.shape[0]), "n_cols": int(df.shape[1])}

    # --- Tipos y listas ---
    cat_cols = [c for c in cfg["columns"]["categorical"] if c in df.columns]
    num_cols = [c for c in cfg["columns"]["numerical"] if c in df.columns]
    target = cfg["data"].get("target")

    # --- Limpieza básica ---
    # Estandarizar strings categóricas (espacios, caso)
    df = standardize_strings(df, cat_cols + ([target] if target in df.columns else []), cfg["normalization"])

    # Duplicados exactos
    n_dups = int(df.duplicated().sum())
    if n_dups > 0:
        df = df.drop_duplicates()
    report["duplicates_removed"] = n_dups

    # Conversión segura de numéricos
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Validación de rangos / outliers ---
    ranges = cfg["validation"].get("ranges", {})
    action = cfg["validation"].get("outlier_action", "clip")
    df = validate_and_handle_outliers(df, ranges, action, report)

    # --- Imputación de faltantes ---
    n_missing_before = int(df.isna().sum().sum())
    df = impute_missing(df, num_cols, cat_cols, cfg["imputation"], report)
    n_missing_after = int(df.isna().sum().sum())
    report["missing_values"] = {"total_before": n_missing_before, "total_after": n_missing_after}

    # --- Normalizaciones adicionales opcionales ---
    # Ejemplo: asegurar categorías consistentes en el target (si aplica)
    if target and target in df.columns:
        # Normaliza espacios múltiples en target, conserva el caso original salvo que cfg pida lowercase
        df[target] = df[target].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)

    # --- Salida ---
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    report["output_path"] = out
    report["n_rows_out"] = int(df.shape[0])

    # Reporte de calidad
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocesamiento: limpieza + validación de rangos + imputación")
    ap.add_argument("--inp", required=True, help="Ruta del CSV de entrada (raw)")
    ap.add_argument("--out", required=True, help="Ruta del CSV de salida (interim/obesity_clean.csv)")
    ap.add_argument("--report", default="reports/data_quality_report.json", help="Reporte de calidad (JSON)")
    ap.add_argument("--params", default="params.yaml", help="Ruta a params.yaml")
    args = ap.parse_args()

    main(inp=args.inp, out=args.out, report_path=args.report, params_path=args.params)
