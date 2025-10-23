# src/serving/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Obesity Classification MLOps API",
    description="API para clasificación de niveles de obesidad usando MLOps pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas de entrada y salida
class HealthData(BaseModel):
    """Schema para datos de entrada de salud"""
    Age: float = Field(..., description="Edad", ge=10, le=90)
    Height: float = Field(..., description="Altura en metros", ge=1.2, le=2.2)
    Weight: float = Field(..., description="Peso en kg", ge=30, le=250)
    FCVC: float = Field(..., description="Frecuencia consumo vegetales (1-3)", ge=1, le=3)
    NCP: float = Field(..., description="Número comidas principales (1-6)", ge=1, le=6)
    CH2O: float = Field(..., description="Consumo agua (1-3)", ge=1, le=3)
    FAF: float = Field(..., description="Actividad física (0-3)", ge=0, le=3)
    TUE: float = Field(..., description="Uso tecnología (0-2)", ge=0, le=2)
    Gender: str = Field(..., description="Género: Female o Male")
    family_history_with_overweight: str = Field(..., description="Historia familiar: yes o no")
    FAVC: str = Field(..., description="Consumo comida alta en calorías: yes o no")
    CAEC: str = Field(..., description="Consumo comida entre comidas: no, Sometimes, Frequently, Always")
    SMOKE: str = Field(..., description="Fumador: yes o no")
    SCC: str = Field(..., description="Monitor calorías: yes o no")
    CALC: str = Field(..., description="Consumo alcohol: no, Sometimes, Frequently, Always")
    MTRANS: str = Field(..., description="Transporte: Walking, Bike, Motorbike, Public_Transportation, Automobile")

class PredictionResponse(BaseModel):
    """Schema para respuesta de predicción"""
    prediction: str = Field(..., description="Nivel de obesidad predicho")
    prediction_proba: Dict[str, float] = Field(..., description="Probabilidades por clase")
    confidence: float = Field(..., description="Confianza de la predicción (0-1)")
    risk_level: str = Field(..., description="Nivel de riesgo: Low, Medium, High")

class BatchPredictionRequest(BaseModel):
    """Schema para predicciones en lote"""
    data: List[HealthData] = Field(..., description="Lista de datos para predicción")

class BatchPredictionResponse(BaseModel):
    """Schema para respuesta de predicciones en lote"""
    predictions: List[PredictionResponse] = Field(..., description="Lista de predicciones")
    summary: Dict[str, Any] = Field(..., description="Resumen de predicciones")

# Variables globales para modelo y preprocessors
model = None
encoder = None
scaler = None
feature_names = None
class_mapping = None

def load_model_artifacts():
    """Cargar modelo y artefactos de preprocessing"""
    global model, encoder, scaler, feature_names, class_mapping
    
    try:
        # Paths de artefactos
        model_path = Path("models/mlflow_model")
        features_path = Path("models/features")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
        
        # Cargar modelo MLflow
        model = mlflow.sklearn.load_model(str(model_path))
        logger.info(f"Modelo cargado desde {model_path}")
        
        # Cargar preprocessors
        encoder_path = features_path / "encoder.joblib"
        scaler_path = features_path / "scaler.joblib"
        meta_path = features_path / "features_meta.json"
        
        if encoder_path.exists():
            encoder = joblib.load(encoder_path)
            logger.info("Encoder cargado")
            
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info("Scaler cargado")
            
        if meta_path.exists():
            import json
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                feature_names = meta.get('feature_names', [])
                class_mapping = meta.get('class_mapping', {})
            logger.info("Metadata cargada")
            
    except Exception as e:
        logger.error(f"Error cargando artefactos: {str(e)}")
        raise

def preprocess_input(data: HealthData) -> np.ndarray:
    """Preprocessar datos de entrada"""
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Aplicar preprocessing similar al entrenamiento
        categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 
                          'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        
        # Separar categoricas y numericas
        df_cat = df[categorical_cols].copy()
        df_num = df[numerical_cols].copy()
        
        # Aplicar encoder y scaler si están disponibles
        if encoder is not None:
            df_cat_encoded = encoder.transform(df_cat)
            if hasattr(df_cat_encoded, 'toarray'):  # OneHotEncoder
                df_cat_encoded = df_cat_encoded.toarray()
            df_cat_encoded = pd.DataFrame(df_cat_encoded)
        else:
            # Encoding manual básico como fallback
            df_cat_encoded = pd.get_dummies(df_cat)
            
        if scaler is not None:
            df_num_scaled = scaler.transform(df_num)
            df_num_scaled = pd.DataFrame(df_num_scaled, columns=numerical_cols)
        else:
            df_num_scaled = df_num
            
        # Concatenar
        df_processed = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
        
        return df_processed.values
        
    except Exception as e:
        logger.error(f"Error en preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error procesando datos: {str(e)}")

def get_risk_level(prediction: str) -> str:
    """Determinar nivel de riesgo basado en predicción"""
    risk_mapping = {
        'Normal_Weight': 'Low',
        'Insufficient_Weight': 'Medium',
        'Overweight_Level_I': 'Medium',
        'Overweight_Level_II': 'Medium',
        'Obesity_Type_I': 'High',
        'Obesity_Type_II': 'High',
        'Obesity_Type_III': 'High'
    }
    return risk_mapping.get(prediction, 'Medium')

# Eventos de startup y shutdown
@app.on_event("startup")
async def startup_event():
    """Cargar modelo al iniciar la aplicación"""
    logger.info("Iniciando API de clasificación de obesidad...")
    load_model_artifacts()
    logger.info("API lista para recibir requests")

# Endpoints
@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Obesity Classification MLOps API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": encoder is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(data: HealthData):
    """Predicción para un único caso"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Preprocessar datos
        X = preprocess_input(data)
        
        # Predicción
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        # Mapear clases si hay mapping disponible
        if class_mapping:
            # Invertir mapping para obtener nombres de clases
            inv_mapping = {v: k for k, v in class_mapping.items()}
            prediction_name = inv_mapping.get(prediction, str(prediction))
        else:
            prediction_name = str(prediction)
            
        # Crear diccionario de probabilidades
        class_names = getattr(model, 'classes_', range(len(prediction_proba)))
        proba_dict = {}
        for i, prob in enumerate(prediction_proba):
            if class_mapping and i in class_mapping.values():
                inv_mapping = {v: k for k, v in class_mapping.items()}
                class_name = inv_mapping[i]
            else:
                class_name = str(class_names[i]) if hasattr(class_names, '__getitem__') else f"class_{i}"
            proba_dict[class_name] = float(prob)
        
        confidence = float(np.max(prediction_proba))
        risk_level = get_risk_level(prediction_name)
        
        return PredictionResponse(
            prediction=prediction_name,
            prediction_proba=proba_dict,
            confidence=confidence,
            risk_level=risk_level
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predicciones en lote"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        predictions = []
        risk_counts = {"Low": 0, "Medium": 0, "High": 0}
        
        for data in request.data:
            pred_response = await predict_single(data)
            predictions.append(pred_response)
            risk_counts[pred_response.risk_level] += 1
            
        summary = {
            "total_predictions": len(predictions),
            "risk_distribution": risk_counts,
            "average_confidence": np.mean([p.confidence for p in predictions])
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción batch: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Información sobre el modelo cargado"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    info = {
        "model_type": type(model).__name__,
        "model_loaded": True,
        "feature_names": feature_names if feature_names else "No disponible",
        "num_features": len(feature_names) if feature_names else "Desconocido",
        "classes": class_mapping if class_mapping else "No disponible",
        "artifacts_status": {
            "encoder": encoder is not None,
            "scaler": scaler is not None,
            "metadata": class_mapping is not None
        }
    }
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)