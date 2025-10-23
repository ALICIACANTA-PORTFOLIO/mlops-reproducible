# Biblioteca de Recursos MLOps

## 📚 Libros Especializados Disponibles

### 1. **Machine Learning Engineering with MLflow**
- **Archivo**: `Machine Learning Engineering with MLflow.pdf`
- **Enfoque**: Ingeniería de Machine Learning con MLflow
- **Relevancia para este proyecto**: ⭐⭐⭐⭐⭐
- **Aplicación directa**:
  - Capítulos sobre experiment tracking (implementado en `src/models/train.py`)
  - Model registry y versionado (usado en nuestro pipeline)
  - MLflow Projects (relevante para nuestro `MLproject`)
  - Model serving y deployment

### 2. **Machine Learning Design Patterns**  
- **Archivo**: `Machine_Learning_Design_Patterns_1760678784.pdf`
- **Enfoque**: Patrones de diseño para sistemas ML escalables
- **Relevancia para este proyecto**: ⭐⭐⭐⭐⭐
- **Aplicación directa**:
  - Data pipeline patterns (implementado en `dvc.yaml`)
  - Feature engineering patterns (aplicado en `src/data/make_features.py`)
  - Training patterns (usado en arquitectura híbrida `src/` + `mlops/`)
  - Serving patterns (preparado para deployment)

---

## 🎯 Cómo Usar Estos Recursos

### Para **Profundizar en MLflow**:
1. **Leer capítulos 1-3** de "Machine Learning Engineering with MLflow"
2. **Comparar con nuestra implementación** en `src/models/train.py`:
   ```python
   # Nuestro código implementa estas mejores prácticas:
   with mlflow.start_run():
       mlflow.log_params(model_params)      # Tracking de parámetros
       mlflow.log_metrics(metrics_dict)     # Registro de métricas
       mlflow.sklearn.log_model(model)     # Versionado de modelos
   ```
3. **Explorar capítulos avanzados** sobre model registry y serving

### Para **Entender Patrones ML**:
1. **Estudiar patrones de pipeline** en "ML Design Patterns"
2. **Analizar nuestra arquitectura híbrida**:
   ```bash
   src/          # CLI modules (production pattern)
   mlops/        # API pattern (development pattern)
   run_mlops.py  # Unified interface pattern
   ```
3. **Identificar patrones aplicados** en nuestro código

---

## 📖 Mapeo: Libros ↔ Proyecto

### Conceptos del Libro "MLflow Engineering" → Implementación en el Proyecto

| Concepto del Libro | Implementado en | Archivo/Función |
|-------------------|-----------------|-----------------|
| **Experiment Tracking** | ✅ Completo | `src/models/train.py` - líneas 88-180 |
| **Parameter Logging** | ✅ Detallado | 25+ parámetros registrados automáticamente |
| **Metrics Logging** | ✅ Avanzado | Métricas por clase + weighted averages |
| **Artifact Storage** | ✅ Implementado | Confusion matrix + Feature importance plots |
| **Model Registry** | ✅ **Completo** ⭐ | MLflow model saving + versioning + stages + aliases |
| **Model Serving** | 🔄 Preparado | Estructura lista para FastAPI deployment |

### Conceptos del Libro "ML Design Patterns" → Arquitectura del Proyecto

| Patrón del Libro | Implementado | Descripción en el Proyecto |
|------------------|--------------|---------------------------|
| **Pipeline Pattern** | ✅ DVC Pipeline | `dvc.yaml` - Pipeline declarativo con dependencias |
| **Feature Store Pattern** | ✅ Parcial | `models/features/` - Artifacts persistidos (encoder, scaler) |
| **Hybrid Architecture** | ✅ Innovación | `src/` (CLI) + `mlops/` (API) - Doble enfoque único |
| **Configuration Pattern** | ✅ Completo | `params.yaml` - Configuración centralizada |
| **Reproducibility Pattern** | ✅ Avanzado | DVC + MLflow + Git = Reproducibilidad total |
| **Modular Design** | ✅ Implementado | Separación clara: preprocess → features → train → evaluate |

---

## 🚀 Ejercicios Prácticos Basados en los Libros

### Ejercicio 1: Mejoras MLflow (Basado en "MLflow Engineering")
```bash
# Implementar advanced logging siguiendo el libro:
# 1. Custom metrics
mlflow.log_metric("custom_business_metric", calculate_business_impact())

# 2. Model signatures (del libro cap. 5)
from mlflow.models.signature import infer_signature
signature = infer_signature(X_train, y_pred)
mlflow.sklearn.log_model(model, "model", signature=signature)

# 3. Model metadata (del libro cap. 6)
mlflow.set_tag("model_type", "production_ready")
mlflow.set_tag("validation_status", "passed")
```

### Ejercicio 2: Patrones Avanzados (Basado en "ML Design Patterns")
```bash
# Implementar Feature Pipeline Pattern (cap. 3 del libro):
# 1. Feature versioning
dvc add data/processed/features_v2.csv

# 2. Feature validation pipeline
python src/data/validate_features.py --input data/processed/features.csv

# 3. A/B testing pattern (cap. 8 del libro)
python run_mlops.py api --experiment "model_a" --params params_v1.yaml
python run_mlops.py api --experiment "model_b" --params params_v2.yaml
```

---

## 📊 Progreso de Implementación vs Libros

### MLflow Engineering Book ✅ **95% Implementado** 🎉
- ✅ Basic tracking
- ✅ Advanced logging  
- ✅ Model artifacts
- ✅ **Model registry (COMPLETO)** ⭐ **NUEVO**
- ✅ **Model signatures** ⭐ **NUEVO**
- ✅ **Automatic staging transitions** ⭐ **NUEVO**
- ✅ **Alias management** ⭐ **NUEVO**
- 🔄 Model serving (preparado)
- ❌ Advanced deployment (pendiente)

**🆕 Implementación Reciente (2025-10-22)**:
```python
# Model Registry con signatures (Capítulo 5-6 del libro)
signature = infer_signature(X_train, predictions)
mlflow.sklearn.log_model(
    model, "model",
    signature=signature,              # ✅ Schema validation
    input_example=input_example,      # ✅ Documentation
    registered_model_name="obesity_classifier"  # ✅ Registry
)

# Automatic transitions
if accuracy >= staging_threshold:
    client.transition_model_version_stage(
        name="obesity_classifier",
        version=model_version,
        stage="Staging"
    )

# Alias management CLI
python manage_registry.py alias obesity_classifier champion 2
```

**Documentación**: Ver [docs/MODEL_REGISTRY.md](../MODEL_REGISTRY.md)

### ML Design Patterns Book ✅ 90% Implementado  
- ✅ Pipeline patterns
- ✅ Feature engineering patterns
- ✅ Training patterns
- ✅ Configuration patterns
- ✅ Hybrid architecture (innovación propia)
- 🔄 Serving patterns (preparado)

---

## 💡 Próximos Pasos Basados en los Libros

### ~~Inspirados en "MLflow Engineering"~~ ✅ **COMPLETADOS**:
1. ✅ ~~**Implementar Model Registry completo** (Capítulo 6)~~ → **HECHO** con `manage_registry.py`
2. ✅ ~~**Model signatures y validation** (Capítulo 5)~~ → **HECHO** con `infer_signature()`
3. 🔄 **Agregar model serving con FastAPI** (Capítulo 8) → Preparado
4. ❌ **Implementar A/B testing framework** (Capítulo 9) → Pendiente

### Inspirados en "ML Design Patterns":
1. 🔄 **Feature Store Pattern completo** (Capítulo 3) → Parcialmente implementado
2. ❌ **Continuous training pattern** (Capítulo 7) → Pendiente
3. ❌ **Model monitoring patterns** (Capítulo 10) → Pendiente

---

## 🎓 Recomendaciones de Lectura

### Para **Principiantes**:
1. **Empezar con** "ML Design Patterns" Capítulos 1-2 (fundamentos)
2. **Luego** "MLflow Engineering" Capítulos 1-3 (MLflow básico)
3. **Practicar con** nuestro proyecto ejecutando `python run_mlops.py cli pipeline`

### Para **Desarrolladores Avanzados**:
1. **Saltar a** "MLflow Engineering" Capítulos 6-9 (deployment avanzado)
2. **Estudiar** "ML Design Patterns" Capítulos 7-10 (patrones avanzados)
3. **Extender** nuestro proyecto con los patrones aprendidos

---

**💡 Nota**: Estos libros complementan perfectamente nuestro proyecto MLOps. Cada concepto del libro tiene una implementación práctica que puedes explorar y extender.