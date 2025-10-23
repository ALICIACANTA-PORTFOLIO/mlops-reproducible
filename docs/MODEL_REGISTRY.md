# 🏷️ Model Registry - Guía Completa

## 📋 ¿Qué es Model Registry?

El **MLflow Model Registry** es un sistema centralizado para:
- ✅ **Versionado de modelos** - Historial completo de versiones
- ✅ **Gestión de lifecycle** - Stages: None → Staging → Production
- ✅ **Aliases** - Identificadores semánticos (champion, challenger)
- ✅ **Metadata enriquecido** - Tags, descripciones, métricas
- ✅ **Transiciones automatizadas** - Promoción automática basada en métricas

---

## 🚀 Características Implementadas

### **1. Registro Automático de Modelos**

Cada entrenamiento registra automáticamente el modelo:

```python
# En src/models/train.py
mlflow.sklearn.log_model(
    model, 
    "model",
    signature=signature,  # ✅ Schema de input/output
    input_example=input_example,  # ✅ Ejemplo de entrada
    registered_model_name="obesity_classifier"  # ✅ Registro automático
)
```

**Resultado**: Modelo versionado con schema y validación automática.

---

### **2. Transiciones Automáticas a Staging**

Si `accuracy >= staging_threshold` (default: 0.85):

```python
client.transition_model_version_stage(
    name="obesity_classifier",
    version=model_version,
    stage="Staging"
)
```

**Resultado**: Modelos buenos promocionados automáticamente.

---

### **3. Sistema de Aliases**

Aliases semánticos para referencia fácil:

```bash
# Asignar alias "champion" al mejor modelo
python manage_registry.py alias obesity_classifier champion 2

# Usar el modelo champion en código
model = mlflow.pyfunc.load_model("models:/obesity_classifier@champion")
```

**Aliases comunes**:
- `champion` - Mejor modelo en Production
- `challenger` - Candidato a reemplazar champion
- `baseline` - Modelo de referencia

---

### **4. Tags Enriquecidos**

Cada versión incluye metadata útil:

```python
Tags automáticos:
- validation_status: "passed" | "needs_review"
- model_type: "random_forest" | "logistic_regression"
- training_date: "2025-10-22"
- n_features: 32
- dataset_size: 2087
```

---

### **5. Model Signatures**

**¿Qué son?** Schemas que validan input/output automáticamente.

```python
signature = infer_signature(X_train, predictions)
```

**Beneficio**: Detecta errores de schema antes de producción.

**Ejemplo**:
```
Inputs: [Age: double, Height: double, Weight: double, ...]
Outputs: [long]
```

---

## 🛠️ Usando manage_registry.py

### **Listar Modelos Registrados**

```bash
python manage_registry.py list
```

**Output**:
```
📊 MODELOS REGISTRADOS (1):

🔹 obesity_classifier
   Descripción: RandomForest obesity classifier
   Versiones: 2 total
      - Production: 1
      - None: 1
```

---

### **Ver Versiones con Métricas**

```bash
python manage_registry.py versions obesity_classifier
```

**Output**:
```
+---------+------------+----------+--------+-----------+
| Version | Stage      | Accuracy | F1     | Aliases   |
+=========+============+==========+========+===========+
| 2       | Production | 0.9266   | 0.9251 | champion  |
| 1       | None       | 0.9266   | 0.9251 | -         |
+---------+------------+----------+--------+-----------+
```

---

### **Promover Modelo a Production**

```bash
python manage_registry.py promote obesity_classifier 2 Production
```

**Resultado**:
```
✅ Modelo obesity_classifier v2 promovido a Production
📌 Versiones anteriores archivadas automáticamente
```

---

### **Asignar Alias**

```bash
python manage_registry.py alias obesity_classifier champion 2
```

**Uso en código**:
```python
# Cargar siempre el modelo champion
model = mlflow.pyfunc.load_model("models:/obesity_classifier@champion")
```

---

### **Comparar Versiones**

```bash
python manage_registry.py compare obesity_classifier 1 2
```

**Output**:
```
📊 COMPARACIÓN: obesity_classifier v1 vs v2

+-----------+--------+--------+-------------+
| Métrica   | v1     | v2     | Diferencia  |
+===========+========+========+=============+
| accuracy  | 0.9266 | 0.9266 | =           |
| f1_macro  | 0.9251 | 0.9251 | =           |
+-----------+--------+--------+-------------+
```

---

### **Encontrar Mejor Modelo**

```bash
python manage_registry.py best obesity_classifier --metric accuracy
```

**Output**:
```
🏆 MEJOR MODELO por 'accuracy':
   Modelo: obesity_classifier
   Versión: 2
   Stage: Production
   accuracy: 0.9266
```

---

## 🔄 Workflow Completo

### **1. Entrenar Modelo**

```bash
python src/models/train.py \
    --data data/processed/features.csv \
    --params params.yaml \
    --model_dir models \
    --metrics reports/metrics.json \
    --fig_cm reports/figures/confusion_matrix.png
```

**Automático**:
- ✅ Modelo registrado
- ✅ Signature inferida
- ✅ Transición a Staging (si accuracy >= 0.85)
- ✅ Tags asignados

---

### **2. Revisar Versiones**

```bash
python manage_registry.py versions obesity_classifier
```

---

### **3. Promover a Production**

```bash
# Opción A: Manual
python manage_registry.py promote obesity_classifier 2 Production

# Opción B: Automático (si accuracy >= threshold)
# Ya se hizo durante entrenamiento
```

---

### **4. Asignar Alias**

```bash
python manage_registry.py alias obesity_classifier champion 2
```

---

### **5. Usar en API/Serving**

```python
import mlflow

# Opción A: Por versión específica
model = mlflow.pyfunc.load_model("models:/obesity_classifier/2")

# Opción B: Por stage
model = mlflow.pyfunc.load_model("models:/obesity_classifier/Production")

# Opción C: Por alias (recomendado)
model = mlflow.pyfunc.load_model("models:/obesity_classifier@champion")
```

---

## 📊 Configuración en params.yaml

```yaml
mlflow:
  experiment_name: obesity_classification_v2
  tracking_uri: ./mlruns
  registered_model_name: obesity_classifier  # Nombre del modelo
  staging_threshold: 0.85  # Accuracy mínimo para Staging
  tags:
    project: mlops-reproducible
    team: data-science
    environment: development
```

---

## 🎯 Beneficios para Portfolio

### **1. Governance Completo**
- ✅ Historial completo de versiones
- ✅ Lifecycle management
- ✅ Auditoría y trazabilidad

### **2. Production-Ready**
- ✅ Model signatures (validación automática)
- ✅ Input examples (documentación)
- ✅ Metadata enriquecido

### **3. Automatización**
- ✅ Transiciones automáticas
- ✅ Detección de mejor modelo
- ✅ Aliases inteligentes

### **4. Enterprise-Grade**
- ✅ Stages (None → Staging → Production)
- ✅ Versionado semántico
- ✅ Rollback fácil

---

## 🏆 Comparación: Antes vs Ahora

### **ANTES (Sin Registry)**
```python
# ❌ Sin versionado
model = joblib.load('model.pkl')

# ❌ Sin validación
predictions = model.predict(data)  # Puede fallar silenciosamente

# ❌ Sin historial
# ¿Cuál modelo es mejor?
# ¿Cuándo fue entrenado?
```

### **AHORA (Con Registry)**
```python
# ✅ Versionado automático
model = mlflow.pyfunc.load_model("models:/obesity_classifier@champion")

# ✅ Validación automática por signature
predictions = model.predict(data)  # Schema validado

# ✅ Historial completo
# python manage_registry.py versions obesity_classifier
# Ver todas las versiones, métricas, stages
```

---

## 📚 Referencias

**Basado en**:
- 📖 "Machine Learning Engineering with MLflow" - Capítulos 5-6
- 🔗 MLflow Model Registry Docs: https://mlflow.org/docs/latest/model-registry.html

**Implementación**:
- `src/models/train.py` - Registro automático
- `manage_registry.py` - CLI para gestión
- `params.yaml` - Configuración

---

**✨ Estado**: ✅ IMPLEMENTADO Y FUNCIONAL  
**📅 Fecha**: 2025-10-22  
**🎯 Impacto Portfolio**: ⭐⭐⭐⭐⭐ (MÁXIMO)
