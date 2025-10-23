# Proyecto MLOps - Clasificación de Obesidad

Un proyecto de Machine Learning Operations (MLOps) limpio, ordenado y funcional para la clasificación de obesidad. Implementa las mejores prácticas de MLOps con un enfoque profesional y reproducible.

## 🌟 Características Destacadas

✅ **MLflow Model Registry** - Versionado y lifecycle management de modelos  
✅ **Model Signatures** - Validación automática de schemas input/output  
✅ **DVC Integration** - Versionado de datos y reproducibilidad  
✅ **FastAPI** - API REST production-ready  
✅ **Pytest** - Suite completa de pruebas  
✅ **Enterprise-Grade** - Automatización de transiciones de modelos (Staging → Production)

---

## 🏷️ Model Registry

El proyecto implementa **MLflow Model Registry** con capacidades avanzadas:

```bash
# Listar modelos registrados
python manage_registry.py list

# Ver versiones y métricas
python manage_registry.py versions obesity_classifier

# Promover modelo a Production
python manage_registry.py promote obesity_classifier 2 Production

# Asignar alias "champion"
python manage_registry.py alias obesity_classifier champion 2

# Comparar versiones
python manage_registry.py compare obesity_classifier 1 2

# Encontrar mejor modelo
python manage_registry.py best obesity_classifier --metric accuracy
```

**Características**:
- 🔄 Registro automático durante entrenamiento
- ✅ Model signatures para validación
- 📊 Transiciones automáticas a Staging (accuracy >= 0.85)
- 🏆 Sistema de aliases (champion, challenger)
- 📈 Tags enriquecidos con metadata
- 🔍 CLI completa para gestión

**Documentación completa**: [docs/MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md)

---

## 🧪 Pruebas

```bash
# Ejecutar todas las pruebas
python -m pytest tests/ -v

# Pruebas específicas
python -m pytest tests/test_data_validation.py -v

# Probar API (si está corriendo)
python test_api.py
```

## 📝 Requisitos

- Python 3.8+
- scikit-learn 1.3+
- pandas 2.0+
- numpy 1.21+
- mlflow 2.0+ (opcional)
- dvc 3.0+ (opcional)

## 🤝 Contribución

Este es un proyecto de portfolio que demuestra implementación profesional de MLOps siguiendo las mejores prácticas de la industria.

## 📄 Licencia

Proyecto educativo - MIT License

---

## � Estructura del Proyecto (Arquitectura Híbrida)

```
mlops-reproducible/
├── src/                      # 🔧 Módulos CLI (DVC Pipeline)
│   ├── data/
│   │   ├── preprocess.py     # Limpieza y validación de datos
│   │   └── make_features.py  # Ingeniería de características
│   └── models/
│       ├── train.py          # Entrenamiento con MLflow
│       ├── evaluate.py       # Evaluación y métricas
│       └── predict.py        # Predicciones batch
├── mlops/                    # 🐍 API Python (Uso Interactivo)
│   ├── __init__.py          # Inicialización del módulo
│   ├── config.py            # Gestión de configuración
│   ├── dataset.py           # Procesamiento de datos
│   ├── features.py          # Ingeniería de características
│   ├── modeling.py          # Entrenamiento y evaluación
│   └── train.py            # Pipeline principal
├── data/                    # 📊 Datos versionados con DVC
│   ├── raw/                # Datos originales
│   ├── interim/            # Datos procesados intermedio
│   └── processed/          # Datos finales procesados
├── models/                  # 🤖 Modelos entrenados
├── notebooks/               # 📓 Notebooks de exploración
├── reports/                 # 📈 Reportes y métricas
├── docs/                    # 📚 Documentación
├── tests/                   # 🧪 Pruebas unitarias
├── run_mlops.py            # 🚀 Interface unificada
├── params.yaml              # ⚙️ Configuración principal
├── dvc.yaml                 # 🔄 Pipeline DVC
└── requirements.txt         # 📦 Dependencias
```

## 🎯 Dos Enfoques, Una Funcionalidad

### **Enfoque 1: `src/` - Módulos CLI (Recomendado para Producción)**

- ✅ **DVC Integration** - Perfecto para pipelines automatizados
- ✅ **Modular** - Cada script es independiente
- ✅ **CI/CD Ready** - Fácil integración en workflows
- ✅ **MLflow Tracking** - Registro automático de experimentos

### **Enfoque 2: `mlops/` - Python API (Recomendado para Desarrollo)**

- ✅ **Interactive** - Perfecto para notebooks y experimentación
- ✅ **Clean API** - Interfaz Python elegante y fácil de usar
- ✅ **Integrated** - Pipeline completo en una sola llamada
- ✅ **Flexible** - Configuración programática

---

## 🧩 Flujo MLOps Implementado

```mermaid
flowchart LR
    A[📊 Datos brutos] --> B[🧹 Limpieza DVC]
    B --> C[⚙️ Feature Engineering]
    C --> D[🤖 Entrenamiento MLflow]
    D --> E[📈 Model Registry]
    E --> F[🚀 Production]
    F --> G[📡 Monitoring]
```

**Pipeline Completo**:
1. **DVC** versiona datos y reproduce pipeline
2. **MLflow** trackea experimentos y registra modelos
3. **Model Registry** gestiona lifecycle (Staging → Production)
4. **FastAPI** sirve predicciones
5. **Pytest** valida calidad

---

## 📊 MLflow: Experiment Tracking & Model Registry

### **¿Qué hace MLflow en este proyecto?**

MLflow maneja **todo el ciclo de vida del modelo ML**:

#### **1. Experiment Tracking** 
Cada entrenamiento registra automáticamente:
- ✅ **Parámetros** (n_estimators, max_depth, etc.)
- ✅ **Métricas** (accuracy, F1, precision por clase)
- ✅ **Artefactos** (modelo, confusion matrix, feature importance)
- ✅ **Metadata** (fecha, duración, versión de código)

```bash
# Entrenar y trackear automáticamente
python src/models/train.py

# Ver experimentos en UI
mlflow ui
# Abre: http://localhost:5000
```

**Resultado**: Historial completo de experimentos para comparar y reproducir.

#### **2. Model Registry** ⭐ (Enterprise-Grade)
Sistema completo de versionado y lifecycle:

```bash
# Ver modelos registrados
python manage_registry.py list

# Ver versiones y métricas
python manage_registry.py versions obesity_classifier
# Output:
# Version | Stage      | Accuracy | F1     | Aliases
# 2       | Production | 0.9266   | 0.9251 | champion
# 1       | None       | 0.9266   | 0.9251 | -

# Promover a Production
python manage_registry.py promote obesity_classifier 2 Production

# Asignar alias para uso fácil
python manage_registry.py alias obesity_classifier champion 2
```

**Uso en código**:
```python
import mlflow

# Cargar modelo por alias (recomendado)
model = mlflow.pyfunc.load_model("models:/obesity_classifier@champion")
predictions = model.predict(new_data)
```

**Características avanzadas**:
- 🔄 **Transiciones automáticas**: Si accuracy >= 0.85 → auto-promoción a Staging
- ✅ **Model Signatures**: Validación automática de schemas input/output
- 📊 **Metadata enriquecido**: Tags (model_type, training_date, validation_status)
- 🏆 **Aliases semánticos**: champion, challenger, baseline

**Documentación completa**: [docs/MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md)

---

## 🔄 DVC: Data Version Control & Pipeline

### **¿Qué hace DVC en este proyecto?**

DVC maneja **versionado de datos y reproducibilidad del pipeline**:

#### **1. Versionado de Datos**
Los datos grandes se versioanan con DVC (no con Git):

```bash
# Ver datos versionados
cat data/raw/ObesityDataSet_raw_and_data_sinthetic.csv.dvc

# Descargar datos (si no los tienes)
dvc pull

# Actualizar datos
dvc add data/raw/nuevo_dataset.csv
git add data/raw/nuevo_dataset.csv.dvc
git commit -m "Update dataset"
dvc push
```

**Beneficio**: Git solo guarda el hash, no archivos grandes (2.1MB versionado como 165 bytes).

#### **2. Pipeline Reproducible**
`dvc.yaml` define el pipeline completo con dependencias:

```yaml
stages:
  preprocess:
    cmd: python src/data/preprocess.py
    deps:
      - data/raw/ObesityDataSet_raw_and_data_sinthetic.csv
    outs:
      - data/interim/obesity_clean.csv
      
  make_features:
    cmd: python src/data/make_features.py
    deps:
      - data/interim/obesity_clean.csv
    outs:
      - data/processed/features.csv
      
  train:
    cmd: python src/models/train.py
    deps:
      - data/processed/features.csv
      - params.yaml
    outs:
      - models/obesity_model.pkl
      - reports/metrics.json
```

**Ejecutar pipeline completo**:
```bash
# Reproducir todo el pipeline
dvc repro

# Ver DAG visual
dvc dag

# Output:
# +--------------+
# | preprocess   |
# +--------------+
#       *
#       *
#       *
# +--------------+
# | make_features|
# +--------------+
#       *
#       *
#       *
# +--------------+
# | train        |
# +--------------+
```

**DVC solo re-ejecuta stages modificados** (cache inteligente).

#### **3. Reproducibilidad Garantizada**
```bash
# Alguien clona tu repo
git clone <repo>
dvc pull  # Descarga datos

# Reproduce exactamente tus resultados
dvc repro

# Resultado: Mismo modelo, mismas métricas (0.0000 difference)
```

**Validado con tests**:
```bash
pytest tests/test_data_validation.py::test_reproducibility
# ✅ PASSED - Difference: 0.0000000000
```

---

## 🔗 MLflow + DVC: Mejor Juntos

| Herramienta | Qué versiona | Cuándo usar |
|-------------|--------------|-------------|
| **Git** | Código Python, configs | Siempre |
| **DVC** | Datos, modelos grandes | Archivos > 1MB |
| **MLflow** | Experimentos, parámetros, métricas | Cada entrenamiento |

**Workflow combinado**:
```bash
# 1. DVC reproduce pipeline
dvc repro

# 2. MLflow trackea experimentos automáticamente
# (cada stage de DVC registra en MLflow)

# 3. Git versiona código + metadatos
git add dvc.lock params.yaml
git commit -m "Experiment: increased max_depth"

# 4. DVC versiona datos y modelos
dvc push

# 5. MLflow Model Registry gestiona lifecycle
python manage_registry.py promote obesity_classifier 2 Production
```

**Resultado**: Reproducibilidad total con historial completo.

---

## ⚙️ Stack Tecnológico

| Categoría | Herramienta | Propósito | Uso en el proyecto |
|-----------|-------------|-----------|-------------------|
| 🐍 **Core** | Python 3.10 | Lenguaje base | Todo el código |
| 📊 **ML** | scikit-learn | Modelo ML | RandomForest (92.66% accuracy) |
| 🔄 **Data Versioning** | **DVC 3.30** | Versionar datos/modelos | `dvc.yaml` pipeline + `dvc pull/push` |
| 📈 **Experiment Tracking** | **MLflow 2.8** | Trackear experimentos | Auto-logging en `train.py` |
| 🏷️ **Model Registry** | **MLflow Registry** | Lifecycle de modelos | Staging → Production |
| ⚡ **API** | FastAPI | Servir predicciones | REST API en `start_api.py` |
| 🧪 **Testing** | Pytest | Testing automatizado | 3/3 tests passing |
| 📦 **Environment** | Conda | Gestión de entorno | Python 3.10.19 reproducible |
| 📝 **Config** | YAML | Configuración | `params.yaml`, `dvc.yaml` |
| 🔍 **Version Control** | Git/GitHub | Código versionado | Repo completo |

### **Comandos Rápidos por Herramienta**:

#### **DVC** (Datos y Pipeline)
```bash
dvc pull              # Descargar datos
dvc repro             # Ejecutar pipeline completo
dvc dag               # Ver grafo de dependencias
dvc push              # Subir datos/modelos
```

#### **MLflow** (Experimentos y Modelos)
```bash
mlflow ui                                    # Ver experimentos (localhost:5000)
python manage_registry.py list               # Ver modelos registrados
python manage_registry.py versions <model>   # Ver versiones y métricas
```

#### **FastAPI** (Servir Modelo)
```bash
uvicorn start_api:app --reload              # Iniciar API (localhost:8000)
python test_api.py                          # Test API
```

#### **Pytest** (Testing)
```bash
pytest tests/ -v                            # Ejecutar todos los tests
pytest tests/test_data_validation.py -v     # Test específico
```

---

## ⚙️ Instalación y Configuración

### **Setup Inicial**

```bash
# 1️⃣ Clonar repositorio
git clone https://github.com/ALICIACANTA-PORTFOLIO/mlops-reproducible.git
cd mlops-reproducible

# 2️⃣ Crear entorno Python 3.10
conda create -n mlops-reproducible python=3.10.19 -y
conda activate mlops-reproducible

# 3️⃣ Instalar dependencias
pip install -r requirements.txt

# 4️⃣ Descargar datos (si están en DVC remote)
dvc pull
```

**Verificación**:
```bash
# Ejecutar tests
pytest tests/ -v
# ✅ 3/3 tests passing

# Ver estructura MLflow
ls mlruns/
```

---

## 🚀 Quick Start

### **Opción 1: Pipeline Completo con DVC** (Recomendado)

```bash
# Ejecutar todas las etapas: preprocess → features → train
dvc repro

# Ver métricas
dvc metrics show

# Ver DAG del pipeline
dvc dag
```

**Output esperado**:
```
Running stage 'preprocess'...
Running stage 'make_features'...
Running stage 'train'...
✅ Model trained: Accuracy 0.9266

reports/metrics.json:
    accuracy: 0.9266
    f1_macro: 0.9251
```

### **Opción 2: Entrenar Directamente con MLflow**

```bash
# Entrenar y registrar en MLflow Registry automáticamente
python src/models/train.py \
    --data data/processed/features.csv \
    --params params.yaml \
    --model_dir models \
    --metrics reports/metrics.json

# Ver experimento en UI
mlflow ui
# Abre: http://localhost:5000
```

**Auto-ejecuta**:
- ✅ Tracking de parámetros (25+ params)
- ✅ Registro de métricas (accuracy, F1, etc.)
- ✅ Registro en Model Registry
- ✅ Transición a Staging si accuracy >= 0.85

### **Opción 3: Usar Python API (Desarrollo Interactivo)**

```python
from mlops import train, config

# Cargar configuración
params = config.load_params('params.yaml')

# Entrenar modelo
train.train_model(params)
```

---

## 🧮 Trabajar con DVC

### **Ver y modificar parámetros**

```bash
# Editar configuración
nano params.yaml

# Cambiar hiperparámetros (ejemplo)
random_forest:
  n_estimators: 200  # era 100
  max_depth: 15      # era 10

# Re-ejecutar pipeline (solo lo modificado)
dvc repro

# Comparar métricas con versión anterior
dvc metrics diff
```

**Output**:
```diff
Path              Metric     Old      New      Change
reports/metrics.json
                  accuracy   0.9266   0.9312   +0.0046
                  f1_macro   0.9251   0.9298   +0.0047
```

### **Versionado de datos**

```bash
# Agregar nuevo dataset
dvc add data/raw/nuevo_dataset.csv

# Commitear metadata (no el archivo grande)
git add data/raw/nuevo_dataset.csv.dvc .gitignore
git commit -m "Add new dataset"

# Subir datos a remote (si está configurado)
dvc push
```

---

## 🧠 Trabajar con MLflow

### **Ver experimentos en UI**

```bash
# Iniciar interfaz web
mlflow ui

# Abre: http://localhost:5000
```

**En la UI puedes**:
- 📊 Comparar experimentos lado a lado
- 📈 Ver gráficos de métricas
- 🔍 Inspeccionar parámetros y artefactos
- 📥 Descargar modelos entrenados

### **Gestionar Model Registry**

```bash
# Listar modelos registrados
python manage_registry.py list

# Ver versiones del modelo
python manage_registry.py versions obesity_classifier

# Comparar dos versiones
python manage_registry.py compare obesity_classifier 1 2

# Promover a Production
python manage_registry.py promote obesity_classifier 2 Production

# Asignar alias "champion"
python manage_registry.py alias obesity_classifier champion 2

# Encontrar mejor modelo
python manage_registry.py best obesity_classifier --metric accuracy
```

### **Usar modelo en código**

```python
import mlflow
import pandas as pd

# Cargar modelo por alias (recomendado)
model = mlflow.pyfunc.load_model("models:/obesity_classifier@champion")

# O por stage
model = mlflow.pyfunc.load_model("models:/obesity_classifier/Production")

# O por versión específica
model = mlflow.pyfunc.load_model("models:/obesity_classifier/2")

# Hacer predicción
new_data = pd.DataFrame([{
    'Age': 25, 'Weight': 80, 'Height': 1.70,
    'FCVC': 2, 'CH2O': 2, 'FAF': 3, ...
}])
prediction = model.predict(new_data)
print(prediction)  # ['Normal_Weight']
```

**Documentación completa**: [docs/MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md)

---

## 🌐 API REST con FastAPI

### **Iniciar servidor**

```bash
# Desarrollo (auto-reload)
uvicorn start_api:app --reload --port 8000

# Producción
uvicorn start_api:app --host 0.0.0.0 --port 8000
```

**Endpoints disponibles**: http://localhost:8000/docs (Swagger UI automática)

### **Realizar predicciones**

```bash
# Predicción individual
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 25,
    "Height": 1.70,
    "Weight": 80,
    "FCVC": 2,
    "CH2O": 2,
    "FAF": 3,
    "TUE": 1,
    "CALC": 2,
    "MTRANS": "Public_Transportation"
  }'

# Response:
# {
#   "prediction": "Normal_Weight",
#   "probabilities": {
#     "Normal_Weight": 0.89,
#     "Overweight_Level_I": 0.08,
#     "Obesity_Type_I": 0.03
#   }
# }
```

### **Test automatizado de API**

```bash
# Ejecutar todos los tests de API
python test_api.py

# Output:
# ✅ Health check passed
# ✅ Single prediction passed
# ✅ Batch prediction passed
# ✅ Model info passed
# 4/4 tests successful
```

🧪 Pruebas automáticas

:: Ejecutar tests unitarios y de integración:
```bash
    pytest -q

🔁 Integración continua (CI)
Cada push al repositorio activa el flujo de CI definido en .github/workflows/ci.yml, que ejecuta:
```bash
    dvc repro
    pytest
    dvc metrics show

Esto asegura reproducibilidad y validación automática de los cambios.

🧑‍💻 Roles simulados en el flujo MLOps
Rol         	    Responsabilidad
Data Engineer	    Limpieza, integración y versionado de datos.
Data Scientist	    Exploración, modelado y métricas.
ML Engineer	        Empaquetado, entrenamiento y evaluación automatizada.
DevOps Engineer	    Automatización, CI/CD y monitoreo de despliegues.

📈 Métricas base (ejemplo)
Métrica	        Valor
Accuracy	    0.972
F1-macro	    0.961
Recall	        0.958
Precision	    0.965

Los resultados pueden variar según los hiperparámetros definidos en params.yaml.

🧭 Lecciones aprendidas
- La reproducibilidad es la clave del éxito en Machine Learning.
- Versionar datos y modelos permite auditar y mejorar continuamente.
- MLflow y DVC simplifican el control de experimentos.
- La colaboración entre roles técnicos asegura trazabilidad y calidad.
- CI/CD garantiza estabilidad en cada versión del pipeline.

🧩 Próximos pasos
- Incorporar monitoreo con EvidentlyAI.
- Automatizar el reentrenamiento por data drift.
- Contenerizar todo el flujo con Docker Compose.
- Desplegar en nube (AWS / Azure / GCP).

📚 Referencias
- Introducing MLOps – Mark Treveil, O’Reilly Media (2020).
- Machine Learning Engineering in Action – Ben Wilson (2022).
- Documentación oficial: DVC
 | MLflow

🏁 Conclusión

“La reproducibilidad es el puente entre la experimentación y la producción.”
Este proyecto demuestra cómo implementar un flujo MLOps real: automatizado, trazable y escalable.

---

📘 **Instrucciones finales:**
1. Copia todo el bloque de arriba.
2. Crea un nuevo archivo llamado `README.md` en tu repositorio local (`mlops-reproducible/`).
3. Pega el contenido y guarda.
4. Ejecuta:
   ```bash
   git add README.md
   git commit -m "Add professional README for MLOps reproducible project"
   git push origin main
````
