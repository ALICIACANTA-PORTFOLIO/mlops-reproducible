# Proyecto MLOps - Clasificación de Obesidad

Un proyecto de Machine Learning Operations (MLOps) limpio, ordenado y funcional para la clasificación de obesidad. Implementa las mejores prácticas de MLOps con un enfoque profesional y reproducible.

## 🎯 Objetivo

Crear un pipeline de machine learning reproducible y profesional para clasificar niveles de obesidad utilizando datos demográficos y hábitos alimentarios.

## � Estructura del Proyecto

```
mlops-reproducible/
├── mlops/                     # Paquete principal MLOps
│   ├── __init__.py           # Inicialización del módulo
│   ├── config.py             # Gestión de configuración
│   ├── dataset.py            # Procesamiento de datos
│   ├── features.py           # Ingeniería de características
│   ├── modeling.py           # Entrenamiento y evaluación
│   └── train.py             # Pipeline principal
├── data/                     # Datos versionados con DVC
│   ├── raw/                 # Datos originales
│   ├── interim/             # Datos procesados intermedio
│   └── processed/           # Datos finales procesados
├── models/                  # Modelos entrenados
├── notebooks/               # Notebooks de exploración
├── reports/                 # Reportes y métricas
├── docs/                    # Documentación
├── tests/                   # Pruebas unitarias
├── params.yaml              # Configuración principal
├── dvc.yaml                 # Pipeline DVC
└── requirements.txt         # Dependencias
```

## 🚀 Instalación Rápida

```bash
# Clonar el repositorio
git clone <repository-url>
cd mlops-reproducible

# Instalar dependencias
pip install -r requirements.txt

# Configurar DVC (opcional)
dvc pull
```

## 📊 Datos

**Dataset**: Clasificación de Obesidad

- **Muestras**: 2,089 registros
- **Características**: 16 variables (edad, peso, altura, hábitos)
- **Clases**: 7 niveles de obesidad
- **Fuente**: Datos sintéticos y reales de hábitos alimentarios

### Variables Principales:

- Demográficas: Edad, Género, Peso, Altura
- Hábitos: Frecuencia comidas, Consumo vegetales, Actividad física
- Comportamiento: Uso tecnología, Consumo alcohol, Transporte

## � Uso del Sistema - Interface Unificada

### **Opción A: Interface Unificada (Recomendada)**

```bash
# Pipeline completo - Enfoque CLI (DVC/Producción)
python run_mlops.py cli pipeline --params params.yaml

# Pipeline completo - Enfoque API (Desarrollo/Interactivo)
python run_mlops.py api pipeline --params params.yaml --experiment obesity_v1

# Pasos individuales - CLI
python run_mlops.py cli preprocess --input data/raw/dataset.csv --output data/interim/clean.csv
python run_mlops.py cli features --input data/interim/clean.csv --output data/processed/features.csv
python run_mlops.py cli train --data data/processed/features.csv
python run_mlops.py cli evaluate --data data/processed/features.csv
python run_mlops.py cli predict --model models/mlflow_model --features data/processed/features.csv

# Pasos individuales - API
python run_mlops.py api train --params params.yaml --experiment obesity_v1
python run_mlops.py api predict --model models/model.pkl --data data/new.csv --output predictions.csv
```

### **Opción B: Uso Directo - Enfoque CLI (`src/`)**

```bash
# Pipeline paso a paso
python src/data/preprocess.py --inp data/raw/dataset.csv --out data/interim/clean.csv
python src/data/make_features.py --inp data/interim/clean.csv --out data/processed/features.csv
python src/models/train.py --data data/processed/features.csv
python src/models/evaluate.py --data data/processed/features.csv
python src/models/predict.py --features_csv data/processed/features.csv

# O usando DVC (recomendado para producción)
dvc repro  # Ejecuta todo el pipeline definido en dvc.yaml
```

### **Opción C: Uso Directo - Enfoque API (`mlops/`)**

```python
# Entrenamiento básico
from mlops import train_model
results = train_model()
print(f"Accuracy: {results['test_metrics']['accuracy']:.3f}")

# Uso paso a paso
from mlops import Config, DataProcessor, FeatureEngineer, ModelTrainer

config = Config("params.yaml")
processor = DataProcessor(config.config)
engineer = FeatureEngineer(config.config)
trainer = ModelTrainer(config.config)

# Procesamiento completo
df = processor.load_data("data/raw/dataset.csv")
X_train, X_test, y_train, y_test = processor.split_data(df)
X_train_feat = engineer.create_features(X_train)
model = trainer.train_model(X_train_feat, y_train)

# Predicciones
from mlops import predict_batch
predictions = predict_batch(
    model_path="models/model.pkl",
    data_path="data/new_data.csv",
    output_path="predictions.csv"
)
```

## ⚙️ Configuración

El archivo `params.yaml` controla todos los aspectos del pipeline:

```yaml
# Datos
data:
  raw_path: "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
  target_column: "NObeyesdad"
  test_size: 0.2

# Características
features:
  selection_method: "mutual_info"
  n_features: 10
  create_interactions: true

# Modelo
model:
  hyperparameter_tuning: true
  cv_folds: 5
  parameters:
    n_estimators: 200
    max_depth: 15
    random_state: 42

# MLflow
mlflow:
  tracking: true
  experiment_name: "obesity_classification"
```

## 🔄 Pipeline MLOps Completo

### **Arquitectura de Doble Enfoque**

| Fase                       | `src/` CLI Modules          | `mlops/` Python API      | Propósito                         |
| -------------------------- | --------------------------- | ------------------------ | --------------------------------- |
| **1. Preprocessing**       | `src/data/preprocess.py`    | `mlops.DataProcessor`    | Limpieza, validación, imputación  |
| **2. Feature Engineering** | `src/data/make_features.py` | `mlops.FeatureEngineer`  | Codificación, escalado, selección |
| **3. Training**            | `src/models/train.py`       | `mlops.ModelTrainer`     | Entrenamiento con MLflow tracking |
| **4. Evaluation**          | `src/models/evaluate.py`    | `mlops.evaluate_model()` | Métricas y visualizaciones        |
| **5. Prediction**          | `src/models/predict.py`     | `mlops.predict_batch()`  | Inferencia batch/online           |

### **Funcionalidades Clave**

#### **Procesamiento (`src/data/`)**

- ✅ **Limpieza inteligente** - Eliminación duplicados, validación rangos
- ✅ **Imputación configurable** - Estrategias por tipo de variable
- ✅ **Normalización strings** - Consistencia en categóricas
- ✅ **Reporte de calidad** - JSON con métricas de limpieza

#### **Features (`src/data/make_features.py`)**

- ✅ **Encoding flexible** - OneHot, Ordinal, Label encoding
- ✅ **Scaling robusto** - Standard, MinMax, Robust scalers
- ✅ **Artefactos ML** - Guardado automático encoder/scaler
- ✅ **Target mapping** - Codificación consistente del target

#### **Entrenamiento (`src/models/`)**

- ✅ **MLflow integration** - Tracking automático experimentos
- ✅ **Múltiples algoritmos** - RandomForest, LogisticRegression
- ✅ **Métricas completas** - Accuracy, F1-macro, Precision, Recall
- ✅ **Visualizaciones** - Matriz confusión, feature importance

#### **API Python (`mlops/`)**

- ✅ **Interface limpia** - Uso programático sencillo
- ✅ **Pipeline integrado** - Una llamada, pipeline completo
- ✅ **Configuración flexible** - YAML + overrides programáticos
- ✅ **Interoperabilidad** - Compatible con notebooks

## 📈 Métricas de Rendimiento

### **Resultados Típicos (Dataset Obesidad)**

- **Accuracy**: 91.5% - 96.5%
- **F1-macro**: 91.2% - 96.2%
- **F1-weighted**: ~96.5%
- **Cross-validation**: 95.8% ± 1.2%

## 🛠️ Características Técnicas

### **Arquitectura Híbrida Única**

- ✅ **Doble enfoque** - CLI para producción, API para desarrollo
- ✅ **Interoperabilidad** - Ambas estructuras usan la misma configuración
- ✅ **Flexibilidad** - Elige el enfoque según tu caso de uso
- ✅ **Consistencia** - Resultados idénticos en ambos enfoques

### **Calidad de Código**

- ✅ **Type hints completos** - Tanto en `src/` como `mlops/`
- ✅ **Documentación exhaustiva** - Docstrings y comentarios detallados
- ✅ **Validaciones robustas** - Control de errores en ambos enfoques
- ✅ **Patrones profesionales** - Siguiendo mejores prácticas de MLOps

### **MLOps Completo**

- ✅ **Versionado de datos** - DVC integration nativa
- ✅ **Experiment tracking** - MLflow automático en ambos enfoques
- ✅ **Reproducibilidad** - Configuración centralizada `params.yaml`
- ✅ **Artefactos ML** - Guardado automático de encoders/scalers/modelos

## 🎯 Cuándo Usar Cada Enfoque

### **Usa `src/` CLI cuando:**

- 🏗️ **Configurando pipelines DVC** para producción
- 🤖 **Implementando CI/CD** automatizado
- 🔄 **Necesites ejecución modular** paso a paso
- 📊 **Trabajando con grandes volúmenes** de datos

### **Usa `mlops/` API cuando:**

- 📓 **Desarrollando en Jupyter** notebooks
- 🔬 **Experimentando interactivamente** con parámetros
- 🚀 **Prototipando rápidamente** nuevas ideas
- 🐍 **Integrando en aplicaciones** Python existentes

### **Usa Interface Unificada (`run_mlops.py`) cuando:**

- 🌟 **Quieras lo mejor de ambos** mundos
- 🔧 **Estés aprendiendo MLOps** y quieras flexibilidad
- 🎛️ **Necesites cambiar entre enfoques** dinámicamente
- 📚 **Estés documentando** o enseñando MLOps

## 📚 Documentación Adicional

- [`docs/GRADIO_INTERFACE.md`](docs/GRADIO_INTERFACE.md) - **🎨 Interfaz web amigable (Gradio UI)**
- [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md) - **🚀 API REST para servir el modelo (FastAPI)**
- [`docs/MLOPS_INTEGRATION.md`](docs/MLOPS_INTEGRATION.md) - **Guía completa de integración MLOps (DVC + MLflow + CI/CD)**
- [`docs/TECHNICAL_GUIDE.md`](docs/TECHNICAL_GUIDE.md) - Guía técnica detallada
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - Arquitectura del sistema
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) - Guía de despliegue
- [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb) - Análisis exploratorio

## 🚀 **API REST - Servir el Modelo**

### **Inicio Rápido de la API:**

```bash
# 1. Entrenar el modelo (si no está entrenado)
python run_mlops.py cli pipeline

# 2. Iniciar la API
python start_api.py --reload

# 3. Probar la API
python test_api.py
```

### **Endpoints Principales:**

- **GET** `/health` - Health check de la API
- **POST** `/predict` - Predicción individual
- **POST** `/predict/batch` - Predicciones en lote
- **GET** `/model/info` - Información del modelo
- **GET** `/docs` - Documentación interactiva (Swagger)

### **Ejemplo de Uso:**

```python
import requests

# Datos de ejemplo
data = {
    "Age": 28, "Height": 1.75, "Weight": 85,
    "Gender": "Male", "FAVC": "yes",
    # ... resto de campos
}

# Predicción
response = requests.post("http://127.0.0.1:8000/predict", json=data)
result = response.json()

print(f"Predicción: {result['prediction']}")
print(f"Confianza: {result['confidence']:.3f}")
print(f"Nivel de riesgo: {result['risk_level']}")
```

**📋 Documentación completa**: [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md)

## 🎨 **Interfaz Web Gradio - UI Amigable**

### **Interfaz Visual para Usuarios Finales:**

```bash
# 1. Instalar dependencias (si no están)
pip install gradio plotly

# 2. Entrenar el modelo (si no está entrenado)
python run_mlops.py cli pipeline

# 3. Iniciar interfaz Gradio
python start_gradio.py

# 4. Abrir en navegador
# http://127.0.0.1:7860
```

### **🌟 Características de la Interfaz:**

- **🎯 Formulario interactivo** - Sliders, dropdowns, radio buttons
- **📊 Visualizaciones en tiempo real** - Gráficos Plotly interactivos
- **💡 Recomendaciones personalizadas** - Consejos específicos por resultado
- **🔬 Casos de ejemplo** - Datos predefinidos para prueba rápida
- **📱 Diseño responsivo** - Se adapta a móviles y tablets
- **🎨 Interfaz profesional** - Tema personalizado y atractivo

### **🎮 Ejemplo de Uso Gradio:**

1. **📝 Llenar formulario** - Edad, peso, altura, hábitos
2. **🎯 Hacer clic** en "Analizar mi Estado de Salud"
3. **📊 Ver resultados** - Clasificación + confianza + gráficos
4. **💡 Leer consejos** - Recomendaciones personalizadas

### **🎯 Comparación: FastAPI vs Gradio**

| Interfaz    | Audiencia        | Formato   | Uso Principal        |
| ----------- | ---------------- | --------- | -------------------- |
| **FastAPI** | Desarrolladores  | JSON/REST | Integración con apps |
| **Gradio**  | Usuarios finales | Web UI    | Demos y prototipos   |

**📋 Documentación completa**: [`docs/GRADIO_INTERFACE.md`](docs/GRADIO_INTERFACE.md)

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

## 🧩 Flujo MLOps implementado

````mermaid
flowchart LR
    A[Datos brutos] --> B[Limpieza y validación]
    B --> C[Feature Engineering]
    C --> D[Entrenamiento del modelo]
    D --> E[Evaluación y registro (MLflow)]
    E --> F[Despliegue (FastAPI / Docker)]
    F --> G[Monitoreo y retroalimentación]

⚙️ Tecnologías utilizadas

| Categoría                  | Herramienta           | Propósito                                  |
| -------------------------- | --------------------- | ------------------------------------------ |
| Control de versiones       | **Git / GitHub**      | Versionado de código y CI/CD               |
| Versionado de datos        | **DVC**               | Control y trazabilidad de datasets         |
| Tracking de experimentos   | **MLflow**            | Registro de parámetros, métricas y modelos |
| Automatización de pipeline | **DVC Pipelines**     | Ejecución reproducible de etapas           |
| Despliegue de API          | **FastAPI + Uvicorn** | Servir modelos en producción               |
| Reproducibilidad           | **Conda / MLproject** | Control de entornos                        |
| Testing                    | **Pytest**            | Verificación funcional del pipeline        |
| CI/CD                      | **GitHub Actions**    | Automatización de ejecuciones y pruebas    |


⚙️ Instalación y configuración

1️⃣ Clonar el repositorio
git clone https://github.com/ALICIACANTA-MNA/mlops-reproducible.git
cd mlops-reproducible

2️⃣ Crear entorno Conda
conda env create -f conda.yaml
conda activate mlops-reproducible

3️⃣ Inicializar DVC
dvc init
dvc pull  # si existe almacenamiento remoto configurado

🧮 Ejecución del pipeline

Ejecutar todas las fases definidas en el archivo dvc.yaml:
```bash
    dvc repro

:: Visualizar métricas:

dvc metrics show


:: Repetir con nuevos parámetros:

vi params.yaml
dvc repro && dvc metrics diff


🧠 Registro y seguimiento de experimentos (MLflow)

:: Iniciar interfaz gráfica de MLflow:

mlflow ui

Abrir en el navegador: http://localhost:5000

Desde allí podrás:
- Visualizar métricas comparativas
- Registrar artefactos y modelos
- Seguir el historial de ejecuciones

🌐 Despliegue local con FastAPI

:: Iniciar el servicio:
```bash
    uvicorn src.serving.api:app --reload --port 8080

:: Realizar una predicción:
```bash
    curl -X POST "http://127.0.0.1:8080/predict" -H "Content-Type: application/json" \
     -d '{"Age":25,"Weight":80,"Height":1.70,"CH2O":2,"FAF":3,"FCVC":2,"TUE":1}'

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
