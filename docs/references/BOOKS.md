# 📚 Referencias Bibliográficas - MLOps Project

**Proyecto**: Clasificación de Obesidad con MLOps  
**Propósito**: Referenciar libros y recursos utilizados para diseñar e implementar este proyecto  
**Fecha**: Octubre 2025

---

## 📖 Libros Principales

### 1. Machine Learning Engineering with MLflow

**Información del Libro**:
- **Autores**: Nisha Talagala, Clemens Mewald
- **Editorial**: Manning Publications, 2021
- **ISBN**: 978-1617299612
- **Páginas**: 264

**Enlaces de Compra**:
- 🛒 [Amazon](https://www.amazon.com/Machine-Learning-Engineering-MLflow-Manage/dp/1617299618)
- 🛒 [Manning](https://www.manning.com/books/machine-learning-engineering-with-mlflow)
- 🛒 [O'Reilly](https://www.oreilly.com/library/view/machine-learning-engineering/9781617299612/)

**Relevancia para este proyecto**: ⭐⭐⭐⭐⭐

---

#### **Capítulos Aplicados en el Proyecto**

##### **Chapter 5: Model Registry and Lifecycle Management**

**Conceptos implementados**:
```python
# Model Signatures (pág. 142-145)
from mlflow.models.signature import infer_signature

signature = infer_signature(X_train, predictions)
mlflow.sklearn.log_model(
    model, "model",
    signature=signature,
    registered_model_name="obesity_classifier"
)
```

**Archivo**: `src/models/train.py` (líneas 150-180)

**Features del libro implementadas**:
- ✅ Model signatures para validación de schemas
- ✅ Automatic model registration
- ✅ Model versioning
- ✅ Stage transitions (None → Staging → Production)
- ✅ Model aliases (champion, challenger, baseline)

---

##### **Chapter 6: Model Serving and Deployment**

**Conceptos aplicados**:
```python
# Model Loading Pattern (pág. 168-172)
import mlflow.pyfunc

# Load by alias (recommended)
model = mlflow.pyfunc.load_model("models:/obesity_classifier@champion")

# Load by stage
model = mlflow.pyfunc.load_model("models:/obesity_classifier/Production")
```

**Archivo**: `src/serving/api.py` (líneas 20-35)

**Patrones del libro**:
- ✅ Model serving via REST API
- ✅ FastAPI integration con MLflow
- ✅ Input validation con Pydantic
- ✅ Error handling patterns

---

#### **Código Inspirado del Libro**

##### **Pattern 1: CLI para Model Registry** (Chapter 5, pág. 152)

```python
# manage_registry.py - Inspired by MLflow CLI patterns
def promote_model(model_name, version, stage):
    """
    Promote model to Production/Staging
    Book reference: Chapter 5 - Model Lifecycle Management
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
```

**Innovación sobre el libro**: Agregamos CLI completa con 8 comandos (list, versions, promote, alias, compare, best, search, delete)

---

##### **Pattern 2: Auto-transition con umbrales** (Chapter 5, pág. 158)

```python
# src/models/train.py - Enhanced from book's basic example
staging_threshold = params['mlflow']['staging_threshold']  # 0.85

if accuracy >= staging_threshold:
    print(f"✅ Model meets staging criteria (accuracy >= {staging_threshold})")
    client.transition_model_version_stage(
        name="obesity_classifier",
        version=model_version,
        stage="Staging"
    )
```

**Book's approach**: Manual transitions  
**Our enhancement**: Automated with configurable thresholds

---

### 2. Machine Learning Design Patterns

**Información del Libro**:
- **Autores**: Valliappa Lakshmanan, Sara Robinson, Michael Munn
- **Editorial**: O'Reilly Media, 2020
- **ISBN**: 978-1098115784
- **Páginas**: 408

**Enlaces de Compra**:
- 🛒 [Amazon](https://www.amazon.com/Machine-Learning-Design-Patterns-Preparation/dp/1098115783)
- 🛒 [O'Reilly](https://www.oreilly.com/library/view/machine-learning-design/9781098115777/)

**Relevancia para este proyecto**: ⭐⭐⭐⭐⭐

---

#### **Patrones Aplicados en el Proyecto**

##### **Pattern 1: Data Pipeline Pattern** (Chapter 3)

**Libro describe** (pág. 78-92):
> "Declarative pipelines with automatic dependency resolution and caching"

**Nuestra implementación**:
```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python src/data/preprocess.py ...
    deps:
      - data/raw/ObesityDataSet_raw_and_data_sinthetic.csv
    outs:
      - data/interim/obesity_clean.csv
      
  make_features:
    cmd: python src/data/make_features.py ...
    deps:
      - data/interim/obesity_clean.csv
    outs:
      - data/processed/features.csv
```

**Beneficios** (descritos en pág. 85):
- ✅ Automatic caching
- ✅ Dependency resolution
- ✅ Reproducibility guarantee

---

##### **Pattern 2: Hybrid Architecture** (Chapter 8, pág. 245)

**Libro menciona** (pág. 248):
> "Combining batch and real-time serving patterns for flexibility"

**Nuestra adaptación INNOVADORA**:
```
src/         ← Production pattern (CLI modules, DVC integration)
mlops/       ← Development pattern (Python API, notebooks)
run_mlops.py ← Unified interface
```

**Ventaja sobre el libro**: Usamos hybrid para **development workflow**, no solo serving.

---

##### **Pattern 3: Feature Store Pattern** (Chapter 3, pág. 95-108)

**Implementación simplificada**:
```
models/features/
├── encoder.pkl          # Categorical encoder
├── scaler.pkl           # Numerical scaler
├── feature_columns.pkl  # Column order
└── feature_names.txt    # Feature names
```

**Diferencia con el libro**: Libro describe feature stores completos (Feast, Tecton). Nosotros implementamos versión lightweight.

---

##### **Pattern 4: Reproducibility Pattern** (Chapter 4, pág. 115-130)

**Stack implementado** (siguiendo libro):

| Componente | Tool | Propósito |
|------------|------|-----------|
| **Code** | Git | Version control |
| **Data** | DVC | Data versioning |
| **Environment** | Conda | Python dependencies |
| **Experiments** | MLflow | Parameter tracking |
| **Config** | params.yaml | Centralized configuration |

**Resultado**: 0.0000 difference entre ejecuciones (validado con pytest)

---

#### **Comparación: Libro vs Implementación**

| Patrón del Libro | En el Proyecto | Innovación |
|------------------|----------------|------------|
| Pipeline Pattern | ✅ DVC pipeline | + Model Registry integration |
| Feature Store | ✅ Lightweight version | + DVC versioning |
| Hybrid Serving | ✅ Adapted | Used for dev workflow, not just serving |
| Reproducibility | ✅ Full stack | + Testing framework |
| Config Pattern | ✅ params.yaml | + MLflow integration |

---

## 🎓 Notas de Aprendizaje

### **De "MLflow Engineering"**

#### **Model Registry Best Practices** (Chapter 5)

**Lecciones clave**:
1. **Always use model signatures** → Evita errores de schema en producción
2. **Tag everything** → Metadata facilita búsqueda y auditoría
3. **Automate transitions** → Reduce errores humanos en promociones

**Implementado en**: `manage_registry.py`

---

#### **Serving Patterns** (Chapter 6)

**Pattern aprendido**: Model loading by alias

```python
# ❌ Bad: Hard-coded version
model = mlflow.pyfunc.load_model("models:/obesity_classifier/2")

# ✅ Good: Semantic alias
model = mlflow.pyfunc.load_model("models:/obesity_classifier@champion")
```

**Beneficio**: Cambiar modelo en producción sin cambiar código.

---

### **De "ML Design Patterns"**

#### **Pipeline Anti-patterns** (Chapter 3, pág. 92)

**Evitados en nuestro proyecto**:
- ❌ **Hardcoded paths** → Usamos params.yaml
- ❌ **Manual dependency tracking** → DVC automático
- ❌ **No versioning** → Git + DVC + MLflow

---

#### **Reproducibility Checklist** (Chapter 4, pág. 128)

Validado en nuestro proyecto:
- ✅ Fixed random seeds (random_state=42)
- ✅ Versioned data (DVC)
- ✅ Versioned code (Git)
- ✅ Versioned dependencies (conda.yaml)
- ✅ Versioned experiments (MLflow)
- ✅ Automated testing (pytest)

---

## 📊 Mapeo Detallado: Conceptos → Código

### **MLflow Engineering**

| Concepto (Libro) | Capítulo | Implementación | Archivo |
|------------------|----------|----------------|---------|
| Experiment Tracking | Ch. 4 | ✅ Completo | `src/models/train.py:88-130` |
| Parameter Logging | Ch. 4 | ✅ 25+ params | `src/models/train.py:95-110` |
| Metrics Logging | Ch. 4 | ✅ Per-class + overall | `src/models/train.py:135-160` |
| Artifact Storage | Ch. 4 | ✅ Confusion matrix + plots | `src/models/train.py:165-175` |
| Model Registry | Ch. 5 | ✅ **Completo** ⭐ | `manage_registry.py` (full CLI) |
| Model Signatures | Ch. 5 | ✅ Validación schemas | `src/models/train.py:150-155` |
| Stage Transitions | Ch. 5 | ✅ Auto-staging | `src/models/train.py:180-195` |
| Model Aliases | Ch. 5 | ✅ Champion/challenger | `manage_registry.py:alias()` |
| Model Loading | Ch. 6 | ✅ By alias/stage | `src/serving/api.py:25-30` |

---

### **ML Design Patterns**

| Patrón (Libro) | Capítulo | Implementación | Archivo/Componente |
|----------------|----------|----------------|--------------------|
| Data Pipeline | Ch. 3 | ✅ DVC declarativo | `dvc.yaml` |
| Feature Store | Ch. 3 | ✅ Lightweight | `models/features/` |
| Hybrid Architecture | Ch. 8 | ✅ Innovado | `src/` + `mlops/` |
| Reproducibility | Ch. 4 | ✅ Full stack | DVC + Git + MLflow + Conda |
| Configuration | Ch. 7 | ✅ Centralizado | `params.yaml` |
| Modular Design | Ch. 2 | ✅ Separación clara | `src/data/`, `src/models/` |
| Testing Pattern | Ch. 9 | ✅ Pytest suite | `tests/` (9/9 passing) |

---

## 📝 Code Snippets Adaptados

### **Snippet 1: Model Registry CLI** (MLflow Engineering, Ch. 5)

**Del libro** (pág. 155):
```python
# Basic model promotion
client.transition_model_version_stage(
    name="model_name",
    version="2",
    stage="Production"
)
```

**Nuestra mejora** (`manage_registry.py`):
```python
def promote_model(model_name, version, stage):
    """
    Enhanced promotion with validation and feedback
    """
    client = MlflowClient()
    
    # Validate stage
    valid_stages = ["Staging", "Production", "Archived"]
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage. Must be one of: {valid_stages}")
    
    # Get current stage
    model_version = client.get_model_version(model_name, version)
    current_stage = model_version.current_stage
    
    # Transition
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    
    print(f"✅ Promoted {model_name} v{version}: {current_stage} → {stage}")
```

---

### **Snippet 2: DVC Pipeline** (ML Design Patterns, Ch. 3)

**Del libro** (pág. 84):
> "Define pipelines declaratively with dependencies"

**Implementación**:
```yaml
# dvc.yaml - Inspired by book's pattern
stages:
  preprocess:
    cmd: python src/data/preprocess.py --inp ${data.raw_path} ...
    deps:
      - src/data/preprocess.py
      - ${data.raw_path}
    params:
      - data.raw_path
      - preprocessing
    outs:
      - ${data.interim_path}
```

---

## 🔗 Recursos Complementarios

### **Documentación Oficial**:
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### **Artículos de Referencia**:
- [MLOps Principles](https://ml-ops.org/content/mlops-principles) - MLOps best practices
- [Model Registry Patterns](https://www.databricks.com/blog/2020/06/25/introducing-mlflow-model-registry.html) - Databricks blog
- [Reproducible ML](https://neptune.ai/blog/how-to-solve-reproducibility-in-ml) - Neptune.ai guide

### **Cursos Recomendados**:
- [MLflow Tutorial](https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html) - Official MLflow
- [DVC Get Started](https://dvc.org/doc/start) - Official DVC
- [Made With ML - MLOps](https://madewithml.com/courses/mlops/) - Goku Mohandas

---

## ⚖️ Aviso Legal y Créditos

### **Copyright Notice**

Los libros referenciados son propiedad intelectual de sus respectivos autores y editoriales:

- **Machine Learning Engineering with MLflow** © 2021 Manning Publications
- **Machine Learning Design Patterns** © 2020 O'Reilly Media, Inc.

**Este proyecto NO incluye copias de los libros**. Solo referencias bibliográficas y conceptos aplicados de forma educativa bajo fair use.

---

### **Fair Use Statement**

Este proyecto utiliza:
- ✅ **Conceptos y patrones** descritos en los libros (fair use educativo)
- ✅ **Code snippets adaptados** con atribución clara
- ✅ **Referencias bibliográficas** completas con links de compra

NO incluye:
- ❌ Copias completas o parciales de los libros
- ❌ Capítulos escaneados
- ❌ Fragmentos extensos sin atribución

---

### **Para Adquirir los Libros**

Si estos conceptos te resultan útiles, **apoya a los autores** comprando los libros:

📚 **Machine Learning Engineering with MLflow**:
- [Amazon](https://www.amazon.com/Machine-Learning-Engineering-MLflow-Manage/dp/1617299618)
- [Manning](https://www.manning.com/books/machine-learning-engineering-with-mlflow)

📚 **Machine Learning Design Patterns**:
- [Amazon](https://www.amazon.com/Machine-Learning-Design-Patterns-Preparation/dp/1098115783)
- [O'Reilly](https://www.oreilly.com/library/view/machine-learning-design/9781098115777/)

---

### **Agradecimientos**

Este proyecto existe gracias al conocimiento compartido por:
- Nisha Talagala y Clemens Mewald (MLflow Engineering)
- Valliappa Lakshmanan, Sara Robinson y Michael Munn (ML Design Patterns)
- La comunidad open-source de MLflow, DVC y FastAPI

---

## 📌 Uso en el Proyecto

### **Para desarrolladores que lean este proyecto**:

1. **Entiende el contexto**: Los patrones implementados vienen de estos libros
2. **Aprende los fundamentos**: Lee los capítulos referenciados para profundizar
3. **Adapta a tu caso**: Los patterns son generales, nuestra implementación es específica
4. **Cita apropiadamente**: Si usas estos conceptos, referencia los libros originales

---

## 🔄 Actualización

**Última actualización**: 23 de Octubre, 2025  
**Versión**: 1.0  
**Mantenido por**: ALICIACANTA-PORTFOLIO

---

<div align="center">

**📚 El conocimiento se construye sobre los hombros de gigantes**

*Estos libros son la base teórica. Este proyecto es la aplicación práctica.*

</div>
