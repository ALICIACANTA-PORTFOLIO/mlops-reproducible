# MLOps Reproducible - Context for AI Assistants

## 🎯 Project Overview

**Project Name**: MLOps Obesity Classification Pipeline  
**Type**: Machine Learning Operations (MLOps) Portfolio Project  
**Goal**: Demonstrate enterprise-grade MLOps practices with reproducible ML pipeline  
**Status**: Production-Ready ✅  
**Date**: October 2025

---

## 📊 Project Summary

This project implements a **complete MLOps pipeline** for obesity classification using RandomForest classifier, achieving **92.66% accuracy**. It showcases advanced ML engineering practices including:

- ✅ **MLflow Model Registry** with automatic staging transitions
- ✅ **DVC Pipeline** for data versioning and reproducibility
- ✅ **Model Signatures** for production validation
- ✅ **FastAPI REST API** for model serving
- ✅ **Automated Testing** with pytest (3/3 passing)
- ✅ **Reproducibility Guarantee** (0.0000 difference between runs)

---

## 🏗️ Architecture

### **Hybrid Architecture Pattern**

The project uses a **unique dual-approach architecture**:

1. **`src/` - CLI Modules** (Production Pattern)
   - DVC pipeline integration
   - Modular, independent scripts
   - CI/CD ready
   - MLflow auto-tracking

2. **`mlops/` - Python API** (Development Pattern)
   - Interactive development
   - Clean Python interface
   - Notebook-friendly
   - Programmatic configuration

---

## 📁 Project Structure

```
mlops-reproducible/
├── src/                           # Production CLI modules
│   ├── data/
│   │   ├── preprocess.py         # Data cleaning and validation
│   │   └── make_features.py      # Feature engineering
│   ├── models/
│   │   ├── train.py              # Training + MLflow Registry
│   │   ├── evaluate.py           # Model evaluation
│   │   └── predict.py            # Batch predictions
│   └── serving/
│       └── api.py                # FastAPI application
│
├── mlops/                         # Development Python API
│   ├── config.py                 # Configuration management
│   ├── dataset.py                # Data processing
│   ├── features.py               # Feature engineering
│   ├── modeling.py               # Training and evaluation
│   └── train.py                  # Main pipeline
│
├── manage_registry.py            # MLflow Registry CLI tool
├── start_api.py                  # API startup script
├── test_api.py                   # API testing script
├── run_mlops.py                  # Unified interface
│
├── data/                         # DVC-versioned data
│   ├── raw/                      # Original dataset (2.1MB)
│   ├── interim/                  # Intermediate processing
│   └── processed/                # Final features
│
├── models/                       # Trained models
│   ├── features/                 # Feature artifacts (encoder, scaler)
│   └── mlflow_model/             # MLflow packaged model
│
├── mlruns/                       # MLflow tracking data
├── reports/                      # Metrics and figures
├── tests/                        # Pytest test suite
├── docs/                         # Documentation
│
├── params.yaml                   # Central configuration
├── dvc.yaml                      # DVC pipeline definition
├── requirements.txt              # Python dependencies
└── conda.yaml                    # Conda environment
```

---

## 🔧 Technology Stack

| Category | Tool | Version | Purpose |
|----------|------|---------|---------|
| **Language** | Python | 3.10.19 | Core language |
| **ML Framework** | scikit-learn | 1.3.2 | RandomForest model |
| **Experiment Tracking** | MLflow | 2.8.1 | Tracking + Registry |
| **Data Versioning** | DVC | 3.30.0 | Data + pipeline versioning |
| **API Framework** | FastAPI | 0.104.1 | REST API serving |
| **Testing** | Pytest | 7.4.3 | Automated testing |
| **Environment** | Conda | - | Python 3.10 management |

---

## 🎯 Key Features

### **1. MLflow Model Registry (Enterprise-Grade)**

**Location**: `manage_registry.py`, `src/models/train.py`

**Features**:
- Automatic model registration during training
- Model signatures with schema validation
- Automatic staging transitions (accuracy >= 0.85 → Staging)
- Semantic aliases (champion, challenger, baseline)
- Version comparison and metrics tracking
- CLI tool for registry management

**Usage**:
```bash
# List models
python manage_registry.py list

# View versions with metrics
python manage_registry.py versions obesity_classifier

# Promote to Production
python manage_registry.py promote obesity_classifier 2 Production

# Assign alias
python manage_registry.py alias obesity_classifier champion 2
```

**Code Pattern**:
```python
# In src/models/train.py
signature = infer_signature(X_train, predictions)
mlflow.sklearn.log_model(
    model, "model",
    signature=signature,
    input_example=input_example,
    registered_model_name="obesity_classifier"
)

# Auto-transition to Staging
if accuracy >= staging_threshold:
    client.transition_model_version_stage(
        name="obesity_classifier",
        version=model_version,
        stage="Staging"
    )
```

---

### **2. DVC Pipeline (Reproducibility)**

**Location**: `dvc.yaml`

**Pipeline Stages**:
1. `preprocess` - Data cleaning and validation
2. `make_features` - Feature engineering
3. `train` - Model training with MLflow

**Features**:
- Automatic dependency tracking
- Cache intelligent re-execution
- Data versioning (2.1MB → 165 bytes metadata)
- Reproducibility guarantee (0.0000 difference)

**Usage**:
```bash
dvc repro              # Run full pipeline
dvc dag                # View dependency graph
dvc metrics show       # Display metrics
dvc metrics diff       # Compare versions
```

---

### **3. FastAPI REST API**

**Location**: `src/serving/api.py`, `start_api.py`

**Endpoints**:
- `GET /` - Health check
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions
- `GET /model_info` - Model metadata

**Features**:
- Automatic Swagger UI (`/docs`)
- CORS enabled
- Input validation with Pydantic
- Model loading from MLflow
- Feature preprocessing pipeline

**Usage**:
```bash
# Start API
python start_api.py --reload

# Test
python test_api.py
# Or open: http://localhost:8000/docs
```

---

### **4. Automated Testing**

**Location**: `tests/`

**Test Suite**:
- `test_data_validation.py` - Data quality and reproducibility
- `test_advanced_framework.py` - Advanced validation framework
- `test_api.py` - API endpoints testing

**Features**:
- Reproducibility validation (0.0000 difference)
- Data schema validation
- Feature engineering validation
- API endpoint testing (4/4 passing)

**Usage**:
```bash
pytest tests/ -v
# ✅ 3/3 tests passing
```

---

## 📊 Model Details

**Model Type**: Random Forest Classifier  
**Target**: 7 obesity categories  
**Features**: 32 engineered features  
**Dataset**: 2,087 samples  

**Performance**:
- **Accuracy**: 92.66%
- **F1-Score (macro)**: 92.51%
- **F1-Score (weighted)**: 92.66%
- **Precision**: 93.03%
- **Recall**: 92.66%

**Top Classes Performance**:
- Obesity_Type_III: Precision 0.98, Recall 0.96, F1 0.97
- Overweight_Level_II: Precision 0.90, Recall 0.94, F1 0.92
- Normal_Weight: Precision 0.95, Recall 0.93, F1 0.94

---

## 🔄 Development Workflow

### **Typical Workflow**:

```bash
# 1. Activate environment
conda activate mlops-reproducible

# 2. Train model (auto-registers in MLflow)
python src/models/train.py
# Or: dvc repro

# 3. Verify registry
python manage_registry.py versions obesity_classifier

# 4. Start API (Terminal 1)
python start_api.py --reload

# 5. Test API (Terminal 2)
python test_api.py

# 6. Run tests
pytest tests/ -v
```

---

## ⚙️ Configuration

**Central Config**: `params.yaml`

**Key Sections**:
```yaml
data:
  raw_path: data/raw/ObesityDataSet_raw_and_data_sinthetic.csv
  processed_path: data/processed/features.csv

random_forest:
  n_estimators: 100
  max_depth: 10
  random_state: 42  # For reproducibility

mlflow:
  experiment_name: obesity_classification_v2
  tracking_uri: ./mlruns
  registered_model_name: obesity_classifier
  staging_threshold: 0.85  # Auto-promote to Staging
  
  tags:
    project: mlops-reproducible
    team: data-science
    environment: development
```

---

## 🎨 Code Patterns

### **Pattern 1: MLflow Experiment Tracking**

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="obesity_training"):
    # Log parameters
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    })
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision": precision
    })
    
    # Log model with registry
    signature = infer_signature(X_train, predictions)
    mlflow.sklearn.log_model(
        model, "model",
        signature=signature,
        registered_model_name="obesity_classifier"
    )
```

### **Pattern 2: Feature Engineering Pipeline**

```python
from mlops import features

# Load and process data
df = pd.read_csv("data/processed/features.csv")

# Apply transformations
df_transformed = features.apply_transformations(df)

# Encode categorical
df_encoded = features.encode_categorical(df_transformed)

# Scale numerical
df_scaled = features.scale_numerical(df_encoded)
```

### **Pattern 3: Model Loading from Registry**

```python
import mlflow.pyfunc

# Load by alias (recommended)
model = mlflow.pyfunc.load_model("models:/obesity_classifier@champion")

# Load by stage
model = mlflow.pyfunc.load_model("models:/obesity_classifier/Production")

# Load by version
model = mlflow.pyfunc.load_model("models:/obesity_classifier/2")
```

---

## 🚨 Important Guidelines

### **For Code Generation**:

1. **Always use random_state=42** for reproducibility
2. **Log everything to MLflow** (params, metrics, artifacts)
3. **Use type hints** in all functions
4. **Add docstrings** to modules and functions
5. **Follow PEP 8** style guide
6. **Use pathlib.Path** instead of os.path
7. **Validate inputs** with Pydantic schemas

### **For Testing**:

1. **Tests must be deterministic** (same results every time)
2. **Use fixtures** for common setup
3. **Mock external dependencies** (API calls, file I/O)
4. **Test reproducibility** (compare model outputs)
5. **Verify data schemas** before processing

### **For MLflow**:

1. **Always infer signatures** before logging models
2. **Include input_example** for documentation
3. **Tag models** with metadata (date, type, status)
4. **Use registered_model_name** for registry
5. **Set appropriate stages** (None, Staging, Production)

### **For API Development**:

1. **Use Pydantic models** for request/response
2. **Add field validation** (min, max, description)
3. **Include error handling** with HTTPException
4. **Document endpoints** with docstrings
5. **Test all endpoints** before deployment

---

## 📚 Documentation Structure

```
docs/
├── MODEL_REGISTRY.md           # Complete registry guide
├── IMPLEMENTATION_SUMMARY.md   # Project summary and status
├── ARCHITECTURE.md             # System architecture
├── DEPLOYMENT.md               # Deployment instructions
├── MLOPS_INTEGRATION.md        # MLOps tools integration
├── TECHNICAL_GUIDE.md          # Technical deep dive
├── TESTING_REPORT.md           # Test results
├── API_DOCUMENTATION.md        # API reference
├── TROUBLESHOOTING.md          # Common issues
└── 1.4_books/README.md         # ML books analysis
```

---

## 🔗 External References

**Based on Best Practices from**:
- 📖 "Machine Learning Engineering with MLflow" (Chapters 5-6)
- 📖 "Machine Learning Design Patterns"
- 🔗 MLflow Documentation: https://mlflow.org/docs/latest/
- 🔗 DVC Documentation: https://dvc.org/doc
- 🔗 FastAPI Documentation: https://fastapi.tiangolo.com/

---

## 🎯 Portfolio Highlights

**What makes this project stand out**:

1. ✅ **Enterprise-Grade Model Registry**
   - Not just "saving models"
   - Complete lifecycle management
   - Automated workflows
   - Production validation

2. ✅ **Perfect Reproducibility**
   - DVC + MLflow + Git
   - 0.0000 difference between runs
   - Deterministic pipeline

3. ✅ **Hybrid Architecture**
   - CLI for production (DVC)
   - API for development (Python)
   - Unified interface

4. ✅ **Production-Ready API**
   - FastAPI with Swagger
   - Input validation
   - Error handling
   - Automated tests

5. ✅ **Clean Code**
   - Type hints
   - Docstrings
   - PEP 8 compliant
   - Well-tested (3/3 passing)

---

## 🚀 Quick Commands Reference

```bash
# Environment
conda activate mlops-reproducible

# Training
python src/models/train.py
dvc repro

# Registry
python manage_registry.py list
python manage_registry.py versions obesity_classifier

# API
python start_api.py --reload
python test_api.py

# Tests
pytest tests/ -v

# MLflow UI
mlflow ui  # http://localhost:5000

# DVC
dvc repro
dvc dag
dvc metrics show
```

---

## 📝 Common Tasks

### **Add New Feature**:
1. Update `src/data/make_features.py`
2. Update feature list in `params.yaml`
3. Re-run: `dvc repro`
4. Verify reproducibility: `pytest tests/test_data_validation.py::test_reproducibility`

### **Retrain Model**:
1. Modify hyperparameters in `params.yaml`
2. Run: `python src/models/train.py`
3. Check MLflow: `python manage_registry.py versions obesity_classifier`
4. Compare: `python manage_registry.py compare obesity_classifier 1 2`

### **Deploy New Version**:
1. Train and register: `python src/models/train.py`
2. Verify metrics: `python manage_registry.py versions obesity_classifier`
3. Promote: `python manage_registry.py promote obesity_classifier 2 Production`
4. Assign alias: `python manage_registry.py alias obesity_classifier champion 2`
5. Restart API: `python start_api.py --reload`

---

## ⚠️ Common Pitfalls

1. **❌ Forgetting to activate conda environment**
   - Always: `conda activate mlops-reproducible`

2. **❌ Running test_api.py without starting API**
   - Start API first: `python start_api.py --reload`

3. **❌ Using wrong uvicorn command**
   - Wrong: `uvicorn start_api:app`
   - Right: `uvicorn src.serving.api:app`

4. **❌ Not setting random_state**
   - Always use `random_state=42` for reproducibility

5. **❌ Forgetting to log to MLflow**
   - Always wrap training in `with mlflow.start_run():`

---

## 🎓 Learning Resources

- **Model Registry**: `docs/MODEL_REGISTRY.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
- **Books Analysis**: `docs/1.4_books/README.md`

---

**Last Updated**: October 23, 2025  
**Maintainer**: ALICIACANTA-PORTFOLIO  
**License**: MIT  
**Status**: Production-Ready ✅
