# Proyecto MLOps - Clasificación de Obesidad

Un proyecto de Machine Learning Operations (MLOps) limpio, ordenado y funcional para la clasificación de obesidad. Implementa las mejores prácticas de MLOps con un enfoque profesional y reproducible.

#
#
#
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
