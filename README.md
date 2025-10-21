# 🧠 mlops-reproducible

### _Pipeline completo de Machine Learning con enfoque MLOps_

![MLOps pipeline](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*BGgKwjJTRQ7BC8lX7aYf0w.png)

---

## 🚀 Descripción general

**mlops-reproducible** es un proyecto demostrativo que implementa un flujo **MLOps completo y reproducible**, basado en buenas prácticas de ingeniería de datos, ciencia de datos y DevOps.  
El objetivo es mostrar cómo transformar notebooks experimentales en un **pipeline automatizado, versionado y trazable** desde los datos hasta el modelo en producción.

> 🔍 Este proyecto es ideal como material educativo o portafolio profesional para demostrar competencias en MLOps.

---

## 🎯 Objetivos del proyecto

- Desarrollar un **pipeline modular y automatizado** de Machine Learning.
- Garantizar la **reproducibilidad** de experimentos mediante DVC y MLflow.
- Integrar control de versiones de código y datos con **Git + DVC**.
- Aplicar **CI/CD**, pruebas automáticas y registro de métricas.
- Desplegar el modelo entrenado con **FastAPI**.
- Documentar cada fase del flujo para aprendizaje y reutilización.

---

## 📂 Estructura del proyecto

mlops-reproducible/
├── notebooks/ # Fase exploratoria (EDA, entrenamiento, validación)
├── data/ # Datos versionados con DVC
│ ├── raw/ # Datos originales
│ ├── interim/ # Datos limpios
│ └── processed/ # Features finales
├── src/ # Código fuente modular
│ ├── data/ # preprocess, features
│ ├── models/ # train, evaluate, predict
│ ├── serving/ # API de despliegue (FastAPI)
│ └── utils/ # Funciones auxiliares (io, config)
├── models/ # Modelos entrenados (DVC)
├── reports/ # Métricas y figuras
├── tests/ # Pruebas automáticas
├── params.yaml # Configuración de hiperparámetros
├── dvc.yaml # Definición del pipeline
├── conda.yaml # Entorno reproducible
├── MLproject # Integración con MLflow
├── .github/workflows/ci.yml # Pipeline de CI/CD
└── README.md

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
