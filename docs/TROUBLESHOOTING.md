# Solución de Problemas Comunes - MLOps Pipeline

## ❌ Error: "Path already exists and is not empty"

### **Problema**

```
mlflow.exceptions.MlflowException: Path 'models\mlflow_model' already exists and is not empty
```

### **Causa**

MLflow no puede sobrescribir directorios de modelos existentes por defecto. Esto ocurre cuando:

- Ejecutas el entrenamiento múltiples veces
- El directorio `models/mlflow_model` ya contiene un modelo previo
- MLflow intenta guardar en una ruta que no está vacía

### **Solución Implementada**

En `src/models/train.py`, agregamos limpieza automática del directorio:

```python
# Guardar y loggear el modelo
Path(model_dir).mkdir(parents=True, exist_ok=True)
local_model_path = Path(model_dir) / "mlflow_model"

# Si el directorio existe, eliminarlo primero para evitar conflictos
if local_model_path.exists():
    shutil.rmtree(local_model_path)

mlflow.sklearn.save_model(model, str(local_model_path))
mlflow.sklearn.log_model(model, artifact_path="model")
```

### **Alternativas**

1. **Nombres únicos por timestamp:**

```python
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
local_model_path = Path(model_dir) / f"mlflow_model_{timestamp}"
```

2. **Usar MLflow run_id:**

```python
run_id = mlflow.active_run().info.run_id[:8]
local_model_path = Path(model_dir) / f"mlflow_model_{run_id}"
```

3. **Configuración en params.yaml:**

```yaml
model:
  save_path: "models/mlflow_model"
  overwrite: true # o false para preservar
```

## ✅ Resultado

- ✅ Pipeline ejecuta sin errores
- ✅ Modelos se guardan correctamente
- ✅ MLflow tracking funciona
- ✅ No hay conflictos en re-ejecuciones

## 🔄 Verificación

Para verificar que funciona:

```bash
# Ejecutar pipeline completo
python run_mlops.py cli pipeline

# Verificar modelo guardado
ls models/mlflow_model/

# Verificar en MLflow UI
mlflow ui
```

## 📝 Mejores Prácticas

1. **Siempre limpia modelos antiguos** en scripts de entrenamiento
2. **Usa versionado de modelos** en MLflow Model Registry
3. **Configura paths únicos** para experimentos paralelos
4. **Documenta la estrategia de versionado** en tu equipo

---

_Esta solución garantiza que el pipeline MLOps sea robusto y reproducible._
