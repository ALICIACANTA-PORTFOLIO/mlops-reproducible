# API de Clasificación de Obesidad - MLOps

## 🎯 Descripción

API REST para clasificación de niveles de obesidad basada en datos demográficos y hábitos de salud. Construida con FastAPI y integrada con el pipeline MLOps del proyecto.

## 🚀 Inicio Rápido

### 1. Entrenar el modelo (si no lo has hecho)

```bash
python run_mlops.py cli pipeline
```

### 2. Iniciar la API

```bash
# Opción 1: Script personalizado
python start_api.py --reload

# Opción 2: Uvicorn directo
uvicorn src.serving.api:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Probar la API

```bash
# Ejecutar tests automatizados
python test_api.py

# O abrir documentación interactiva
# http://127.0.0.1:8000/docs
```

## 📋 Endpoints Disponibles

### 🔍 **Health Check**

- **GET** `/health`
- **Descripción**: Verificar estado de la API y modelo
- **Respuesta**:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "encoder_loaded": true,
  "scaler_loaded": true
}
```

### 🎯 **Predicción Individual**

- **POST** `/predict`
- **Descripción**: Clasificar nivel de obesidad para una persona
- **Request Body**:

```json
{
  "Age": 28,
  "Height": 1.75,
  "Weight": 85,
  "FCVC": 2,
  "NCP": 3,
  "CH2O": 2,
  "FAF": 1,
  "TUE": 2,
  "Gender": "Male",
  "family_history_with_overweight": "yes",
  "FAVC": "yes",
  "CAEC": "Sometimes",
  "SMOKE": "no",
  "SCC": "no",
  "CALC": "Sometimes",
  "MTRANS": "Public_Transportation"
}
```

- **Respuesta**:

```json
{
  "prediction": "Overweight_Level_I",
  "prediction_proba": {
    "Normal_Weight": 0.15,
    "Overweight_Level_I": 0.65,
    "Obesity_Type_I": 0.2
  },
  "confidence": 0.65,
  "risk_level": "Medium"
}
```

### 📊 **Predicción en Lote**

- **POST** `/predict/batch`
- **Descripción**: Clasificar múltiples personas de una vez
- **Request Body**:

```json
{
  "data": [
    {
      /* datos persona 1 */
    },
    {
      /* datos persona 2 */
    },
    {
      /* datos persona 3 */
    }
  ]
}
```

- **Respuesta**:

```json
{
  "predictions": [
    /* lista de predicciones */
  ],
  "summary": {
    "total_predictions": 3,
    "risk_distribution": {
      "Low": 1,
      "Medium": 1,
      "High": 1
    },
    "average_confidence": 0.78
  }
}
```

### 📋 **Información del Modelo**

- **GET** `/model/info`
- **Descripción**: Obtener información sobre el modelo cargado
- **Respuesta**:

```json
{
  "model_type": "RandomForestClassifier",
  "model_loaded": true,
  "num_features": 16,
  "artifacts_status": {
    "encoder": true,
    "scaler": true,
    "metadata": true
  }
}
```

## 📝 Especificación de Datos de Entrada

### Variables Numéricas:

| Campo    | Descripción                  | Rango   | Ejemplo |
| -------- | ---------------------------- | ------- | ------- |
| `Age`    | Edad en años                 | 10-90   | 28      |
| `Height` | Altura en metros             | 1.2-2.2 | 1.75    |
| `Weight` | Peso en kg                   | 30-250  | 85      |
| `FCVC`   | Frecuencia consumo vegetales | 1-3     | 2       |
| `NCP`    | Número comidas principales   | 1-6     | 3       |
| `CH2O`   | Consumo de agua              | 1-3     | 2       |
| `FAF`    | Frecuencia actividad física  | 0-3     | 1       |
| `TUE`    | Tiempo uso tecnología        | 0-2     | 2       |

### Variables Categóricas:

| Campo                            | Descripción                     | Valores Posibles                                                      |
| -------------------------------- | ------------------------------- | --------------------------------------------------------------------- |
| `Gender`                         | Género                          | "Male", "Female"                                                      |
| `family_history_with_overweight` | Historia familiar               | "yes", "no"                                                           |
| `FAVC`                           | Consume comida alta en calorías | "yes", "no"                                                           |
| `CAEC`                           | Come entre comidas              | "no", "Sometimes", "Frequently", "Always"                             |
| `SMOKE`                          | Fumador                         | "yes", "no"                                                           |
| `SCC`                            | Monitorea calorías consumidas   | "yes", "no"                                                           |
| `CALC`                           | Consumo de alcohol              | "no", "Sometimes", "Frequently", "Always"                             |
| `MTRANS`                         | Transporte usual                | "Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile" |

### Clases de Salida:

| Clase                 | Descripción        | Nivel de Riesgo |
| --------------------- | ------------------ | --------------- |
| `Insufficient_Weight` | Peso insuficiente  | Medium          |
| `Normal_Weight`       | Peso normal        | Low             |
| `Overweight_Level_I`  | Sobrepeso Nivel I  | Medium          |
| `Overweight_Level_II` | Sobrepeso Nivel II | Medium          |
| `Obesity_Type_I`      | Obesidad Tipo I    | High            |
| `Obesity_Type_II`     | Obesidad Tipo II   | High            |
| `Obesity_Type_III`    | Obesidad Tipo III  | High            |

## 💻 Ejemplos de Uso

### Python con requests:

```python
import requests

# Datos de ejemplo
data = {
    "Age": 30,
    "Height": 1.70,
    "Weight": 80,
    # ... resto de campos
}

# Hacer predicción
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=data
)

result = response.json()
print(f"Predicción: {result['prediction']}")
print(f"Confianza: {result['confidence']}")
```

### cURL:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 30,
    "Height": 1.70,
    "Weight": 80,
    "FCVC": 2,
    "NCP": 3,
    "CH2O": 2,
    "FAF": 1,
    "TUE": 1,
    "Gender": "Male",
    "family_history_with_overweight": "no",
    "FAVC": "yes",
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "SCC": "yes",
    "CALC": "no",
    "MTRANS": "Walking"
  }'
```

### JavaScript/Fetch:

```javascript
const data = {
  Age: 30,
  Height: 1.7,
  Weight: 80,
  // ... resto de campos
};

fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(data),
})
  .then((response) => response.json())
  .then((result) => {
    console.log("Predicción:", result.prediction);
    console.log("Confianza:", result.confidence);
  });
```

## 🔧 Configuración y Deployment

### Variables de Entorno:

```bash
# Opcional: configurar puerto y host
export API_HOST=127.0.0.1
export API_PORT=8000

# Para producción
export WORKERS=4
export LOG_LEVEL=info
```

### Docker (opcional):

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "start_api.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Nginx (para producción):

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 Monitoreo y Performance

### Métricas del Modelo:

- **Accuracy**: 91.5% - 96.5%
- **F1-macro**: 91.2% - 96.2%
- **Tiempo de respuesta**: <100ms (predicción individual)
- **Throughput**: ~500 predicciones/segundo

### Logs:

```bash
# Ver logs en tiempo real
tail -f api.log

# Logs incluyen:
# - Requests HTTP
# - Tiempos de respuesta
# - Errores de predicción
# - Estado de carga del modelo
```

## 🛠️ Troubleshooting

### Error: "Modelo no cargado"

```bash
# Verificar que existe el modelo
ls models/mlflow_model/

# Re-entrenar si es necesario
python run_mlops.py cli pipeline
```

### Error: "Encoder/Scaler no encontrado"

```bash
# Verificar artefactos de features
ls models/features/

# Re-generar features si es necesario
python run_mlops.py cli features --input data/interim/obesity_clean.csv --output data/processed/features.csv
```

### Error de validación de datos:

- Verificar que todos los campos requeridos estén presentes
- Verificar que los valores estén en los rangos válidos
- Verificar que las variables categóricas tengan valores válidos

## 🔗 Links Útiles

- **Documentación interactiva**: http://127.0.0.1:8000/docs
- **OpenAPI Schema**: http://127.0.0.1:8000/openapi.json
- **Health Check**: http://127.0.0.1:8000/health
- **Proyecto MLOps**: [README principal](../README.md)

## 📈 Roadmap

### Próximas mejoras:

- [ ] Autenticación JWT
- [ ] Rate limiting
- [ ] Modelo ensemble
- [ ] Cache de predicciones
- [ ] Métricas de Prometheus
- [ ] Deployment en cloud (AWS/Azure)
- [ ] A/B testing de modelos
- [ ] Feedback loop para reentrenamiento
