# Interfaz Web Gradio - Clasificador de Obesidad

## 🎨 Descripción

Interfaz web amigable y visualmente atractiva para el clasificador de obesidad, construida con **Gradio**. Proporciona una experiencia de usuario intuitiva para interactuar con el modelo MLOps sin necesidad de conocimientos técnicos.

## ✨ Características

### 🎯 **Interfaz Intuitiva**

- **Formulario interactivo** con sliders, dropdowns y radio buttons
- **Validación en tiempo real** de entrada de datos
- **Ejemplos predefinidos** para pruebas rápidas
- **Diseño responsivo** que se adapta a diferentes pantallas

### 📊 **Visualizaciones Interactivas**

- **Gráfico de probabilidades** por clase de obesidad
- **Análisis de IMC** con rangos de referencia
- **Indicadores visuales** de nivel de riesgo
- **Gráficos Plotly** interactivos

### 💡 **Recomendaciones Inteligentes**

- **Sugerencias personalizadas** basadas en la predicción
- **Consejos de salud** específicos por categoría
- **Recomendaciones de ejercicio** según actividad actual
- **Consejos nutricionales** según hábitos alimenticios

## 🚀 Inicio Rápido

### 1. **Instalar Dependencias**

```bash
pip install gradio plotly
# O instalar desde requirements.txt actualizado
pip install -r requirements.txt
```

### 2. **Entrenar el Modelo** (si no lo has hecho)

```bash
python run_mlops.py cli pipeline
```

### 3. **Iniciar Gradio**

```bash
# Opción 1: Script personalizado (recomendado)
python start_gradio.py

# Opción 2: Ejecución directa
python src/serving/gradio_app.py

# Opción 3: Con parámetros personalizados
python start_gradio.py --port 7860 --host 127.0.0.1 --share
```

### 4. **Abrir en el Navegador**

```
http://127.0.0.1:7860
```

## 🎮 Cómo Usar la Interfaz

### **Paso 1: Datos Personales**

- 🎂 **Edad**: Usar el slider (10-90 años)
- 👤 **Género**: Seleccionar Male o Female
- 📏 **Altura**: Ajustar en metros (1.20-2.20m)
- ⚖️ **Peso**: Establecer en kg (30-250kg)

### **Paso 2: Hábitos Alimenticios**

- 🥬 **Consumo de Vegetales**: 1=Bajo, 2=Medio, 3=Alto
- 🍽️ **Comidas Principales**: Número por día (1-6)
- 💧 **Consumo de Agua**: Nivel de hidratación
- 🍟 **Comida Alta en Calorías**: Frecuencia de consumo
- 🍪 **Comer Entre Comidas**: Hábito de snacking
- 🍷 **Consumo de Alcohol**: Frecuencia

### **Paso 3: Estilo de Vida**

- 🏃‍♀️ **Actividad Física**: 0=Ninguna, 3=Intensa
- 📱 **Uso de Tecnología**: Tiempo de pantalla
- 🚗 **Transporte Principal**: Medio habitual
- 🚭 **Fumar**: Sí/No
- 📊 **Monitor de Calorías**: ¿Llevas control?
- 👨‍👩‍👧‍👦 **Historia Familiar**: Antecedentes familiares

### **Paso 4: Análisis**

- 🎯 **Hacer clic** en "Analizar mi Estado de Salud"
- 📊 **Ver resultados** con clasificación y confianza
- 📈 **Revisar gráficos** interactivos
- 💡 **Leer recomendaciones** personalizadas

## 📋 Casos de Ejemplo Incluidos

### **👩 Persona Saludable**

- Edad: 25, Altura: 1.70m, Peso: 65kg
- Alto consumo de vegetales, ejercicio regular
- Resultado esperado: **Normal Weight** 🟢

### **👨 Persona con Sobrepeso**

- Edad: 35, Altura: 1.75m, Peso: 85kg
- Bajo consumo de vegetales, vida sedentaria
- Resultado esperado: **Overweight Level I** 🟡

### **👩 Persona Obesa**

- Edad: 45, Altura: 1.60m, Peso: 95kg
- Malos hábitos alimenticios, sin ejercicio
- Resultado esperado: **Obesity Type I** 🔴

## 🎨 Características Visuales

### **🎨 Diseño Atractivo**

- **Tema personalizado** con colores profesionales
- **Header gradient** con información del proyecto
- **Iconos intuitivos** para cada sección
- **Grupos organizados** por categoría

### **📊 Gráficos Interactivos**

- **Barras horizontales** para probabilidades por clase
- **Código de colores** (verde=saludable, rojo=riesgo)
- **Indicador visual** de la predicción principal
- **Gráfico de IMC** con tu posición relativa

### **💡 Feedback Inteligente**

- **Colores de riesgo**: 🟢 Bajo, 🟡 Medio, 🔴 Alto
- **Mensajes contextuales** según la clasificación
- **Recomendaciones específicas** por área de mejora
- **Descargo de responsabilidad** médica

## 🔧 Configuración Avanzada

### **Parámetros de Inicio**

```bash
python start_gradio.py --help

# Opciones disponibles:
--port 7860          # Puerto personalizado
--host 127.0.0.1     # Host personalizado
--share              # Crear enlace público temporal
--debug              # Modo debug con más información
```

### **Variables de Entorno**

```bash
# Configuración opcional
export GRADIO_SERVER_NAME=127.0.0.1
export GRADIO_SERVER_PORT=7860
export GRADIO_THEME=soft  # soft, default, monochrome
```

### **Personalización del Tema**

```python
# En gradio_app.py, modificar:
theme = gr.themes.Soft(
    primary_hue="blue",      # Color principal
    secondary_hue="green",   # Color secundario
    neutral_hue="gray"       # Color neutral
)
```

## 🚀 Integración con APIs

### **FastAPI + Gradio**

```bash
# Terminal 1: Iniciar FastAPI
python start_api.py --reload

# Terminal 2: Iniciar Gradio
python start_gradio.py

# Ahora tienes ambas interfaces:
# - API REST: http://127.0.0.1:8000
# - Gradio UI: http://127.0.0.1:7860
```

### **Comparación FastAPI vs Gradio**

| Característica    | FastAPI            | Gradio           |
| ----------------- | ------------------ | ---------------- |
| **Audiencia**     | Desarrolladores    | Usuarios finales |
| **Formato**       | JSON REST          | Interfaz visual  |
| **Documentación** | Swagger automático | UI intuitiva     |
| **Integración**   | Otras aplicaciones | Demo interactivo |
| **Casos de uso**  | Producción         | Prototipado/Demo |

## 📊 Monitoreo y Analytics

### **Logs de Gradio**

```bash
# Ver logs en tiempo real
tail -f gradio.log

# Información registrada:
# - Usuarios conectados
# - Predicciones realizadas
# - Errores de entrada
# - Tiempo de respuesta
```

### **Métricas de Uso**

- **Número de predicciones** por sesión
- **Casos de ejemplo** más utilizados
- **Errores de validación** comunes
- **Tiempo promedio** por predicción

## 🔄 Workflow Completo

### **Para Desarrollo:**

```bash
# 1. Desarrollo del modelo
python run_mlops.py cli pipeline

# 2. Probar API
python start_api.py --reload
python test_api.py

# 3. Probar Gradio
python start_gradio.py --debug

# 4. Demo completo
# - FastAPI: http://127.0.0.1:8000
# - Gradio: http://127.0.0.1:7860
```

### **Para Presentaciones:**

```bash
# Enlace público temporal (para demos remotas)
python start_gradio.py --share

# Resultado: https://abc123.gradio.live (temporal)
```

## 🛠️ Troubleshooting

### **Error: Modelo no cargado**

```bash
# Verificar modelo existe
ls models/mlflow_model/

# Re-entrenar si es necesario
python run_mlops.py cli pipeline
```

### **Error: Dependencias faltantes**

```bash
# Instalar Gradio y Plotly
pip install gradio plotly

# O actualizar requirements completo
pip install -r requirements.txt
```

### **Error: Puerto ocupado**

```bash
# Cambiar puerto
python start_gradio.py --port 7861

# O usar puerto aleatorio
python start_gradio.py --port 0
```

### **Problemas de renderizado**

- **Limpiar caché** del navegador
- **Probar en modo incógnito**
- **Verificar JavaScript** habilitado
- **Usar navegador actualizado**

## 🎯 Roadmap y Mejoras

### **Próximas características:**

- [ ] **Modo oscuro** toggle
- [ ] **Múltiples idiomas** (español, inglés)
- [ ] **Historial de predicciones** por sesión
- [ ] **Comparación de resultados** lado a lado
- [ ] **Export de resultados** a PDF
- [ ] **Autenticación** de usuarios
- [ ] **Analytics** avanzados con dashboard
- [ ] **Integración** con dispositivos móviles

### **Mejoras técnicas:**

- [ ] **Cache** de predicciones
- [ ] **Rate limiting** para evitar spam
- [ ] **Validación avanzada** de entrada
- [ ] **Modelos A/B testing** en la UI
- [ ] **Feedback loop** de usuarios
- [ ] **Deployment** en cloud

## 🔗 Enlaces Relacionados

- **API REST**: [`docs/API_DOCUMENTATION.md`](API_DOCUMENTATION.md)
- **MLOps Integration**: [`docs/MLOPS_INTEGRATION.md`](MLOPS_INTEGRATION.md)
- **Proyecto Principal**: [`README.md`](../README.md)
- **Gradio Oficial**: https://gradio.app/
- **Plotly Docs**: https://plotly.com/python/

---

**💡 La interfaz Gradio complementa perfectamente la API FastAPI, ofreciendo una experiencia completa: API para desarrolladores + UI para usuarios finales.**
