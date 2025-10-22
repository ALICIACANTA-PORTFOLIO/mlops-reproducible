# src/serving/gradio_app_simple.py
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
import json
import plotly.express as px
import plotly.graph_objects as go

# Variables globales
model = None
encoder = None
scaler = None

def load_model_artifacts():
    """Cargar modelo y preprocessors"""
    global model, encoder, scaler
    
    try:
        model_path = Path("models/mlflow_model")
        features_path = Path("models/features")
        
        if not model_path.exists():
            return False, f"Modelo no encontrado en {model_path}"
        
        model = mlflow.sklearn.load_model(str(model_path))
        
        if (features_path / "encoder.joblib").exists():
            encoder = joblib.load(features_path / "encoder.joblib")
        if (features_path / "scaler.joblib").exists():
            scaler = joblib.load(features_path / "scaler.joblib")
        
        return True, "Modelo cargado exitosamente"
    except Exception as e:
        return False, f"Error: {str(e)}"

def predict_obesity(age, height, weight, fcvc, ncp, ch2o, faf, tue,
                   gender, family_history, favc, caec, smoke, scc, calc, mtrans):
    """Realizar predicción"""
    
    if model is None:
        return "❌ Modelo no disponible", None, None, None
    
    try:
        # Calcular IMC
        imc = weight / (height ** 2)
        
        # Preparar datos
        data = {
            'Age': age, 'Height': height, 'Weight': weight,
            'FCVC': fcvc, 'NCP': ncp, 'CH2O': ch2o,
            'FAF': faf, 'TUE': tue, 'Gender': gender,
            'family_history_with_overweight': family_history,
            'FAVC': favc, 'CAEC': caec, 'SMOKE': smoke,
            'SCC': scc, 'CALC': calc, 'MTRANS': mtrans
        }
        
        df = pd.DataFrame([data])
        
        # Preprocessing (copiado de la API que funciona)
        categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 
                          'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        
        # Separar categoricas y numericas
        df_cat = df[categorical_cols].copy()
        df_num = df[numerical_cols].copy()
        
        # Aplicar encoder y scaler
        if encoder is not None:
            df_cat_encoded = encoder.transform(df_cat)
            if hasattr(df_cat_encoded, 'toarray'):  # OneHotEncoder
                df_cat_encoded = df_cat_encoded.toarray()
            df_cat_encoded = pd.DataFrame(df_cat_encoded)
        else:
            df_cat_encoded = pd.get_dummies(df_cat)
            
        if scaler is not None:
            df_num_scaled = scaler.transform(df_num)
            df_num_scaled = pd.DataFrame(df_num_scaled, columns=numerical_cols)
        else:
            df_num_scaled = df_num
            
        # Concatenar
        df = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
        
        # Predicción
        X = df.values
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        # Verificar que tenemos datos válidos
        print(f"Debug - prediction: {prediction}")
        print(f"Debug - prediction_proba: {prediction_proba}")
        
        # Mapeo de clases
        class_names = ['Peso_Insuficiente', 'Normal', 'Sobrepeso_Nivel_I', 
                      'Sobrepeso_Nivel_II', 'Obesidad_Tipo_I', 'Obesidad_Tipo_II', 
                      'Obesidad_Tipo_III']
        
        prediction_name = class_names[prediction]
        confidence = prediction_proba.max() * 100
        
        # Resultado texto
        result_text = f"""
        **🎯 RESULTADO DEL ANÁLISIS**
        
        **Clasificación:** {prediction_name}
        **Confianza:** {confidence:.1f}%
        **IMC:** {imc:.1f} kg/m²
        """
        
        # Gráfico de probabilidades - CORREGIDO
        colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#34495e']
        
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Bar(
            x=class_names,
            y=prediction_proba,
            marker_color=colors[:len(class_names)],
            text=[f'{prob:.1%}' for prob in prediction_proba],
            textposition='auto',
            name='Probabilidad'
        ))
        
        fig_prob.update_layout(
            title='📊 Probabilidades por Clase',
            xaxis_title='Tipo de Obesidad',
            yaxis_title='Probabilidad',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, color='black'),
            margin=dict(l=50, r=50, t=80, b=100),
            xaxis=dict(tickangle=45, color='black'),
            yaxis=dict(color='black'),
            showlegend=False
        )
        
        # Gráfico de IMC - CORREGIDO
        fig_imc = go.Figure()
        
        # Rangos de IMC
        imc_data = {
            'Bajo peso': (0, 18.5, '#3498db'),
            'Normal': (18.5, 24.9, '#2ecc71'), 
            'Sobrepeso': (25, 29.9, '#f39c12'),
            'Obesidad I': (30, 34.9, '#e67e22'),
            'Obesidad II': (35, 39.9, '#e74c3c'),
            'Obesidad III': (40, 50, '#c0392b')
        }
        
        # Crear barras para rangos de IMC
        for i, (categoria, (min_val, max_val, color)) in enumerate(imc_data.items()):
            fig_imc.add_trace(go.Bar(
                x=[categoria],
                y=[max_val - min_val],
                base=[min_val],
                name=categoria,
                marker_color=color,
                opacity=0.7,
                showlegend=True
            ))
        
        # Agregar punto del usuario
        fig_imc.add_trace(go.Scatter(
            x=['Tu IMC'],
            y=[imc],
            mode='markers',
            marker=dict(
                size=20, 
                color='red',
                line=dict(width=3, color='white'),
                symbol='diamond'
            ),
            name=f'Tu IMC: {imc:.1f}',
            showlegend=True
        ))
        
        fig_imc.update_layout(
            title='📏 Tu IMC vs Rangos Normales',
            yaxis_title='IMC (kg/m²)',
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, color='black'),
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=True,
            xaxis=dict(color='black'),
            yaxis=dict(color='black')
        )
        
        # Recomendaciones simples
        if "Normal" in prediction_name:
            recomendaciones = """
            **✅ ¡Excelente!** Tu peso está en el rango normal.
            
            **Mantén:**
            - Dieta equilibrada
            - Ejercicio regular
            - Buenos hábitos de sueño
            """
        else:
            recomendaciones = f"""
            **⚠️ Atención:** Tu clasificación es {prediction_name}
            
            **Recomendaciones:**
            - Consulta con un médico
            - Considera cambios en la dieta
            - Incrementa la actividad física
            - Mantén un seguimiento regular
            """
        
        return result_text, fig_prob, fig_imc, recomendaciones
        
    except Exception as e:
        error_msg = f"❌ Error en predicción: {str(e)}"
        return error_msg, None, None, error_msg

# Cargar modelo
success, message = load_model_artifacts()

def create_gradio_interface():
    """Crear interfaz simple y funcional"""
    
    with gr.Blocks(
        title="Clasificador de Obesidad MLOps",
        theme=gr.themes.Soft(),
        css="""
        /* Estilos para mejor contraste */
        .gradio-container {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Headers con mejor contraste */
        h1, h2, h3, h4 {
            color: var(--body-text-color) !important;
            text-shadow: none !important;
        }
        
        /* Status cards */
        .status-success {
            background: linear-gradient(135deg, #d1fae5, #a7f3d0);
            border: 1px solid #10b981;
            border-radius: 8px;
            padding: 12px;
            color: #065f46;
        }
        
        .status-error {
            background: linear-gradient(135deg, #fee2e2, #fca5a5);
            border: 1px solid #ef4444;
            border-radius: 8px;
            padding: 12px;
            color: #991b1b;
        }
        
        /* API docs styling */
        .api-docs {
            background: linear-gradient(135deg, #eff6ff, #dbeafe);
            border: 1px solid #3b82f6;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .code-block {
            background: #1e293b;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
            overflow-x: auto;
        }
        """
    ) as demo:
        
        # Header principal con colores profesionales
        gr.HTML("""
        <div style='text-align: center; padding: 24px; background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%); border-radius: 12px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(59, 130, 246, 0.2);'>
            <h1 style='color: white; margin: 0; font-size: 2.4em; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>🏥 Clasificador de Obesidad</h1>
            <p style='color: rgba(255,255,255,0.95); margin: 12px 0 0 0; font-size: 1.1em; font-weight: 500;'>Sistema MLOps - ML + DevOps + Data Engineering</p>
            <p style='color: rgba(255,255,255,0.8); margin: 8px 0 0 0; font-size: 0.9em;'>🚀 Powered by MLflow + FastAPI + Gradio</p>
        </div>
        """)
        
        # Estado del modelo con mejor contraste
        if success:
            gr.HTML(f"""
            <div class='status-success'>
                <strong>✅ Modelo Operativo</strong><br>
                <small>{message}</small>
            </div>
            """)
        else:
            gr.HTML(f"""
            <div class='status-error'>
                <strong>❌ Error del Modelo</strong><br>
                <small>{message}</small>
            </div>
            """)
        
        # Crear tabs para organizar mejor el contenido
        with gr.Tabs():
            with gr.Tab("🎯 Análisis"):
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="section-personal">')
                        gr.Markdown("### 👤 Datos Personales")
                        age = gr.Number(label="Edad", value=25, minimum=10, maximum=90)
                        gender = gr.Radio(label="Género", choices=["Male", "Female"], value="Female")
                        height = gr.Slider(label="Altura (m)", minimum=1.2, maximum=2.2, value=1.70, step=0.01)
                        weight = gr.Slider(label="Peso (kg)", minimum=30, maximum=250, value=70, step=1)
                        
                        gr.Markdown("### 🍎 Alimentación")
                        fcvc = gr.Slider(label="Consumo Vegetales (1-3)", minimum=1, maximum=3, value=2, step=1)
                        ncp = gr.Slider(label="Comidas por día", minimum=1, maximum=6, value=3, step=1)
                        ch2o = gr.Slider(label="Consumo Agua (1-3)", minimum=1, maximum=3, value=2, step=1)
                        favc = gr.Radio(label="¿Comida alta en calorías?", choices=["yes", "no"], value="no")
                        caec = gr.Dropdown(label="¿Comes entre comidas?", choices=["Sometimes", "Frequently", "Always", "no"], value="Sometimes")
                        calc = gr.Radio(label="¿Consumes alcohol?", choices=["yes", "no"], value="no")
                        
                        gr.Markdown("### 🏃 Estilo de Vida")
                        faf = gr.Slider(label="Actividad Física (0-3)", minimum=0, maximum=3, value=1, step=1)
                        tue = gr.Slider(label="Uso Tecnología (0-2)", minimum=0, maximum=2, value=1, step=1)
                        mtrans = gr.Dropdown(label="Transporte", choices=["Walking", "Public_Transportation", "Automobile", "Bike"], value="Walking")
                        smoke = gr.Radio(label="¿Fumas?", choices=["yes", "no"], value="no")
                        scc = gr.Radio(label="¿Monitoreas calorías?", choices=["yes", "no"], value="no")
                        family_history = gr.Radio(label="¿Historial familiar?", choices=["yes", "no"], value="no")
                        gr.HTML('</div>')
                        
                        predict_btn = gr.Button("🎯 Analizar", variant="primary", size="lg")
            
                    with gr.Column():
                        gr.HTML('<div class="section-results">')
                        gr.Markdown("### 🎯 Resultados")
                        result_text = gr.Markdown("Completa los datos y haz clic en 'Analizar'")
                        prob_plot = gr.Plot(label="📊 Probabilidades por Clase")
                        imc_plot = gr.Plot(label="📏 Análisis de IMC")
                        recommendations = gr.Markdown("Las recomendaciones aparecerán aquí")
                        gr.HTML('</div>')
        
        # Ejemplos
        gr.Markdown("### 📝 Ejemplos")
        gr.Examples(
            examples=[
                # [age, height, weight, fcvc, ncp, ch2o, faf, tue, gender, family_history, favc, caec, smoke, scc, calc, mtrans]
                [25, 1.70, 65, 3, 3, 3, 2, 1, "Female", "no", "no", "Sometimes", "no", "no", "no", "Walking"],
                [35, 1.75, 85, 1, 4, 1, 0, 2, "Male", "yes", "yes", "Frequently", "no", "no", "no", "Automobile"]
            ],
            inputs=[age, height, weight, fcvc, ncp, ch2o, faf, tue, gender, family_history, favc, caec, smoke, scc, calc, mtrans]
        )
        
        # Conectar función
        predict_btn.click(
            fn=predict_obesity,
            inputs=[age, height, weight, fcvc, ncp, ch2o, faf, tue, gender, family_history, favc, caec, smoke, scc, calc, mtrans],
            outputs=[result_text, prob_plot, imc_plot, recommendations]
        )
        
        # Documentación de API
        gr.HTML("""
        <div class='api-docs'>
            <h2 style='color: #1e40af; margin-top: 0;'>🚀 API REST del Modelo</h2>
            <p>Este modelo también está disponible como API REST para integración en aplicaciones.</p>
            
            <h3>📍 Endpoints Disponibles</h3>
            <ul>
                <li><strong>Health Check:</strong> <code>GET /health</code></li>
                <li><strong>Predicción:</strong> <code>POST /predict</code></li>
                <li><strong>Predicción Batch:</strong> <code>POST /predict/batch</code></li>
                <li><strong>Documentación:</strong> <code>GET /docs</code></li>
            </ul>
            
            <h3>💡 Ejemplo de Uso - Python</h3>
            <div class='code-block'>
import requests<br>
import json<br><br>

# URL del API<br>
api_url = "http://127.0.0.1:8000"<br><br>

# Datos de ejemplo<br>
data = {<br>
&nbsp;&nbsp;"age": 25,<br>
&nbsp;&nbsp;"height": 1.70,<br>
&nbsp;&nbsp;"weight": 70,<br>
&nbsp;&nbsp;"fcvc": 3,<br>
&nbsp;&nbsp;"ncp": 3,<br>
&nbsp;&nbsp;"ch2o": 2,<br>
&nbsp;&nbsp;"faf": 2,<br>
&nbsp;&nbsp;"tue": 1,<br>
&nbsp;&nbsp;"gender": "Female",<br>
&nbsp;&nbsp;"family_history": "no",<br>
&nbsp;&nbsp;"favc": "no",<br>
&nbsp;&nbsp;"caec": "Sometimes",<br>
&nbsp;&nbsp;"smoke": "no",<br>
&nbsp;&nbsp;"scc": "no",<br>
&nbsp;&nbsp;"calc": "no",<br>
&nbsp;&nbsp;"mtrans": "Walking"<br>
}<br><br>

# Realizar predicción<br>
response = requests.post(f"{api_url}/predict", json=data)<br>
result = response.json()<br><br>

print(f"Clasificación: {result['classification']}")<br>
print(f"Confianza: {result['confidence']:.2f}")
            </div>
            
            <h3>🌐 Ejemplo de Uso - JavaScript</h3>
            <div class='code-block'>
const apiUrl = "http://127.0.0.1:8000";<br><br>

const data = {<br>
&nbsp;&nbsp;age: 25,<br>
&nbsp;&nbsp;height: 1.70,<br>
&nbsp;&nbsp;weight: 70,<br>
&nbsp;&nbsp;fcvc: 3,<br>
&nbsp;&nbsp;ncp: 3,<br>
&nbsp;&nbsp;ch2o: 2,<br>
&nbsp;&nbsp;faf: 2,<br>
&nbsp;&nbsp;tue: 1,<br>
&nbsp;&nbsp;gender: "Female",<br>
&nbsp;&nbsp;family_history: "no",<br>
&nbsp;&nbsp;favc: "no",<br>
&nbsp;&nbsp;caec: "Sometimes",<br>
&nbsp;&nbsp;smoke: "no",<br>
&nbsp;&nbsp;scc: "no",<br>
&nbsp;&nbsp;calc: "no",<br>
&nbsp;&nbsp;mtrans: "Walking"<br>
};<br><br>

fetch(`${apiUrl}/predict`, {<br>
&nbsp;&nbsp;method: 'POST',<br>
&nbsp;&nbsp;headers: { 'Content-Type': 'application/json' },<br>
&nbsp;&nbsp;body: JSON.stringify(data)<br>
})<br>
.then(response => response.json())<br>
.then(result => {<br>
&nbsp;&nbsp;console.log(`Clasificación: ${result.classification}`);<br>
&nbsp;&nbsp;console.log(`Confianza: ${result.confidence.toFixed(2)}`);<br>
});
            </div>
            
            <h3>📋 Comandos para Iniciar API</h3>
            <div class='code-block'>
# Iniciar API REST<br>
python start_api.py<br><br>

# O directamente con uvicorn<br>
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload<br><br>

# Ver documentación interactiva<br>
# Abrir: http://127.0.0.1:8000/docs
            </div>
            
            <p><strong>🔗 Más información:</strong> <a href="http://127.0.0.1:8000/docs" target="_blank">Documentación Interactiva (Swagger)</a></p>
        </div>
        """)
        
    return demo

def main():
    """Ejecutar aplicación"""
    demo = create_gradio_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()
