# src/serving/gradio_app.py - Versión Simple y Clara
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
import json
import plotly.graph_objects as go

# Variables globales
model = None
encoder = None
scaler = None
feature_names = None
class_mapping = None

def load_model_artifacts():
    """Cargar modelo y artefactos"""
    global model, encoder, scaler, feature_names, class_mapping
    
    try:
        model_path = Path("models/mlflow_model")
        features_path = Path("models/features")
        
        if not model_path.exists():
            return False, f"Modelo no encontrado en {model_path}"
        
        model = mlflow.sklearn.load_model(str(model_path))
        
        encoder_path = features_path / "encoder.joblib"
        scaler_path = features_path / "scaler.joblib" 
        meta_path = features_path / "features_meta.json"
        
        if encoder_path.exists():
            encoder = joblib.load(encoder_path)
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                feature_names = meta.get('feature_names', [])
                class_mapping = meta.get('class_mapping', {})
        
        return True, "Modelo cargado exitosamente"
        
    except Exception as e:
        return False, f"Error cargando modelo: {str(e)}"

def predict_obesity(age, height, weight, fcvc, ncp, ch2o, faf, tue,
                   gender, family_history, favc, caec, smoke, scc, calc, mtrans):
    """Realizar predicción de obesidad"""
    
    if model is None:
        return "❌ Modelo no disponible", None, None, None
    
    try:
        imc = weight / (height ** 2)
        
        data = {
            'Age': age, 'Height': height, 'Weight': weight,
            'FCVC': fcvc, 'NCP': ncp, 'CH2O': ch2o,
            'FAF': faf, 'TUE': tue, 'Gender': gender,
            'family_history_with_overweight': family_history,
            'FAVC': favc, 'CAEC': caec, 'SMOKE': smoke,
            'SCC': scc, 'CALC': calc, 'MTRANS': mtrans
        }
        
        df = pd.DataFrame([data])
        
        if encoder is not None:
            categorical_columns = ['Gender', 'family_history_with_overweight', 
                                 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = encoder[col].transform(df[col])
        
        if scaler is not None:
            df_scaled = scaler.transform(df)
            df = pd.DataFrame(df_scaled, columns=df.columns)
        
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]
        
        if class_mapping:
            prediction_name = [k for k, v in class_mapping.items() if v == prediction][0]
        else:
            prediction_name = str(prediction)
        
        confidence = prediction_proba.max() * 100
        result_text = f"""
        <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); 
                    border: 1px solid #10b981; border-left: 4px solid #059669;
                    border-radius: 12px; padding: 20px; margin: 16px 0;">
            <h3 style="color: #065f46; margin: 0 0 12px 0;">🎯 <strong>Resultado del Análisis</strong></h3>
            <p style="margin: 8px 0; color: #374151;"><strong>Clasificación:</strong> {prediction_name}</p>
            <p style="margin: 8px 0; color: #374151;"><strong>Confianza:</strong> {confidence:.1f}%</p>
            <p style="margin: 8px 0; color: #374151;"><strong>IMC Calculado:</strong> {imc:.1f} kg/m²</p>
        </div>
        """
        
        # Gráfico de probabilidades
        class_names = ['Peso_Insuficiente', 'Normal', 'Sobrepeso_Nivel_I', 
                      'Sobrepeso_Nivel_II', 'Obesidad_Tipo_I', 'Obesidad_Tipo_II', 'Obesidad_Tipo_III']
        colors = ['#10b981', '#22c55e', '#84cc16', '#eab308', '#f59e0b', '#ef4444', '#dc2626']
        
        fig = go.Figure(data=[
            go.Bar(
                x=class_names, y=prediction_proba, marker_color=colors,
                text=[f'{prob:.2%}' for prob in prediction_proba], textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='📊 Distribución de Probabilidades por Clase',
            xaxis_title='Clasificación de Obesidad', yaxis_title='Probabilidad',
            height=400, showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
            title_font_color='#1e293b', title_font_size=16, font=dict(size=12, color='#374151'),
            margin=dict(l=20, r=20, t=60, b=100), xaxis=dict(tickangle=45)
        )
        
        # Gráfico de IMC
        imc_ranges = {
            'Bajo peso': (0, 18.5, '#60a5fa'), 'Normal': (18.5, 24.9, '#34d399'), 
            'Sobrepeso': (25, 29.9, '#fbbf24'), 'Obesidad I': (30, 34.9, '#fb923c'),
            'Obesidad II': (35, 39.9, '#f87171'), 'Obesidad III': (40, 50, '#ef4444')
        }
        
        fig_imc = go.Figure()
        for categoria, (min_val, max_val, color) in imc_ranges.items():
            fig_imc.add_trace(go.Bar(
                x=[categoria], y=[max_val - min_val], base=[min_val],
                name=categoria, marker_color=color, opacity=0.8
            ))
        
        fig_imc.add_trace(go.Scatter(
            x=['Tu IMC'], y=[imc], mode='markers',
            marker=dict(size=15, color='#1e40af', line=dict(width=2, color='white'), symbol='diamond'),
            name=f'Tu IMC: {imc:.1f}'
        ))
        
        fig_imc.update_layout(
            title='📏 Tu IMC en Contexto', yaxis_title='IMC (kg/m²)', height=350,
            showlegend=True, plot_bgcolor='white', paper_bgcolor='white',
            title_font_color='#1e293b', title_font_size=16, font=dict(color='#374151', size=12),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        recommendations = generar_recomendaciones(prediction_name, imc, faf, fcvc)
        return result_text, fig, fig_imc, recommendations
        
    except Exception as e:
        return f"❌ Error en predicción: {str(e)}", None, None, None

def generar_recomendaciones(prediction, imc, faf, fcvc):
    """Generar recomendaciones personalizadas"""
    
    recomendaciones = f"""
    <div style="padding: 20px; background: linear-gradient(135deg, #f0f9ff, #e0f2fe); 
                border-radius: 12px; border-left: 4px solid #0ea5e9;">
        <h3 style="color: #0c4a6e; margin-bottom: 16px;">💡 Recomendaciones Personalizadas</h3>
    """
    
    if "Normal" in prediction:
        recomendaciones += """
        <div style="background: white; padding: 15px; border-radius: 8px; margin-bottom: 12px;">
            <strong style="color: #059669;">✅ ¡Excelente!</strong> Mantén tu estilo de vida saludable.
            <ul style="margin: 8px 0; color: #374151;">
                <li>Continúa con una dieta balanceada</li>
                <li>Mantén tu rutina de ejercicio regular</li>
                <li>Hidratación adecuada (2-3L agua/día)</li>
            </ul>
        </div>
        """
    elif "Sobrepeso" in prediction or "Obesidad" in prediction:
        recomendaciones += """
        <div style="background: white; padding: 15px; border-radius: 8px; margin-bottom: 12px;">
            <strong style="color: #d97706;">⚠️ Atención:</strong> Es recomendable hacer algunos ajustes.
            <ul style="margin: 8px 0; color: #374151;">
                <li>Consulta con un nutricionista profesional</li>
                <li>Incrementa la actividad física gradualmente</li>
                <li>Reduce alimentos procesados y azúcares</li>
                <li>Controla las porciones de comida</li>
            </ul>
        </div>
        """
    
    if imc < 18.5:
        recomendaciones += """
        <div style="background: #fef3c7; padding: 15px; border-radius: 8px; margin-bottom: 12px;">
            <strong style="color: #92400e;">📈 IMC Bajo:</strong> Considera aumentar tu masa muscular de forma saludable.
        </div>
        """
    elif imc > 30:
        recomendaciones += """
        <div style="background: #fee2e2; padding: 15px; border-radius: 8px; margin-bottom: 12px;">
            <strong style="color: #991b1b;">🎯 IMC Elevado:</strong> Es importante una intervención médica especializada.
        </div>
        """
    
    if faf < 2:
        recomendaciones += """
        <div style="background: #ede9fe; padding: 15px; border-radius: 8px; margin-bottom: 12px;">
            <strong style="color: #6b21a8;">🏃‍♀️ Actividad Física:</strong> Intenta realizar ejercicio al menos 3-4 veces por semana.
        </div>
        """
    
    if fcvc < 2:
        recomendaciones += """
        <div style="background: #ecfdf5; padding: 15px; border-radius: 8px; margin-bottom: 12px;">
            <strong style="color: #065f46;">🥬 Vegetales:</strong> Incrementa tu consumo de vegetales a 3-5 porciones diarias.
        </div>
        """
    
    recomendaciones += "</div>"
    return recomendaciones

# Cargar modelo
success, message = load_model_artifacts()
if not success:
    print(f"⚠️ Advertencia: {message}")

def create_gradio_interface():
    """Crear interfaz Gradio profesional"""
    
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter")
    )
    
    with gr.Blocks(
        theme=theme, title="🏥 Clasificador de Obesidad MLOps",
        css="""
        :root {
            --primary: #2563eb; --primary-hover: #1d4ed8; --success: #10b981;
            --success-light: #d1fae5; --warning: #f59e0b; --warning-light: #fef3c7;
            --error: #ef4444; --gray-50: #f9fafb; --gray-100: #f3f4f6;
            --gray-200: #e5e7eb; --gray-700: #374151; --gray-900: #111827; --white: #ffffff;
        }
        
        .gradio-container {
            font-family: 'Inter', system-ui, sans-serif !important;
            background: var(--white) !important; color: var(--gray-900) !important;
            max-width: 1200px !important; margin: 0 auto !important;
        }
        
        .block, .gr-group {
            background: var(--white) !important; border: 1px solid var(--gray-200) !important;
            border-radius: 12px !important; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        }
        
        .gr-group { padding: 20px !important; }
        
        .main-header {
            background: linear-gradient(135deg, var(--primary), var(--primary-hover)) !important;
            color: white !important; text-align: center !important; padding: 2.5rem !important;
            border-radius: 16px !important; margin-bottom: 2rem !important;
            box-shadow: 0 8px 32px rgba(37, 99, 235, 0.25) !important;
        }
        
        .main-header * { color: white !important; }
        
        .section-personal {
            background: linear-gradient(135deg, #eff6ff, #dbeafe) !important;
            border-left: 4px solid var(--primary) !important; border-radius: 12px !important; padding: 20px !important;
        }
        
        .section-results {
            background: linear-gradient(135deg, var(--warning-light), #fed7aa) !important;
            border-left: 4px solid var(--warning) !important; border-radius: 12px !important; padding: 20px !important;
        }
        
        .section-alimenticios {
            background: linear-gradient(135deg, #ecfdf5, var(--success-light)) !important;
            border-left: 4px solid var(--success) !important; border-radius: 12px !important; padding: 20px !important;
        }
        
        .section-lifestyle {
            background: linear-gradient(135deg, #fdf2f8, #fce7f3) !important;
            border-left: 4px solid #ec4899 !important; border-radius: 12px !important; padding: 20px !important;
        }
        
        label, .label, .gr-markdown, h1, h2, h3, h4, h5, h6 {
            color: var(--gray-900) !important; font-weight: 500 !important;
        }
        
        input, textarea, select, .gr-textbox input, .gr-number input {
            background: var(--white) !important; color: var(--gray-900) !important;
            border: 2px solid var(--gray-200) !important; border-radius: 8px !important; padding: 10px 12px !important;
        }
        
        input:focus, textarea:focus, select:focus {
            border-color: var(--primary) !important; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important; outline: none !important;
        }
        
        .gr-slider { background: var(--white) !important; padding: 15px !important; }
        .gr-slider input[type="range"] { accent-color: var(--primary) !important; }
        
        .gr-button {
            background: linear-gradient(135deg, var(--primary), var(--primary-hover)) !important;
            color: white !important; border: none !important; border-radius: 10px !important;
            font-weight: 600 !important; padding: 14px 28px !important; font-size: 16px !important;
            transition: all 0.2s ease !important;
        }
        
        .gr-button:hover {
            transform: translateY(-1px) !important; box-shadow: 0 4px 20px rgba(37, 99, 235, 0.3) !important;
        }
        
        .gr-radio label, .gr-checkbox label {
            background: var(--gray-50) !important; border: 2px solid var(--gray-200) !important;
            border-radius: 8px !important; padding: 12px 16px !important; margin: 4px !important;
            cursor: pointer !important; transition: all 0.2s ease !important; color: var(--gray-700) !important;
        }
        
        .gr-radio input:checked + label, .gr-checkbox input:checked + label {
            background: #dbeafe !important; border-color: var(--primary) !important;
            color: var(--primary-hover) !important; font-weight: 600 !important;
        }
        
        .gr-examples {
            background: var(--white) !important; border: 1px solid var(--gray-200) !important;
            border-radius: 12px !important; padding: 20px !important; margin-top: 20px !important;
        }
        
        .gr-examples table {
            background: var(--white) !important; border: 1px solid var(--gray-200) !important;
            border-radius: 8px !important; width: 100% !important;
        }
        
        .gr-examples th {
            background: var(--gray-50) !important; color: var(--gray-900) !important;
            font-weight: 600 !important; padding: 12px 8px !important;
            border-bottom: 1px solid var(--gray-200) !important;
        }
        
        .gr-examples td {
            background: var(--white) !important; color: var(--gray-700) !important;
            padding: 10px 8px !important; border-bottom: 1px solid var(--gray-100) !important;
        }
        
        .gr-examples tr:hover td { background: var(--gray-50) !important; }
        
        .gr-plot {
            background: var(--white) !important; border: 1px solid var(--gray-200) !important;
            border-radius: 12px !important; padding: 15px !important;
        }
        """
    ) as demo:
        
        # Header principal
        gr.HTML("""
        <div class="main-header">
            <h1 style="font-size: 2.5rem; margin-bottom: 16px; font-weight: 700;">
                🏥 Clasificador Inteligente de Obesidad
            </h1>
            <h3 style="font-size: 1.2rem; margin: 0 0 12px 0; font-weight: 500; opacity: 0.9;">
                🤖 Sistema de Evaluación de Salud con IA
            </h3>
            <p style="font-size: 0.9rem; margin: 0; opacity: 0.8;">
                🚀 Powered by MLflow + FastAPI + Gradio
            </p>
        </div>
        """)
        
        # Status del modelo
        if success:
            gr.HTML(f"""
            <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); border: 1px solid #10b981; 
                        border-left: 4px solid #10b981; color: #065f46; border-radius: 12px; 
                        padding: 16px; margin: 16px 0;">
                ✅ <strong>Modelo Cargado</strong><br><small>{message}</small>
            </div>
            """)
        else:
            gr.HTML(f"""
            <div style="background: linear-gradient(135deg, #fef2f2, #fee2e2); border: 1px solid #ef4444; 
                        border-left: 4px solid #ef4444; color: #991b1b; border-radius: 12px; 
                        padding: 16px; margin: 16px 0;">
                ❌ <strong>Error del Modelo</strong><br><small>{message}</small>
            </div>
            """)
        
        with gr.Row():
            # Columna izquierda - Inputs
            with gr.Column(scale=1):
                # Datos Personales
                gr.HTML('<div class="section-personal">')
                gr.Markdown("## 👤 **Datos Personales**")
                gr.Markdown("*Información básica sobre ti*")
                
                age = gr.Number(label="🎂 Edad (años)", info="Tu edad actual", minimum=10, maximum=90, value=25)
                gender = gr.Radio(label="👥 Género", choices=["Male", "Female"], value="Female")
                height = gr.Slider(label="📏 Altura (metros)", info="Tu altura en metros", minimum=1.2, maximum=2.2, step=0.01, value=1.70)
                weight = gr.Slider(label="⚖️ Peso (kg)", info="Tu peso actual", minimum=30, maximum=250, step=1, value=70)
                gr.HTML('</div>')
                
                # Hábitos Alimenticios
                gr.HTML('<div class="section-alimenticios">')
                gr.Markdown("## 🍎 **Hábitos Alimenticios**")
                gr.Markdown("*Información sobre tus patrones de alimentación*")
                
                fcvc = gr.Slider(label="🥬 Consumo de Vegetales", info="1=Bajo, 2=Medio, 3=Alto", minimum=1, maximum=3, step=1, value=2)
                ncp = gr.Slider(label="🍽️ Comidas Principales", info="Número de comidas al día", minimum=1, maximum=6, step=1, value=3)
                ch2o = gr.Slider(label="💧 Consumo de Agua", info="1=Bajo, 2=Medio, 3=Alto", minimum=1, maximum=3, step=1, value=2)
                favc = gr.Radio(label="🍟 ¿Consumes comida alta en calorías frecuentemente?", choices=["yes", "no"], value="no")
                caec = gr.Dropdown(label="🍿 ¿Comes entre comidas?", choices=["Sometimes", "Frequently", "Always", "no"], value="Sometimes")
                calc = gr.Radio(label="🍺 Consumo de Alcohol", choices=["yes", "no"], value="no")
                gr.HTML('</div>')
                
                # Estilo de Vida
                gr.HTML('<div class="section-lifestyle">')
                gr.Markdown("## 🏃 **Estilo de Vida**")
                gr.Markdown("*Información sobre tu actividad física y hábitos*")
                
                faf = gr.Slider(label="🏋️ Actividad Física", info="0=Ninguna, 1=Poca, 2=Moderada, 3=Intensa", minimum=0, maximum=3, step=1, value=1)
                tue = gr.Slider(label="📱 Uso de Tecnología", info="0=Bajo, 1=Medio, 2=Alto", minimum=0, maximum=2, step=1, value=1)
                mtrans = gr.Dropdown(label="🚶 Transporte Principal", choices=["Walking", "Public_Transportation", "Automobile", "Bike"], value="Walking")
                smoke = gr.Radio(label="🚭 ¿Fumas?", choices=["yes", "no"], value="no")
                scc = gr.Radio(label="📊 ¿Monitoreas las calorías que consumes?", choices=["yes", "no"], value="no")
                family_history = gr.Radio(label="👨‍👩‍👧‍👦 ¿Historia familiar de sobrepeso?", choices=["yes", "no"], value="no")
                gr.HTML('</div>')
                
                predict_btn = gr.Button("🎯 Analizar mi Estado de Salud", variant="primary", size="lg")
            
            # Columna derecha - Outputs
            with gr.Column(scale=1):
                gr.HTML('<div class="section-results">')
                gr.Markdown("## 🎯 **Resultados del Análisis**")
                gr.Markdown("*Tu evaluación personalizada aparecerá aquí*")
                
                result_text = gr.HTML(value="👈 Completa los datos y haz clic en 'Analizar' para obtener tu evaluación")
                prob_plot = gr.Plot(label="📊 Distribución de Probabilidades")
                imc_plot = gr.Plot(label="📏 Análisis de IMC")
                recommendations = gr.HTML(label="💡 Recomendaciones Personalizadas")
                gr.HTML('</div>')
        
        # Ejemplos predefinidos
        gr.Markdown("## 🔬 **Casos de Ejemplo**")
        
        gr.Examples(
            examples=[
                [25, 1.70, 65, 3, 3, 3, 2, 1, "Female", "no", "no", "Sometimes", "no", "no", "no", "Walking"],
                [35, 1.75, 85, 1, 4, 1, 0, 2, "Male", "yes", "yes", "Frequently", "no", "no", "Sometimes", "Automobile"],
                [45, 1.60, 95, 1, 5, 1, 0, 2, "Female", "yes", "yes", "Always", "yes", "no", "Always", "Public_Transportation"]
            ],
            inputs=[age, height, weight, fcvc, ncp, ch2o, faf, tue, gender, family_history, favc, caec, smoke, scc, calc, mtrans],
            label="Haz clic en un ejemplo para cargar los datos:"
        )
        
        # Footer
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #f1f5f9, #e2e8f0); border-radius: 16px; 
                    padding: 2rem; margin-top: 2rem; text-align: center; border: 1px solid #cbd5e1;">
            <div style="background: #ffffff; border-radius: 12px; padding: 20px; margin-bottom: 16px; 
                       box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);">
                <p style="margin: 0 0 10px 0; font-weight: 600; color: #dc2626;">⚠️ Descargo de Responsabilidad</p>
                <p style="margin: 0 0 10px 0; color: #475569;">Esta herramienta es solo para fines educativos y de referencia. No sustituye el consejo médico profesional.</p>
                <p style="margin: 0; color: #475569;">Siempre consulta con un profesional de la salud para evaluaciones médicas.</p>
            </div>
            <p style="margin: 0; color: #64748b; font-size: 14px;"><strong>🚀 Proyecto MLOps</strong> | Desarrollado con MLflow + FastAPI + Gradio</p>
        </div>
        """)
        
        # Conectar predicción
        predict_btn.click(
            fn=predict_obesity,
            inputs=[age, height, weight, fcvc, ncp, ch2o, faf, tue, gender, family_history, favc, caec, smoke, scc, calc, mtrans],
            outputs=[result_text, prob_plot, imc_plot, recommendations]
        )
        
    return demo

def main():
    """Ejecutar la aplicación Gradio"""
    demo = create_gradio_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, debug=True, show_error=True)

if __name__ == "__main__":
    main()