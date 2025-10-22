# src/serving/gradio_app.py
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
from plotly.subplots import make_subplots

# Variables globales para modelo y preprocessors
model = None
encoder = None
scaler = None
feature_names = None
class_mapping = None

def load_model_artifacts():
    """Cargar modelo y artefactos de preprocessing"""
    global model, encoder, scaler, feature_names, class_mapping
    
    try:
        # Paths de artefactos
        model_path = Path("models/mlflow_model")
        features_path = Path("models/features")
        
        if not model_path.exists():
            return False, f"Modelo no encontrado en {model_path}"
        
        # Cargar modelo MLflow
        model = mlflow.sklearn.load_model(str(model_path))
        
        # Cargar preprocessors
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

def preprocess_input_gradio(age, height, weight, fcvc, ncp, ch2o, faf, tue,
                           gender, family_history, favc, caec, smoke, scc, calc, mtrans):
    """Preprocessar datos de entrada para Gradio"""
    try:
        # Crear DataFrame con los datos
        data = {
            'Age': age,
            'Height': height,
            'Weight': weight,
            'FCVC': fcvc,
            'NCP': ncp,
            'CH2O': ch2o,
            'FAF': faf,
            'TUE': tue,
            'Gender': gender,
            'family_history_with_overweight': family_history,
            'FAVC': favc,
            'CAEC': caec,
            'SMOKE': smoke,
            'SCC': scc,
            'CALC': calc,
            'MTRANS': mtrans
        }
        
        df = pd.DataFrame([data])
        
        # Separar variables categóricas y numéricas
        categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 
                          'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        
        df_cat = df[categorical_cols].copy()
        df_num = df[numerical_cols].copy()
        
        # Aplicar preprocessing
        if encoder is not None:
            df_cat_encoded = encoder.transform(df_cat)
            if hasattr(df_cat_encoded, 'toarray'):
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
        df_processed = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
        
        return df_processed.values
        
    except Exception as e:
        raise Exception(f"Error en preprocessing: {str(e)}")

def predict_obesity(age, height, weight, fcvc, ncp, ch2o, faf, tue,
                   gender, family_history, favc, caec, smoke, scc, calc, mtrans):
    """Función principal de predicción para Gradio"""
    
    if model is None:
        return "❌ Error: Modelo no cargado", None, None, None
    
    try:
        # Calcular IMC para mostrar
        imc = weight / (height ** 2)
        
        # Preprocessar datos
        X = preprocess_input_gradio(age, height, weight, fcvc, ncp, ch2o, faf, tue,
                                   gender, family_history, favc, caec, smoke, scc, calc, mtrans)
        
        # Predicción
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        # Mapear clases si hay mapping disponible
        if class_mapping:
            inv_mapping = {v: k for k, v in class_mapping.items()}
            prediction_name = inv_mapping.get(prediction, str(prediction))
        else:
            prediction_name = str(prediction)
        
        # Determinar nivel de riesgo y color
        risk_mapping = {
            'Normal_Weight': ('🟢 Bajo', 'green'),
            'Insufficient_Weight': ('🟡 Medio', 'orange'),
            'Overweight_Level_I': ('🟡 Medio', 'orange'),
            'Overweight_Level_II': ('🟠 Alto', 'darkorange'),
            'Obesity_Type_I': ('🔴 Muy Alto', 'red'),
            'Obesity_Type_II': ('🔴 Crítico', 'darkred'),
            'Obesity_Type_III': ('⚫ Extremo', 'black')
        }
        
        risk_level, risk_color = risk_mapping.get(prediction_name, ('🟡 Desconocido', 'gray'))
        
        # Crear resultado principal
        confidence = float(np.max(prediction_proba))
        
        # Formatear resultado principal
        result_text = f"""
        ## 🎯 **Resultado de la Predicción**
        
        ### 📊 **Clasificación:** {prediction_name.replace('_', ' ').title()}
        ### 🎚️ **Confianza:** {confidence:.1%}
        ### ⚠️ **Nivel de Riesgo:** {risk_level}
        ### 📏 **IMC Calculado:** {imc:.1f} kg/m²
        
        ---
        
        ### 📈 **Interpretación:**
        """
        
        if prediction_name == 'Normal_Weight':
            result_text += "✅ **Excelente!** Tu peso está dentro del rango saludable."
        elif 'Insufficient' in prediction_name:
            result_text += "⚠️ **Atención:** Peso por debajo del rango recomendado. Consulta con un profesional de la salud."
        elif 'Overweight' in prediction_name:
            result_text += "⚠️ **Cuidado:** Sobrepeso detectado. Considera ajustar dieta y ejercicio."
        elif 'Obesity' in prediction_name:
            result_text += "🚨 **Importante:** Obesidad detectada. Recomendamos consultar con un médico especialista."
        
        # Crear gráfico de probabilidades
        class_names = []
        probabilities = []
        
        for i, prob in enumerate(prediction_proba):
            if class_mapping:
                inv_mapping = {v: k for k, v in class_mapping.items()}
                class_name = inv_mapping.get(i, f"class_{i}")
            else:
                class_name = f"Clase_{i}"
            
            class_names.append(class_name.replace('_', ' ').title())
            probabilities.append(prob)
        
        # Crear gráfico con colores personalizados y modernos
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#f97316', '#ef4444', '#dc2626', '#7c3aed']
        
        fig = px.bar(
            x=probabilities,
            y=class_names,
            orientation='h',
            title='🎯 Probabilidades por Clase de Obesidad',
            labels={'x': 'Probabilidad (%)', 'y': 'Clasificación'},
            color=class_names,
            color_discrete_sequence=colors
        )
        
        fig.update_layout(
            title=dict(
                text='📊 Distribución de Probabilidades por Categoría',
                x=0.5,
                font=dict(size=18, color='#111827', family='Inter', weight='bold')
            ),
            height=420,
            showlegend=False,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(family='Inter', color='#374151', size=12),
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis=dict(
                title='Probabilidad (%)',
                titlefont=dict(size=14, color='#374151'),
                tickfont=dict(size=11, color='#6b7280'),
                gridcolor='#f3f4f6',
                range=[0, 100]
            ),
            yaxis=dict(
                title='Clasificación de Obesidad',
                titlefont=dict(size=14, color='#374151'),
                tickfont=dict(size=11, color='#6b7280'),
                gridcolor='#f3f4f6'
            ),
            coloraxis_colorbar=dict(
                title="Probabilidad",
                titlefont=dict(size=12, color='#374151'),
                tickfont=dict(size=10, color='#6b7280')
            )
        )
        
        # Agregar línea vertical para la predicción con mejor estilo
        max_prob_idx = np.argmax(prediction_proba)
        fig.add_vline(
            x=probabilities[max_prob_idx],
            line_dash="dash",
            line_color="#ef4444",
            line_width=3,
            annotation=dict(
                text="🎯 Predicción",
                font=dict(size=12, color="#ef4444", family='Inter'),
                bgcolor="rgba(239, 68, 68, 0.1)",
                bordercolor="#ef4444",
                borderwidth=1
            )
        )
        
        # Crear gráfico de IMC con paleta optimizada
        imc_ranges = {
            'Bajo peso': (0, 18.5, '#3b82f6'),      # Azul moderno
            'Normal': (18.5, 24.9, '#10b981'),      # Verde éxito
            'Sobrepeso': (25, 29.9, '#f59e0b'),     # Amarillo/naranja
            'Obesidad I': (30, 34.9, '#f97316'),    # Naranja
            'Obesidad II': (35, 39.9, '#ef4444'),   # Rojo suave
            'Obesidad III': (40, 50, '#dc2626')     # Rojo intenso
        }
        
        fig_imc = go.Figure()
        
        for i, (categoria, (min_val, max_val, color)) in enumerate(imc_ranges.items()):
            fig_imc.add_trace(go.Bar(
                x=[categoria],
                y=[max_val - min_val],
                base=[min_val],
                name=categoria,
                marker_color=color,
                opacity=0.7
            ))
        
        # Agregar punto del usuario con diseño mejorado
        fig_imc.add_trace(go.Scatter(
            x=['Tu IMC'],
            y=[imc],
            mode='markers+text',
            marker=dict(
                size=24, 
                color='#1e40af',  # Azul intenso
                line=dict(width=4, color='white'),
                symbol='star'  # Estrella para destacar
            ),
            text=f'🎯 {imc:.1f}',
            textposition="top center",
            textfont=dict(size=14, color='#1e40af', family='Inter'),
            name=f'Tu IMC: {imc:.1f}'
        ))
        
        fig_imc.update_layout(
            title=dict(
                text='📏 Tu IMC en Contexto',
                x=0.5,
                font=dict(size=18, color='#111827', family='Inter', weight='bold')
            ),
            yaxis=dict(
                title='IMC (kg/m²)',
                titlefont=dict(size=14, color='#374151', family='Inter'),
                tickfont=dict(size=12, color='#6b7280'),
                gridcolor='#f3f4f6',
                zerolinecolor='#e5e7eb'
            ),
            xaxis=dict(
                titlefont=dict(size=14, color='#374151', family='Inter'),
                tickfont=dict(size=12, color='#6b7280'),
                gridcolor='#f3f4f6'
            ),
            height=350,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color='#374151')
            ),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            margin=dict(l=30, r=30, t=60, b=80),
            font=dict(family='Inter', color='#374151')
        )
        
        # Crear recomendaciones
        recommendations = generar_recomendaciones(prediction_name, imc, faf, fcvc)
        
        return result_text, fig, fig_imc, recommendations
        
    except Exception as e:
        return f"❌ Error en predicción: {str(e)}", None, None, None

def generar_recomendaciones(prediction, imc, faf, fcvc):
    """Generar recomendaciones personalizadas"""
    recomendaciones = "## 💡 **Recomendaciones Personalizadas**\n\n"
    
    # Recomendaciones según clasificación
    if prediction == 'Normal_Weight':
        recomendaciones += "✅ **¡Mantén tu estilo de vida saludable!**\n"
        recomendaciones += "- Continúa con tu rutina actual\n"
        recomendaciones += "- Mantén una dieta balanceada\n"
    elif 'Overweight' in prediction or 'Obesity' in prediction:
        recomendaciones += "🎯 **Plan de acción recomendado:**\n"
        recomendaciones += "- Consulta con un nutricionista\n"
        recomendaciones += "- Incrementa la actividad física gradualmente\n"
        recomendaciones += "- Reduce el consumo de alimentos procesados\n"
    
    # Recomendaciones según actividad física
    if faf < 1:
        recomendaciones += "\n🏃‍♀️ **Actividad Física:**\n"
        recomendaciones += "- Comienza con caminatas de 30 minutos diarios\n"
        recomendaciones += "- Incorpora ejercicios de fuerza 2-3 veces por semana\n"
    elif faf >= 2:
        recomendaciones += "\n💪 **Excelente nivel de actividad física!**\n"
        recomendaciones += "- Mantén tu rutina de ejercicio\n"
        recomendaciones += "- Varía los tipos de ejercicio para evitar monotonía\n"
    
    # Recomendaciones según consumo de vegetales
    if fcvc < 2:
        recomendaciones += "\n🥬 **Nutrición:**\n"
        recomendaciones += "- Aumenta el consumo de frutas y verduras\n"
        recomendaciones += "- Intenta incluir vegetales en cada comida\n"
        recomendaciones += "- Prueba nuevas recetas saludables\n"
    
    recomendaciones += "\n⚠️ **Nota:** Estas son recomendaciones generales. Siempre consulta con profesionales de la salud para un plan personalizado."
    
    return recomendaciones

# Cargar modelo al importar
success, message = load_model_artifacts()
if not success:
    print(f"⚠️ Advertencia: {message}")

# Crear interfaz Gradio
def create_gradio_interface():
    """Crear la interfaz de Gradio"""
    
    # Tema profesional y consistente
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono")
    )
    
    with gr.Blocks(
        theme=theme,
        title="🏥 Clasificador de Obesidad MLOps",
        css="""
        /* ===== PALETA DE COLORES OPTIMIZADA ===== */
        :root {
            --primary-blue: #2563eb;
            --primary-blue-hover: #1d4ed8;
            --success-green: #10b981;
            --success-bg: #ecfdf5;
            --error-red: #ef4444;
            --error-bg: #fef2f2;
            --text-primary: #111827;
            --text-secondary: #6b7280;
            --border-light: #e5e7eb;
            --bg-white: #ffffff;
            --bg-light: #f9fafb;
        }
        
        /* ===== CONTENEDOR PRINCIPAL ===== */
        .gradio-container, .app, body, html {
            background-color: var(--bg-white) !important;
            color: var(--text-primary) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }
        
        /* ===== BLOQUES Y PANELES ===== */
        .block, .gr-box, .gr-form, .gr-panel, .gr-group {
            background-color: var(--bg-white) !important;
            border: 1px solid var(--border-light) !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* ===== HEADER PRINCIPAL ===== */
        .main-header {
            background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-blue-hover) 100%) !important;
            color: white !important;
            text-align: center;
            padding: 30px;
            border-radius: 16px;
            margin: 20px 0;
            box-shadow: 0 8px 25px rgba(37, 99, 235, 0.15);
            border: none !important;
        }
        
        .main-header h1, .main-header h2, .main-header * {
            color: white !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        
        /* ===== TIPOGRAFÍA ===== */
        label, .label {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            margin-bottom: 8px !important;
        }
        
        p, span, div {
            color: var(--text-primary) !important;
            line-height: 1.5 !important;
        }
        
        .markdown h2, .markdown h3 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            margin: 16px 0 12px 0 !important;
        }
        
        /* ===== INPUTS Y CONTROLES ===== */
        input, textarea, select {
            background-color: var(--bg-white) !important;
            color: var(--text-primary) !important;
            border: 2px solid var(--border-light) !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-size: 14px !important;
            transition: border-color 0.2s ease !important;
        }
        
        input:focus, textarea:focus, select:focus {
            border-color: var(--primary-blue) !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
            outline: none !important;
        }
        
        /* ===== SLIDERS ===== */
        .gr-slider {
            background-color: var(--bg-white) !important;
            padding: 20px !important;
        }
        
        .gr-slider input[type="range"] {
            background: linear-gradient(to right, var(--primary-blue) 0%, var(--border-light) 0%) !important;
            border-radius: 8px !important;
            height: 8px !important;
        }
        
        .gr-slider .gr-slider-track {
            background-color: var(--border-light) !important;
        }
        
        /* ===== RADIO BUTTONS ===== */
        .gr-radio label {
            background-color: var(--bg-white) !important;
            border: 2px solid var(--border-light) !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
            margin: 4px !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }
        
        .gr-radio input:checked + label {
            background-color: var(--primary-blue) !important;
            color: white !important;
            border-color: var(--primary-blue) !important;
        }
        
        /* ===== BOTONES ===== */
        .gr-button, button {
            background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-blue-hover) 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 14px 28px !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
        }
        
        .gr-button:hover, button:hover {
            background: linear-gradient(135deg, var(--primary-blue-hover) 0%, #1e40af 100%) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4) !important;
        }
        
        /* ===== GRÁFICOS ===== */
        .gr-plot, .plotly, .plotly-graph-div {
            background-color: var(--bg-white) !important;
            border-radius: 12px !important;
            border: 1px solid var(--border-light) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        }
        
        /* ===== STATUS CARDS ===== */
        .success-status {
            background: var(--success-bg) !important;
            border: 2px solid var(--success-green) !important;
            border-radius: 12px !important;
            padding: 20px !important;
            margin: 16px 0 !important;
        }
        
        .success-status * {
            color: #065f46 !important;
            font-weight: 600 !important;
        }
        
        .error-status {
            background: var(--error-bg) !important;
            border: 2px solid var(--error-red) !important;
            border-radius: 12px !important;
            padding: 20px !important;
            margin: 16px 0 !important;
        }
        
        .error-status * {
            color: #991b1b !important;
            font-weight: 600 !important;
        }
        
        /* ===== SECCIONES Y CONTENEDORES ===== */
        .gr-column, .gr-row {
            background-color: transparent !important;
        }
        
        .gr-group {
            background-color: var(--bg-light) !important;
            border: 2px solid var(--border-light) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            margin: 16px 0 !important;
        }
        
        /* ===== MEJORAS DE ACCESIBILIDAD ===== */
        .gr-button:focus, button:focus {
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.5) !important;
            outline: none !important;
        }
        
        /* ===== EMOJIS Y ICONOS ===== */
        .emoji-icon {
            font-size: 1.2em !important;
            margin-right: 8px !important;
        }
        
        /* ===== RESPONSIVE DESIGN ===== */
        @media (max-width: 768px) {
            .gr-button, button {
                padding: 12px 20px !important;
                font-size: 14px !important;
            }
            
            .main-header {
                padding: 20px !important;
            }
            
            .gr-group {
                padding: 16px !important;
            }
        }
        """
    ) as demo:
        
        # Header principal con diseño mejorado
        gr.HTML("""
        <div class="main-header">
            <h1 style="font-size: 2.8rem; margin-bottom: 16px; font-weight: 800; letter-spacing: -0.02em;">
                🏥 Clasificador Inteligente de Obesidad
            </h1>
            <h3 style="font-size: 1.4rem; margin: 0 0 12px 0; font-weight: 600; opacity: 0.95;">
                🤖 Sistema de Evaluación de Salud con IA
            </h3>
            <p style="font-size: 1rem; margin: 0; opacity: 0.85; font-weight: 500;">
                🚀 Powered by MLflow + FastAPI + Gradio
            </p>
        </div>
        """)
        
        # Estado del modelo
        with gr.Row():
            status_class = "status-card success-status" if success else "status-card error-status"
            gr.HTML(f"""
            <div class="{status_class}">
                <div style="text-align: center;">
                    <h4 style="margin: 0 0 5px 0;">
                        {'✅ Modelo Cargado' if success else '❌ Error de Modelo'}
                    </h4>
                    <p style="margin: 0; font-size: 14px;">{message}</p>
                </div>
            </div>
            """)
        
        with gr.Row():
            # Columna izquierda - Inputs
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="background: linear-gradient(90deg, #f0f9ff 0%, #e0f2fe 100%); 
                           border-left: 4px solid #3b82f6; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
                    <h2 style="color: #1e40af; font-size: 1.4rem; margin: 0; font-weight: 700;">
                        📝 Datos Personales
                    </h2>
                    <p style="color: #475569; margin: 8px 0 0 0; font-size: 0.9rem;">
                        Información básica sobre ti
                    </p>
                </div>
                """)
                
                with gr.Group():
                    age = gr.Slider(
                        minimum=10, maximum=90, value=25, step=1,
                        label="🎂 Edad (años)", info="Tu edad actual"
                    )
                    gender = gr.Radio(
                        choices=["Male", "Female"], value="Male",
                        label="👤 Género"
                    )
                    height = gr.Slider(
                        minimum=1.20, maximum=2.20, value=1.70, step=0.01,
                        label="📏 Altura (metros)", info="Tu altura en metros"
                    )
                    weight = gr.Slider(
                        minimum=30, maximum=250, value=70, step=1,
                        label="⚖️ Peso (kg)", info="Tu peso actual"
                    )
                
                gr.HTML("""
                <div style="background: linear-gradient(90deg, #f0fdf4 0%, #dcfce7 100%); 
                           border-left: 4px solid #10b981; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
                    <h2 style="color: #047857; font-size: 1.4rem; margin: 0; font-weight: 700;">
                        🍎 Hábitos Alimenticios
                    </h2>
                    <p style="color: #475569; margin: 8px 0 0 0; font-size: 0.9rem;">
                        Información sobre tus patrones de alimentación
                    </p>
                </div>
                """)
                
                with gr.Group():
                    fcvc = gr.Slider(
                        minimum=1, maximum=3, value=2, step=1,
                        label="🥬 Consumo de Vegetales", info="1=Bajo, 2=Medio, 3=Alto"
                    )
                    ncp = gr.Slider(
                        minimum=1, maximum=6, value=3, step=1,
                        label="🍽️ Comidas Principales", info="Número de comidas al día"
                    )
                    ch2o = gr.Slider(
                        minimum=1, maximum=3, value=2, step=1,
                        label="💧 Consumo de Agua", info="1=Bajo, 2=Medio, 3=Alto"
                    )
                    favc = gr.Radio(
                        choices=["yes", "no"], value="no",
                        label="🍟 ¿Consumes comida alta en calorías frecuentemente?"
                    )
                    caec = gr.Dropdown(
                        choices=["no", "Sometimes", "Frequently", "Always"],
                        value="Sometimes",
                        label="🍪 ¿Comes entre comidas?"
                    )
                    calc = gr.Dropdown(
                        choices=["no", "Sometimes", "Frequently", "Always"],
                        value="no",
                        label="🍷 Consumo de Alcohol"
                    )
                
                gr.Markdown("## 💪 **Estilo de Vida**")
                
                with gr.Group():
                    faf = gr.Slider(
                        minimum=0, maximum=3, value=1, step=1,
                        label="🏃‍♀️ Actividad Física", info="0=Ninguna, 1=Poca, 2=Moderada, 3=Intensa"
                    )
                    tue = gr.Slider(
                        minimum=0, maximum=2, value=1, step=1,
                        label="📱 Uso de Tecnología", info="0=Bajo, 1=Medio, 2=Alto"
                    )
                    mtrans = gr.Dropdown(
                        choices=["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"],
                        value="Walking",
                        label="🚗 Transporte Principal"
                    )
                    smoke = gr.Radio(
                        choices=["yes", "no"], value="no",
                        label="🚭 ¿Fumas?"
                    )
                    scc = gr.Radio(
                        choices=["yes", "no"], value="no",
                        label="📊 ¿Monitoreas las calorías que consumes?"
                    )
                    family_history = gr.Radio(
                        choices=["yes", "no"], value="no",
                        label="👨‍👩‍👧‍👦 ¿Historia familiar de sobrepeso?"
                    )
                
                # Botón de predicción
                predict_btn = gr.Button(
                    "🎯 Analizar mi Estado de Salud",
                    variant="primary",
                    size="lg"
                )
            
            # Columna derecha - Outputs
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="background: linear-gradient(90deg, #fef3c7 0%, #fde68a 100%); 
                           border-left: 4px solid #f59e0b; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
                    <h2 style="color: #92400e; font-size: 1.4rem; margin: 0; font-weight: 700;">
                        🎯 Resultados del Análisis
                    </h2>
                    <p style="color: #475569; margin: 8px 0 0 0; font-size: 0.9rem;">
                        Tu evaluación personalizada aparecerá aquí
                    </p>
                </div>
                """)
                
                result_text = gr.Markdown(
                    value="👈 Completa los datos y haz clic en '🎯 Analizar mi Estado de Salud' para obtener tu evaluación"
                )
                
                prob_plot = gr.Plot(
                    label="📊 Distribución de Probabilidades"
                )
                
                imc_plot = gr.Plot(
                    label="📏 Análisis de IMC"
                )
                
                recommendations = gr.Markdown(
                    label="💡 Recomendaciones Personalizadas"
                )
        
        # Ejemplos predefinidos
        gr.Markdown("## 🔬 **Casos de Ejemplo**")
        
        with gr.Row():
            gr.Examples(
                examples=[
                    # Persona saludable
                    [25, 1.70, 65, 3, 3, 3, 2, 1, "Female", "no", "no", "Sometimes", "no", "yes", "no", "Walking"],
                    # Persona con sobrepeso
                    [35, 1.75, 85, 1, 4, 1, 0, 2, "Male", "yes", "yes", "Frequently", "no", "no", "Sometimes", "Automobile"],
                    # Persona obesa
                    [45, 1.60, 95, 1, 5, 1, 0, 2, "Female", "yes", "yes", "Always", "yes", "no", "Always", "Public_Transportation"]
                ],
                inputs=[age, height, weight, fcvc, ncp, ch2o, faf, tue, gender, family_history, favc, caec, smoke, scc, calc, mtrans],
                label="Haz clic en un ejemplo para cargar los datos:"
            )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 25px; 
                    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); 
                    border-radius: 15px; border: 1px solid #cbd5e1;">
            <div style="background: white; padding: 20px; border-radius: 10px; 
                       box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 15px;">
                <p style="margin: 0 0 10px 0; font-weight: 600; color: #dc2626;">
                    ⚠️ Descargo de Responsabilidad
                </p>
                <p style="margin: 0 0 10px 0; color: #475569;">
                    Esta herramienta es solo para fines educativos y de referencia. No sustituye el consejo médico profesional.
                </p>
                <p style="margin: 0; color: #475569;">
                    Siempre consulta con un profesional de la salud para evaluaciones médicas.
                </p>
            </div>
            <p style="margin: 0; color: #64748b; font-size: 14px;">
                <strong>🚀 Proyecto MLOps</strong> | Desarrollado con MLflow + FastAPI + Gradio
            </p>
        </div>
        """)
        
        # Conectar la función de predicción
        predict_btn.click(
            fn=predict_obesity,
            inputs=[age, height, weight, fcvc, ncp, ch2o, faf, tue,
                   gender, family_history, favc, caec, smoke, scc, calc, mtrans],
            outputs=[result_text, prob_plot, imc_plot, recommendations]
        )
        
    return demo

# Función principal para ejecutar la aplicación
def main():
    """Ejecutar la aplicación Gradio"""
    demo = create_gradio_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()