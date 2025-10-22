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
        return "❌ Modelo no disponible", None, None, "No se pudo cargar el modelo"
    
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
        
        # Preprocessing
        if encoder is not None:
            categorical_columns = ['Gender', 'family_history_with_overweight', 
                                 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = encoder[col].transform(df[col])
        
        if scaler is not None:
            df_scaled = scaler.transform(df)
            df = pd.DataFrame(df_scaled, columns=df.columns)
        
        # Predicción
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]
        
        # Mapear predicción
        if class_mapping:
            prediction_name = [k for k, v in class_mapping.items() if v == prediction][0]
        else:
            prediction_name = str(prediction)
        
        confidence = prediction_proba.max() * 100
        
        # Resultado simple
        result_text = f"""
        ### 🎯 Resultado del Análisis
        
        **Clasificación:** {prediction_name}  
        **Confianza:** {confidence:.1f}%  
        **IMC:** {imc:.1f} kg/m²
        """
        
        # Gráfico simple de probabilidades
        class_names = ['Peso_Insuficiente', 'Normal', 'Sobrepeso_Nivel_I', 
                      'Sobrepeso_Nivel_II', 'Obesidad_Tipo_I', 'Obesidad_Tipo_II', 'Obesidad_Tipo_III']
        
        colors = ['#10b981', '#22c55e', '#84cc16', '#eab308', '#f59e0b', '#ef4444', '#dc2626']
        
        fig = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=prediction_proba,
                marker_color=colors,
                text=[f'{prob:.1%}' for prob in prediction_proba],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Distribución de Probabilidades',
            xaxis_title='Clasificación',
            yaxis_title='Probabilidad',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(tickangle=45)
        )
        
        # Gráfico de IMC simple
        fig_imc = go.Figure()
        
        # Rangos de IMC
        ranges = [(0, 18.5, 'Bajo peso', '#60a5fa'),
                 (18.5, 24.9, 'Normal', '#34d399'),
                 (25, 29.9, 'Sobrepeso', '#fbbf24'),
                 (30, 34.9, 'Obesidad I', '#fb923c'),
                 (35, 39.9, 'Obesidad II', '#f87171'),
                 (40, 50, 'Obesidad III', '#ef4444')]
        
        for min_val, max_val, nombre, color in ranges:
            fig_imc.add_trace(go.Bar(
                x=[nombre],
                y=[max_val - min_val],
                base=[min_val],
                name=nombre,
                marker_color=color,
                opacity=0.7
            ))
        
        # Punto del usuario
        fig_imc.add_trace(go.Scatter(
            x=['Tu IMC'],
            y=[imc],
            mode='markers',
            marker=dict(size=12, color='red'),
            name=f'Tu IMC: {imc:.1f}'
        ))
        
        fig_imc.update_layout(
            title='Tu IMC en Contexto',
            yaxis_title='IMC (kg/m²)',
            height=300,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Recomendaciones simples
        recommendations = generar_recomendaciones_simples(prediction_name, imc)
        
        return result_text, fig, fig_imc, recommendations
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, "Error en la predicción"

def generar_recomendaciones_simples(prediction, imc):
    """Generar recomendaciones simples"""
    
    recomendaciones = "### 💡 Recomendaciones\n\n"
    
    if "Normal" in prediction:
        recomendaciones += "✅ **¡Excelente!** Mantén tu estilo de vida saludable.\n"
        recomendaciones += "- Continúa con ejercicio regular\n"
        recomendaciones += "- Mantén una dieta balanceada\n"
    elif "Sobrepeso" in prediction or "Obesidad" in prediction:
        recomendaciones += "⚠️ **Atención:** Considera hacer algunos ajustes.\n"
        recomendaciones += "- Consulta con un profesional de salud\n"
        recomendaciones += "- Incrementa la actividad física\n"
        recomendaciones += "- Revisa tu alimentación\n"
    
    if imc < 18.5:
        recomendaciones += "\n📈 **IMC Bajo:** Considera ganar peso de forma saludable."
    elif imc > 30:
        recomendaciones += "\n🎯 **IMC Alto:** Importante consultar con un médico."
    
    recomendaciones += "\n\n⚠️ **Nota:** Esta es solo una evaluación automatizada. Consulta siempre con profesionales de la salud."
    
    return recomendaciones

# Cargar modelo
success, message = load_model_artifacts()

def create_gradio_interface():
    """Crear interfaz Gradio simple y clara"""
    
    with gr.Blocks(title="🏥 Clasificador de Obesidad") as demo:
        
        gr.Markdown("# 🏥 Clasificador de Obesidad")
        gr.Markdown("Sistema de evaluación de salud con IA")
        
        # Status del modelo
        if success:
            gr.Markdown(f"✅ **Modelo cargado:** {message}")
        else:
            gr.Markdown(f"❌ **Error:** {message}")
        
        with gr.Row():
            # Columna de inputs
            with gr.Column():
                gr.Markdown("## 👤 Datos Personales")
                
                age = gr.Number(label="Edad (años)", value=25, minimum=10, maximum=90)
                gender = gr.Radio(label="Género", choices=["Male", "Female"], value="Female")
                height = gr.Slider(label="Altura (metros)", minimum=1.2, maximum=2.2, value=1.70, step=0.01)
                weight = gr.Slider(label="Peso (kg)", minimum=30, maximum=250, value=70, step=1)
                
                gr.Markdown("## 🍎 Hábitos Alimenticios")
                
                fcvc = gr.Slider(label="Consumo de Vegetales (1=Bajo, 3=Alto)", minimum=1, maximum=3, value=2, step=1)
                ncp = gr.Slider(label="Comidas Principales por día", minimum=1, maximum=6, value=3, step=1)
                ch2o = gr.Slider(label="Consumo de Agua (1=Bajo, 3=Alto)", minimum=1, maximum=3, value=2, step=1)
                favc = gr.Radio(label="¿Comes alimentos altos en calorías?", choices=["yes", "no"], value="no")
                caec = gr.Dropdown(label="¿Comes entre comidas?", choices=["Sometimes", "Frequently", "Always", "no"], value="Sometimes")
                calc = gr.Radio(label="¿Consumes alcohol?", choices=["yes", "no"], value="no")
                
                gr.Markdown("## 🏃 Estilo de Vida")
                
                faf = gr.Slider(label="Actividad Física (0=Ninguna, 3=Intensa)", minimum=0, maximum=3, value=1, step=1)
                tue = gr.Slider(label="Uso de Tecnología (0=Bajo, 2=Alto)", minimum=0, maximum=2, value=1, step=1)
                mtrans = gr.Dropdown(label="Transporte Principal", choices=["Walking", "Public_Transportation", "Automobile", "Bike"], value="Walking")
                smoke = gr.Radio(label="¿Fumas?", choices=["yes", "no"], value="no")
                scc = gr.Radio(label="¿Monitoreas calorías?", choices=["yes", "no"], value="no")
                family_history = gr.Radio(label="¿Historia familiar de sobrepeso?", choices=["yes", "no"], value="no")
                
                predict_btn = gr.Button("🎯 Analizar Estado de Salud", variant="primary")
            
            # Columna de resultados
            with gr.Column():
                gr.Markdown("## 🎯 Resultados")
                
                result_text = gr.Markdown("Completa los datos y haz clic en 'Analizar'")
                prob_plot = gr.Plot(label="Probabilidades por Clase")
                imc_plot = gr.Plot(label="Análisis de IMC")
                recommendations = gr.Markdown("")
        
        # Ejemplos
        gr.Markdown("## 🔬 Ejemplos")
        
        gr.Examples(
            examples=[
                [25, "Female", 1.70, 65, 3, 3, 3, "no", "Sometimes", "no", 2, 1, "Walking", "no", "no", "no"],
                [35, "Male", 1.75, 85, 1, 4, 1, "yes", "Frequently", "no", 0, 2, "Automobile", "no", "no", "yes"],
                [45, "Female", 1.60, 95, 1, 5, 1, "yes", "Always", "yes", 0, 2, "Public_Transportation", "yes", "no", "yes"]
            ],
            inputs=[age, gender, height, weight, fcvc, ncp, ch2o, favc, caec, calc, faf, tue, mtrans, smoke, scc, family_history],
            label="Casos de ejemplo:"
        )
        
        # Conectar función
        predict_btn.click(
            fn=predict_obesity,
            inputs=[age, height, weight, fcvc, ncp, ch2o, faf, tue, gender, family_history, favc, caec, smoke, scc, calc, mtrans],
            outputs=[result_text, prob_plot, imc_plot, recommendations]
        )
        
    return demo

def main():
    """Ejecutar aplicación"""
    demo = create_gradio_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()