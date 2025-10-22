#!/usr/bin/env python3
"""Test rápido de la función de predicción"""

# Agregar el directorio del proyecto al path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.serving.gradio_app_simple import predict_obesity

# Datos de prueba (en el orden correcto)
test_data = [
    25,           # age
    1.70,         # height
    65,           # weight
    3,            # fcvc
    3,            # ncp
    3,            # ch2o
    2,            # faf
    1,            # tue
    "Female",     # gender
    "no",         # family_history
    "no",         # favc
    "Sometimes",  # caec
    "no",         # smoke
    "no",         # scc
    "no",         # calc
    "Walking"     # mtrans
]

print("🧪 Probando función de predicción...")
print(f"Datos de entrada: {test_data}")

try:
    result = predict_obesity(*test_data)
    print("✅ Función ejecutada exitosamente")
    print(f"Tipo de resultado: {type(result)}")
    if isinstance(result, tuple):
        print(f"Número de elementos: {len(result)}")
        print(f"Primer elemento (texto): {result[0][:100] if result[0] else 'None'}...")
    else:
        print(f"Resultado: {result}")
        
except Exception as e:
    print(f"❌ Error en predicción: {str(e)}")
    import traceback
    traceback.print_exc()