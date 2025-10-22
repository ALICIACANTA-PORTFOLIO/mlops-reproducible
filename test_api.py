#!/usr/bin/env python3
"""
Cliente de ejemplo para probar la API de clasificación de obesidad
"""
import requests
import json
from typing import Dict, Any

# URL base de la API
BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Probar health check"""
    print("🔍 Probando health check...")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        print("✅ API funcionando correctamente")
        print(f"   Status: {response.json()}")
    else:
        print(f"❌ Error en health check: {response.status_code}")
    return response.status_code == 200

def test_model_info():
    """Probar información del modelo"""
    print("\n📊 Obteniendo información del modelo...")
    response = requests.get(f"{BASE_URL}/model/info")
    if response.status_code == 200:
        info = response.json()
        print("✅ Información del modelo:")
        print(f"   Tipo: {info.get('model_type')}")
        print(f"   Features: {info.get('num_features')}")
        print(f"   Artefactos: {info.get('artifacts_status')}")
    else:
        print(f"❌ Error obteniendo info: {response.status_code}")
    return response.status_code == 200

def test_single_prediction():
    """Probar predicción individual"""
    print("\n🎯 Probando predicción individual...")
    
    # Datos de ejemplo (persona con sobrepeso)
    test_data = {
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
    
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        json=test_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Predicción exitosa:")
        print(f"   Predicción: {result['prediction']}")
        print(f"   Confianza: {result['confidence']:.3f}")
        print(f"   Nivel de riesgo: {result['risk_level']}")
        print(f"   Probabilidades por clase:")
        for clase, prob in result['prediction_proba'].items():
            print(f"     {clase}: {prob:.3f}")
    else:
        print(f"❌ Error en predicción: {response.status_code}")
        print(f"   Detalle: {response.text}")
    
    return response.status_code == 200

def test_batch_prediction():
    """Probar predicción en lote"""
    print("\n🎯 Probando predicción en lote...")
    
    # Datos de ejemplo (3 personas diferentes)
    batch_data = {
        "data": [
            {  # Persona delgada
                "Age": 22,
                "Height": 1.65,
                "Weight": 55,
                "FCVC": 3,
                "NCP": 3,
                "CH2O": 3,
                "FAF": 2,
                "TUE": 1,
                "Gender": "Female",
                "family_history_with_overweight": "no",
                "FAVC": "no",
                "CAEC": "no",
                "SMOKE": "no",
                "SCC": "yes",
                "CALC": "no",
                "MTRANS": "Walking"
            },
            {  # Persona con sobrepeso
                "Age": 35,
                "Height": 1.70,
                "Weight": 80,
                "FCVC": 1,
                "NCP": 4,
                "CH2O": 1,
                "FAF": 0,
                "TUE": 2,
                "Gender": "Male",
                "family_history_with_overweight": "yes",
                "FAVC": "yes",
                "CAEC": "Frequently",
                "SMOKE": "no",
                "SCC": "no",
                "CALC": "Frequently",
                "MTRANS": "Automobile"
            },
            {  # Persona obesa
                "Age": 45,
                "Height": 1.60,
                "Weight": 95,
                "FCVC": 1,
                "NCP": 5,
                "CH2O": 1,
                "FAF": 0,
                "TUE": 2,
                "Gender": "Female",
                "family_history_with_overweight": "yes",
                "FAVC": "yes",
                "CAEC": "Always",
                "SMOKE": "yes",
                "SCC": "no",
                "CALC": "Always",
                "MTRANS": "Public_Transportation"
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        headers={"Content-Type": "application/json"},
        json=batch_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Predicción batch exitosa:")
        print(f"   Total predicciones: {result['summary']['total_predictions']}")
        print(f"   Distribución de riesgo: {result['summary']['risk_distribution']}")
        print(f"   Confianza promedio: {result['summary']['average_confidence']:.3f}")
        
        print("\n   Predicciones individuales:")
        for i, pred in enumerate(result['predictions']):
            print(f"     Persona {i+1}: {pred['prediction']} (riesgo: {pred['risk_level']})")
    else:
        print(f"❌ Error en predicción batch: {response.status_code}")
        print(f"   Detalle: {response.text}")
    
    return response.status_code == 200

def main():
    """Ejecutar todos los tests"""
    print("🧪 Probando API de clasificación de obesidad\n")
    print("=" * 50)
    
    # Tests secuenciales
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except requests.exceptions.ConnectionError:
            print(f"❌ {test_name}: No se puede conectar a la API")
            print("   ¿Está corriendo la API en http://127.0.0.1:8000?")
            results.append((test_name, False))
            break
        except Exception as e:
            print(f"❌ {test_name}: Error inesperado - {str(e)}")
            results.append((test_name, False))
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 Resumen de tests:")
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Tests exitosos: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 ¡Todos los tests pasaron! La API está funcionando correctamente.")
    else:
        print("⚠️  Algunos tests fallaron. Revisa la configuración.")

if __name__ == "__main__":
    main()