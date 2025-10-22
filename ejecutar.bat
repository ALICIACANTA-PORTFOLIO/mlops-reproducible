@echo off
REM =============================================================
REM 🚀 EJECUTOR PASO A PASO - OPTIMIZACIONES MLOps
REM =============================================================

echo.
echo ===============================================
echo 🚀 EJECUTOR DE OPTIMIZACIONES MLOps
echo ===============================================
echo.

REM Verificar que estamos en entorno virtual
python -c "import sys; print('✅ Python:', sys.executable)" 2>nul
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python no disponible
    echo    Asegúrate de activar el entorno virtual: venv\Scripts\activate
    pause
    exit /b 1
)

echo 📋 MENÚ DE OPCIONES:
echo.
echo 1. ✅ Ejecutar TODO automáticamente
echo 2. 🧪 Solo Tests Corregidos
echo 3. 📊 Solo Demo de Optimizaciones  
echo 4. 🔧 Solo Corrección de Parámetros
echo 5. 🎯 MLflow UI (servidor)
echo 6. 📈 Verificar Resultados
echo 7. 🆘 Diagnóstico de Problemas
echo 0. ❌ Salir
echo.

set /p choice="Selecciona una opción (0-7): "

if "%choice%"=="1" goto ejecutar_todo
if "%choice%"=="2" goto solo_tests
if "%choice%"=="3" goto solo_demo
if "%choice%"=="4" goto solo_correccion
if "%choice%"=="5" goto mlflow_ui
if "%choice%"=="6" goto verificar
if "%choice%"=="7" goto diagnostico
if "%choice%"=="0" goto salir

echo ❌ Opción inválida
pause
goto menu

:ejecutar_todo
echo.
echo 🚀 EJECUTANDO TODO AUTOMÁTICAMENTE...
echo ===============================================
python ejecutar_todo.py
pause
goto menu

:solo_tests
echo.
echo 🧪 EJECUTANDO TESTS CORREGIDOS...
echo ===============================================
python test_fix_patch.py
echo.
echo ✅ Tests completados
pause
goto menu

:solo_demo
echo.
echo 📊 EJECUTANDO DEMO DE OPTIMIZACIONES...
echo ===============================================
python demo_optimizations.py
echo.
echo ✅ Demo completado
pause
goto menu

:solo_correccion
echo.
echo 🔧 EJECUTANDO CORRECCIÓN DE PARÁMETROS...
echo ===============================================
python fix_make_classification.py
echo.
echo ✅ Corrección completada
pause
goto menu

:mlflow_ui
echo.
echo 🎯 INICIANDO MLFLOW UI...
echo ===============================================
echo Abriendo servidor en: http://127.0.0.1:5000
echo Presiona Ctrl+C para detener
echo.
mlflow ui --host 127.0.0.1 --port 5000
pause
goto menu

:verificar
echo.
echo 📈 VERIFICANDO RESULTADOS...
echo ===============================================

echo 📁 Archivos generados:
if exist "data\interim\obesity_synthetic_corrected.csv" (
    echo   ✅ obesity_synthetic_corrected.csv
) else (
    echo   ❌ obesity_synthetic_corrected.csv [FALTANTE]
)

if exist "mlruns" (
    echo   ✅ Directorio mlruns
) else (
    echo   ❌ Directorio mlruns [FALTANTE]
)

if exist "reports" (
    echo   ✅ Directorio reports  
) else (
    echo   ❌ Directorio reports [FALTANTE]
)

echo.
echo 📊 Verificando MLflow:
python -c "import mlflow; client = mlflow.MlflowClient(); experiments = client.search_experiments(); print(f'Experimentos: {len(experiments)}'); models = client.search_registered_models(); print(f'Modelos registrados: {len(models)}')" 2>nul
if %errorlevel% neq 0 (
    echo   ❌ Error accediendo a MLflow
) else (
    echo   ✅ MLflow funcionando
)

pause
goto menu

:diagnostico
echo.
echo 🆘 DIAGNÓSTICO DE PROBLEMAS...
echo ===============================================

echo 🔍 Versiones de librerías:
python -c "import sklearn; print(f'sklearn: {sklearn.__version__}')" 2>nul
python -c "import pandas as pd; print(f'pandas: {pd.__version__}')" 2>nul  
python -c "import mlflow; print(f'mlflow: {mlflow.__version__}')" 2>nul

echo.
echo 🧪 Test básico de sklearn:
python -c "from sklearn.datasets import make_classification; X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42); print('✅ sklearn básico funciona')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ Problema con sklearn básico
) else (
    echo ✅ sklearn básico funciona
)

echo.
echo 🧪 Test de parámetros corregidos:
python -c "from sklearn.datasets import make_classification; X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=6, n_redundant=2, n_clusters_per_class=1, random_state=42); print('✅ Parámetros corregidos funcionan')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ Problema con parámetros corregidos
) else (
    echo ✅ Parámetros corregidos funcionan
)

echo.
echo 📁 Estructura de proyecto:
if exist "src" (
    echo ✅ Directorio src
) else (
    echo ❌ Directorio src [FALTANTE]
)

if exist "data" (
    echo ✅ Directorio data
) else (
    echo ❌ Directorio data [FALTANTE]  
)

if exist "test_fix_patch.py" (
    echo ✅ test_fix_patch.py
) else (
    echo ❌ test_fix_patch.py [FALTANTE]
)

if exist "demo_optimizations.py" (
    echo ✅ demo_optimizations.py
) else (
    echo ❌ demo_optimizations.py [FALTANTE]
)

pause
goto menu

:salir
echo.
echo 👋 ¡Hasta luego!
echo.
exit /b 0

:menu
cls
goto :eof