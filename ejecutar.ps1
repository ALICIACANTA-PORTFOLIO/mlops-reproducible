# =============================================================
# 🚀 EJECUTOR POWERSHELL - OPTIMIZACIONES MLOps
# =============================================================

function Show-Menu {
    Clear-Host
    Write-Host ""
    Write-Host "===============================================" -ForegroundColor Yellow
    Write-Host "🚀 EJECUTOR DE OPTIMIZACIONES MLOps" -ForegroundColor Yellow
    Write-Host "===============================================" -ForegroundColor Yellow
    Write-Host ""
    
    # Verificar Python
    try {
        $pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
        Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ ERROR: Python no disponible" -ForegroundColor Red
        Write-Host "   Asegúrate de activar el entorno virtual: .\venv\Scripts\Activate.ps1" -ForegroundColor Red
        return $false
    }
    
    Write-Host ""
    Write-Host "📋 MENÚ DE OPCIONES:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. ✅ Ejecutar TODO automáticamente"
    Write-Host "2. 🧪 Solo Tests Corregidos"
    Write-Host "3. 📊 Solo Demo de Optimizaciones"
    Write-Host "4. 🔧 Solo Corrección de Parámetros"
    Write-Host "5. 🎯 MLflow UI (servidor)"
    Write-Host "6. 📈 Verificar Resultados"
    Write-Host "7. 🆘 Diagnóstico de Problemas"
    Write-Host "0. ❌ Salir"
    Write-Host ""
    
    return $true
}

function Execute-Command {
    param(
        [string]$Command,
        [string]$Description
    )
    
    Write-Host ""
    Write-Host "🚀 $Description" -ForegroundColor Yellow
    Write-Host "===============================================" -ForegroundColor Yellow
    
    try {
        Invoke-Expression $Command
        Write-Host ""
        Write-Host "✅ $Description completado" -ForegroundColor Green
    }
    catch {
        Write-Host ""
        Write-Host "❌ Error en $Description`: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "Presiona Enter para continuar..." -ForegroundColor Gray
    Read-Host
}

function Execute-All {
    Write-Host ""
    Write-Host "🚀 EJECUTANDO TODO AUTOMÁTICAMENTE..." -ForegroundColor Yellow
    Write-Host "===============================================" -ForegroundColor Yellow
    
    python ejecutar_todo.py
    
    Write-Host ""
    Write-Host "Presiona Enter para continuar..." -ForegroundColor Gray
    Read-Host
}

function Show-MLflowUI {
    Write-Host ""
    Write-Host "🎯 INICIANDO MLFLOW UI..." -ForegroundColor Yellow
    Write-Host "===============================================" -ForegroundColor Yellow
    Write-Host "Abriendo servidor en: http://127.0.0.1:5000" -ForegroundColor Cyan
    Write-Host "Presiona Ctrl+C para detener" -ForegroundColor Gray
    Write-Host ""
    
    mlflow ui --host 127.0.0.1 --port 5000
}

function Verify-Results {
    Write-Host ""
    Write-Host "📈 VERIFICANDO RESULTADOS..." -ForegroundColor Yellow
    Write-Host "===============================================" -ForegroundColor Yellow
    
    Write-Host ""
    Write-Host "📁 Archivos generados:" -ForegroundColor Cyan
    
    if (Test-Path "data\interim\obesity_synthetic_corrected.csv") {
        Write-Host "  ✅ obesity_synthetic_corrected.csv" -ForegroundColor Green
    } else {
        Write-Host "  ❌ obesity_synthetic_corrected.csv [FALTANTE]" -ForegroundColor Red
    }
    
    if (Test-Path "mlruns") {
        Write-Host "  ✅ Directorio mlruns" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Directorio mlruns [FALTANTE]" -ForegroundColor Red
    }
    
    if (Test-Path "reports") {
        Write-Host "  ✅ Directorio reports" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Directorio reports [FALTANTE]" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "📊 Verificando MLflow:" -ForegroundColor Cyan
    
    try {
        $mlflowResult = python -c "import mlflow; client = mlflow.MlflowClient(); experiments = client.search_experiments(); print(f'Experimentos: {len(experiments)}'); models = client.search_registered_models(); print(f'Modelos registrados: {len(models)}')"
        Write-Host "  ✅ MLflow funcionando" -ForegroundColor Green
        Write-Host "  $mlflowResult" -ForegroundColor White
    }
    catch {
        Write-Host "  ❌ Error accediendo a MLflow" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "Presiona Enter para continuar..." -ForegroundColor Gray
    Read-Host
}

function Run-Diagnostics {
    Write-Host ""
    Write-Host "🆘 DIAGNÓSTICO DE PROBLEMAS..." -ForegroundColor Yellow
    Write-Host "===============================================" -ForegroundColor Yellow
    
    Write-Host ""
    Write-Host "🔍 Versiones de librerías:" -ForegroundColor Cyan
    
    try {
        $sklearnVersion = python -c "import sklearn; print(f'sklearn: {sklearn.__version__}')"
        Write-Host "  $sklearnVersion" -ForegroundColor White
    } catch {
        Write-Host "  ❌ Error con sklearn" -ForegroundColor Red
    }
    
    try {
        $pandasVersion = python -c "import pandas as pd; print(f'pandas: {pd.__version__}')"
        Write-Host "  $pandasVersion" -ForegroundColor White
    } catch {
        Write-Host "  ❌ Error con pandas" -ForegroundColor Red
    }
    
    try {
        $mlflowVersion = python -c "import mlflow; print(f'mlflow: {mlflow.__version__}')"
        Write-Host "  $mlflowVersion" -ForegroundColor White
    } catch {
        Write-Host "  ❌ Error con mlflow" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "🧪 Tests básicos:" -ForegroundColor Cyan
    
    try {
        python -c "from sklearn.datasets import make_classification; X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42); print('✅ sklearn básico funciona')"
    } catch {
        Write-Host "❌ Problema con sklearn básico" -ForegroundColor Red
    }
    
    try {
        python -c "from sklearn.datasets import make_classification; X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=6, n_redundant=2, n_clusters_per_class=1, random_state=42); print('✅ Parámetros corregidos funcionan')"
    } catch {
        Write-Host "❌ Problema con parámetros corregidos" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "📁 Estructura de proyecto:" -ForegroundColor Cyan
    
    @("src", "data", "test_fix_patch.py", "demo_optimizations.py", "ejecutar_todo.py") | ForEach-Object {
        if (Test-Path $_) {
            Write-Host "  ✅ $_" -ForegroundColor Green
        } else {
            Write-Host "  ❌ $_ [FALTANTE]" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "Presiona Enter para continuar..." -ForegroundColor Gray
    Read-Host
}

# Script principal
while ($true) {
    if (-not (Show-Menu)) {
        break
    }
    
    $choice = Read-Host "Selecciona una opción (0-7)"
    
    switch ($choice) {
        "1" { Execute-All }
        "2" { Execute-Command "python test_fix_patch.py" "Tests Corregidos" }
        "3" { Execute-Command "python demo_optimizations.py" "Demo de Optimizaciones" }
        "4" { Execute-Command "python fix_make_classification.py" "Corrección de Parámetros" }
        "5" { Show-MLflowUI }
        "6" { Verify-Results }
        "7" { Run-Diagnostics }
        "0" {
            Write-Host ""
            Write-Host "👋 ¡Hasta luego!" -ForegroundColor Green
            Write-Host ""
            break
        }
        default {
            Write-Host ""
            Write-Host "❌ Opción inválida" -ForegroundColor Red
            Write-Host "Presiona Enter para continuar..." -ForegroundColor Gray
            Read-Host
        }
    }
}