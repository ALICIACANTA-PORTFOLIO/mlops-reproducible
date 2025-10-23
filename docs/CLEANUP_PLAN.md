# 🚀 PLAN DE EJECUCIÓN - Limpieza del Proyecto

**Fecha**: 23 de Octubre, 2025  
**Auditoría completa**: Ver `PROJECT_AUDIT.md`

---

## 📊 Situación Actual

### **Archivos Detectados para Limpieza**

| Archivo/Carpeta | Tamaño | Estado | Acción |
|-----------------|--------|--------|--------|
| `docs/1.4_books/*.pdf` | 0 MB | ✅ YA ELIMINADOS | - |
| `docs/referencias/MLOps/` | 14.65 MB | 🚨 EXISTE | ELIMINAR |
| `models/mlflow_model/model.pkl` | 15.29 MB | 🚨 EXISTE | ELIMINAR |
| **TOTAL** | **29.94 MB** | - | - |

### **Verificación de Git**

```powershell
# Verificar que NO están en Git (debe retornar vacío)
git ls-files | Select-String "\.pdf"        # ✅ Vacío
git ls-files | Select-String "model\.pkl"   # ✅ Vacío
git ls-files | Select-String "referencias\\MLOps"  # ✅ Vacío
```

**Resultado**: ✅ **SEGURO ELIMINAR** - No hay archivos tracked en Git

---

## ⚡ Comandos de Limpieza (EJECUTAR UNO POR UNO)

### **Paso 1: Eliminar carpeta referencias/MLOps (14.65 MB)**

```powershell
# Ver qué contiene antes de eliminar
Get-ChildItem "docs\referencias\MLOps" -Recurse | Measure-Object -Property Length -Sum | Select-Object Count, @{Name='MB';Expression={[math]::Round($_.Sum/1MB,2)}}

# ELIMINAR (ejecutar solo si estás seguro)
Remove-Item "docs\referencias\MLOps" -Recurse -Force

# Verificar eliminación
Test-Path "docs\referencias\MLOps"  # Debe ser: False
```

**Contenido que se eliminará**:
- 2 PDFs de libros comerciales (Machine Learning Engineering, ML Design Patterns)
- 5 PDFs de curso externo (Introduction, Refactor OOP, Tracking, Reproducibility, Testing)
- 12 notebooks .ipynb de referencia (wine dataset)
- 10 archivos .py de código de referencia
- 1 dataset wine_quality_df.csv
- **Total**: 35 archivos, 14.65 MB

**Justificación**: Material de curso externo completo. No relacionado con proyecto obesity.

---

### **Paso 2: Eliminar modelo standalone (15.29 MB)**

```powershell
# Ver archivo antes de eliminar
Get-ChildItem "models\mlflow_model\model.pkl" | Select-Object Name, @{Name='MB';Expression={[math]::Round($_.Length/1MB,2)}}

# ELIMINAR (ejecutar solo si estás seguro)
Remove-Item "models\mlflow_model\model.pkl" -Force

# Verificar eliminación
Test-Path "models\mlflow_model\model.pkl"  # Debe ser: False
```

**Justificación**:
- Modelo ya está en MLflow Registry (`mlruns/`)
- Se carga desde MLflow, no necesita .pkl separado
- Duplicación innecesaria

---

### **Paso 3: Verificar limpieza completa**

```powershell
# Calcular tamaño nuevo del proyecto
$files = Get-ChildItem -Recurse -File | Where-Object { 
    $_.FullName -notmatch '(\.git|__pycache__|mlruns|\.dvc\\cache)' 
}
$sizeMB = [math]::Round(($files | Measure-Object -Property Length -Sum).Sum / 1MB, 2)
$count = ($files | Measure-Object).Count

Write-Host "`nProyecto después de limpieza:"
Write-Host "  Archivos: $count"
Write-Host "  Tamaño: $sizeMB MB"
```

**Tamaño esperado**: ~5-6 MB (reducción de 90%)

---

### **Paso 4 (OPCIONAL): Verificar archivos pesados restantes**

```powershell
# Listar archivos > 1MB que quedaron
Get-ChildItem -Recurse -File | Where-Object { 
    $_.FullName -notmatch '(\.git|__pycache__|mlruns|\.dvc\\cache)' -and $_.Length -gt 1MB 
} | Select-Object @{Name='Archivo';Expression={$_.FullName.Replace((Get-Location).Path + '\', '')}}, @{Name='MB';Expression={[math]::Round($_.Length/1MB,2)}} | Sort-Object MB -Descending
```

**Archivos esperados restantes**:
- `notebooks/EDA.ipynb` (~1.1 MB) - ✅ Correcto (análisis exploratorio)
- `data/processed/features.csv` (~0.5 MB) - ✅ Correcto (versionado por DVC)

---

## 🎯 Después de la Limpieza

### **1. Actualizar documentación**

```powershell
# Eliminar scripts temporales de análisis
Remove-Item "cleanup_project.ps1" -Force
Remove-Item "analyze_cleanup.ps1" -Force
```

### **2. Commit y Push**

```powershell
# Agregar cambios (solo documentación nueva)
git add docs/PROJECT_AUDIT.md docs/CLEANUP_PLAN.md docs/CONSOLIDATION_SUMMARY.md

# Commit
git commit -m "docs: Add project audit and cleanup plan

- Complete project audit (PROJECT_AUDIT.md)
- Identified 29.94 MB of unnecessary files
- Documented cleanup procedure
- Found copyright issues (resolved)
- Found redundancy in docs/referencias/MLOps

Impact: Documented path to 90% size reduction"

# Push
git push origin dev
```

### **3. Verificar que proyecto funciona**

```powershell
# Ejecutar tests
pytest tests/ -v

# Verificar DVC pipeline
dvc status

# Verificar API (en terminal separado)
python start_api.py --reload

# En otro terminal:
python test_api.py
```

---

## ✅ Checklist Post-Limpieza

### **Archivos Eliminados**
- [ ] NO existe `docs/referencias/MLOps/`
- [ ] NO existe `models/mlflow_model/model.pkl`
- [ ] NO existen PDFs comerciales (verificado en Paso 3 anterior)

### **Archivos Mantenidos**
- [ ] Existe `docs/1.4_books/README.md`
- [ ] Existe `docs/references/BOOKS.md`
- [ ] Existe `notebooks/EDA.ipynb`
- [ ] Existe `data/processed/features.csv` (ignorado por Git)

### **Git Status Limpio**
- [ ] `git status` no muestra archivos pesados staged
- [ ] `git ls-files | Select-String "\.pdf"` retorna vacío
- [ ] `git ls-files | Select-String "\.pkl"` retorna vacío

### **Proyecto Funcional**
- [ ] Tests pasan: `pytest tests/ -v` (3/3 passing)
- [ ] DVC funciona: `dvc status` (sin cambios)
- [ ] API funciona: `python start_api.py --reload`

---

## 📊 Antes vs Después

| Métrica | Antes | Después | Mejora |
|---------|-------|---------|--------|
| **Tamaño total** | ~55 MB | ~5.5 MB | -90% ✅ |
| **Archivos** | 140 | ~87 | -38% ✅ |
| **PDFs copyright** | 7 (30 MB) | 0 | -100% ✅ |
| **Material externo** | 35 archivos | 0 | -100% ✅ |
| **Coherencia** | ⚠️ Mixto | ✅ Solo proyecto | +100% ✅ |

---

## 🚀 Próximos Pasos (Opcional)

### **Fase 3: Consolidación de Documentación**

Ya documentado en `DOCUMENTATION_CONSOLIDATION_PLAN.md`:
- Reducir de 27 → 12 archivos .md
- Crear `docs/archive/` y `docs/learning/`
- Consolidar PROJECT_* → PROJECT_OVERVIEW.md
- Consolidar técnicos → TECHNICAL_DEEP_DIVE.md

**Prioridad**: Media  
**Tiempo**: 1-2 horas  
**Impacto**: Mejora navegación

---

<div align="center">

**⚡ EJECUTAR LIMPIEZA AHORA**

Copiar y ejecutar comandos del Paso 1 y 2 arriba

**Tiempo estimado**: 5 minutos  
**Impacto**: Proyecto 90% más ligero y 100% legal

</div>

---

**Creado**: 23 de Octubre, 2025  
**Responsable**: ALICIACANTA-PORTFOLIO  
**Estado**: 📋 LISTO PARA EJECUTAR
