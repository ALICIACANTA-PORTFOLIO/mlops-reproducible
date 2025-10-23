# 🐍 Guía de Instalación - Python 3.10 con Conda

## 📋 Paso a Paso para Configurar el Entorno

### **PASO 1: Descargar e Instalar Miniconda**

#### A. Descargar Miniconda (Python 3.10)
1. Ve a: https://docs.conda.io/en/latest/miniconda.html
2. Descarga: **Miniconda3 Windows 64-bit** (última versión)
   - Archivo: `Miniconda3-latest-Windows-x86_64.exe`

#### B. Instalar Miniconda
1. Ejecuta el instalador descargado
2. **Importante**: Durante la instalación
   - ✅ Marca: **"Add Miniconda3 to my PATH environment variable"**
   - ✅ Marca: **"Register Miniconda3 as my default Python 3.10"**
3. Completa la instalación

#### C. Reiniciar Terminal
1. **Cierra completamente** PowerShell
2. **Abre nuevo** PowerShell
3. Verifica instalación:
   ```powershell
   conda --version
   # Debería mostrar: conda 24.x.x
   ```

---

### **PASO 2: Crear Entorno del Proyecto**

Desde el directorio del proyecto (`D:\code\portfolio\mlops-reproducible`):

```powershell
# Crear entorno con conda.yaml
conda env create -f conda.yaml

# Esto instalará:
# - Python 3.10
# - Todas las dependencias necesarias
# - Tomará 5-10 minutos
```

---

### **PASO 3: Activar Entorno**

```powershell
# Activar entorno
conda activate mlops-reproducible

# Tu prompt debería cambiar a:
# (mlops-reproducible) PS D:\code\portfolio\mlops-reproducible>
```

---

### **PASO 4: Verificar Instalación**

```powershell
# Verificar Python
python --version
# Debería mostrar: Python 3.10.x

# Verificar paquetes
conda list | Select-String "pandas|numpy|mlflow|dvc"

# Debería mostrar:
# pandas        2.0.x
# numpy         1.24.x
# mlflow        2.8.1
# dvc           3.30.0
```

---

### **PASO 5: Probar el Proyecto**

```powershell
# 1. Importar paquete mlops
python -c "import mlops; print('✅ mlops OK')"

# 2. Ejecutar tests
python -m pytest tests/ -v

# 3. Verificar API
python -c "import fastapi; print('✅ FastAPI OK')"

# 4. Verificar DVC
dvc version
```

---

## 🔧 Troubleshooting

### Si conda no se reconoce después de instalar:

**Opción A: Agregar a PATH manualmente**
1. Busca: "Variables de entorno" en Windows
2. Click: "Variables de entorno..."
3. En "Variables del sistema", busca "Path"
4. Click "Editar"
5. Agregar: `C:\Users\TuUsuario\miniconda3\Scripts`
6. Agregar: `C:\Users\TuUsuario\miniconda3\condabin`
7. Reiniciar PowerShell

**Opción B: Usar Anaconda Prompt**
1. Busca "Anaconda Prompt" en el menú inicio
2. Navega al proyecto:
   ```bash
   cd D:\code\portfolio\mlops-reproducible
   ```
3. Continúa con PASO 2

---

## 📝 Comandos Útiles de Conda

```powershell
# Ver entornos
conda env list

# Activar entorno
conda activate mlops-reproducible

# Desactivar entorno
conda deactivate

# Ver paquetes instalados
conda list

# Actualizar conda
conda update conda

# Eliminar entorno (si necesitas empezar de nuevo)
conda env remove -n mlops-reproducible
```

---

## ✅ Checklist de Verificación

Una vez instalado y configurado:

```
□ conda --version funciona
□ conda activate mlops-reproducible funciona
□ python --version muestra Python 3.10.x
□ import mlops funciona
□ pytest tests/ ejecuta sin errores
□ dvc version muestra versión instalada
□ python start_api.py inicia sin errores
```

---

## 🚀 Próximos Pasos Después de Configurar

Una vez que tengas el entorno funcionando:

```powershell
# 1. Ejecutar pipeline completo
python run_mlops.py cli pipeline

# 2. Ver métricas
dvc metrics show

# 3. Iniciar MLflow UI
mlflow ui --port 5000

# 4. Iniciar API
python start_api.py

# 5. Probar API
python test_api.py
```

---

## 💡 Ventajas de Usar Conda

✅ **Aislamiento perfecto** - No afecta Python 3.12 del sistema
✅ **Reproducibilidad** - Mismo entorno en cualquier máquina
✅ **Gestión simplificada** - Un comando para todo
✅ **Estándar MLOps** - Usado en toda la industria
✅ **Compatible con proyecto** - conda.yaml ya configurado

---

## 📞 Si Necesitas Ayuda

1. **Error durante instalación de conda**: Verifica que tienes permisos de administrador
2. **Error "conda not found"**: Reinicia PowerShell y verifica PATH
3. **Error durante conda env create**: Verifica conexión a internet (descarga paquetes)
4. **Import errors después de activar**: Desactiva y reactiva el entorno

---

## 🎯 Resumen

```
1. Descarga Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Instala (marca "Add to PATH")
3. Reinicia PowerShell
4. conda env create -f conda.yaml
5. conda activate mlops-reproducible
6. ✅ ¡Listo!
```

---

**Tiempo total estimado**: 15-20 minutos (incluyendo descargas)

**¿Listo para empezar?** Descarga Miniconda y avísame cuando hayas completado PASO 1 para continuar. 🚀
