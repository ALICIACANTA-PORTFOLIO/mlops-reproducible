#!/usr/bin/env python3
"""
Script para iniciar la interfaz Gradio de clasificación de obesidad
"""
import sys
from pathlib import Path

def main():
    # Verificar archivo simple
    if Path("src/serving/gradio_app_simple.py").exists():
        print("🎨 Iniciando Gradio - Versión Simple...")
        from src.serving.gradio_app_simple import main as gradio_main
    else:
        print("❌ Error: No se encontró gradio_app_simple.py")
        sys.exit(1)
    
    try:
        print("📍 Host: 127.0.0.1:7860")
        print("🌐 URL: http://127.0.0.1:7860")
        gradio_main()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()