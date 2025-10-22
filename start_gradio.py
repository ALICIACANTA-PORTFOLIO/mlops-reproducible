#!/usr/bin/env python3
"""
Script para iniciar la interfaz Gradio de clasificación de obesidad
"""
import sys
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Iniciar interfaz Gradio de clasificación de obesidad")
    parser.add_argument("--port", type=int, default=7860, help="Puerto para Gradio")
    parser.add_argument("--host", default="127.0.0.1", help="Host para Gradio")
    parser.add_argument("--share", action="store_true", help="Crear enlace público temporal")
    parser.add_argument("--debug", action="store_true", help="Modo debug")
    
    args = parser.parse_args()
    
    # Verificar que estamos en el directorio correcto - buscar archivos existentes
    gradio_files = [
        "src/serving/gradio_app_professional.py",
        "src/serving/gradio_app.py"
    ]
    
    available_file = None
    for file_path in gradio_files:
        if Path(file_path).exists():
            available_file = file_path
            break
    
    if not available_file:
        print("❌ Error: No se encontró archivo Gradio")
        print("   Directorio actual:", Path.cwd())
        print("   Archivos buscados:", gradio_files)
        sys.exit(1)
    
    try:
        print(f"🎨 Iniciando interfaz Gradio de clasificación de obesidad...")
        print(f"📍 Host: {args.host}:{args.port}")
        print(f"🌐 URL: http://{args.host}:{args.port}")
        if args.share:
            print(f"🔗 Enlace público: Se generará automáticamente")
        
        print(f"📁 Usando archivo: {available_file}")
        
        # Importar y ejecutar según el archivo disponible
        if "professional" in available_file:
            from src.serving.gradio_app_professional import main as gradio_main
        else:
            from src.serving.gradio_app import main as gradio_main
        
        # Ejecutar la aplicación
        gradio_main()
        
    except ImportError as e:
        print(f"❌ Error: Dependencia faltante - {str(e)}")
        print("💡 Instala las dependencias con:")
        print("   pip install gradio plotly")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")
        import traceback
        print("📋 Traceback completo:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()