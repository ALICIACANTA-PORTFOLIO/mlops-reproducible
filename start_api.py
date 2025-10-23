#!/usr/bin/env python3
"""
Script para iniciar la API de clasificación de obesidad
"""
import uvicorn
import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Iniciar API de clasificación de obesidad")
    parser.add_argument("--host", default="127.0.0.1", help="Host para la API")
    parser.add_argument("--port", type=int, default=8000, help="Puerto para la API")
    parser.add_argument("--reload", action="store_true", help="Modo desarrollo con auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Número de workers")
    
    args = parser.parse_args()
    
    # Verificar que estamos en el directorio correcto
    if not Path("src/serving/api.py").exists():
        print("❌ Error: Ejecuta este script desde el directorio raíz del proyecto")
        print("   Directorio actual:", Path.cwd())
        print("   Se esperaba encontrar: src/serving/api.py")
        sys.exit(1)
    
    print(f"🚀 Iniciando API de clasificación de obesidad...")
    print(f"📍 Host: {args.host}:{args.port}")
    print(f"📚 Documentación: http://{args.host}:{args.port}/docs")
    print(f"🔄 Reload: {'Activado' if args.reload else 'Desactivado'}")
    
    # Configuración de uvicorn
    config = {
        "app": "src.serving.api:app",
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
    }
    
    if not args.reload:
        config["workers"] = args.workers
    
    uvicorn.run(**config)

if __name__ == "__main__":
    main()