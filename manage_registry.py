"""
🏷️ MODEL REGISTRY MANAGER
=========================
Script para gestionar el Model Registry de MLflow:
- Listar modelos registrados
- Ver versiones y sus métricas
- Promover modelos a Production
- Gestionar aliases (champion, challenger)
- Comparar versiones
"""

import argparse
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import pandas as pd
from tabulate import tabulate
from pathlib import Path
import yaml


def load_config(params_path="params.yaml"):
    """Cargar configuración de MLflow desde params.yaml"""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    
    mlflow_config = params.get('mlflow', {})
    
    # Configurar tracking URI
    tracking_uri = mlflow_config.get('tracking_uri', './mlruns')
    mlflow.set_tracking_uri(tracking_uri)
    
    return mlflow_config


def list_registered_models(client):
    """Listar todos los modelos registrados"""
    try:
        models = client.search_registered_models()
        
        if not models:
            print("📦 No hay modelos registrados todavía")
            return
        
        print(f"\n📊 MODELOS REGISTRADOS ({len(models)}):\n")
        
        for model in models:
            print(f"🔹 {model.name}")
            print(f"   Descripción: {model.description or 'Sin descripción'}")
            print(f"   Última actualización: {model.last_updated_timestamp}")
            print(f"   Tags: {model.tags}")
            
            # Contar versiones por stage
            versions = client.search_model_versions(f"name='{model.name}'")
            stages = {}
            for v in versions:
                stage = v.current_stage
                stages[stage] = stages.get(stage, 0) + 1
            
            print(f"   Versiones: {len(versions)} total")
            for stage, count in stages.items():
                print(f"      - {stage}: {count}")
            print()
            
    except Exception as e:
        print(f"❌ Error listando modelos: {e}")


def list_model_versions(client, model_name, detailed=False):
    """Listar versiones de un modelo específico"""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"📦 No hay versiones para el modelo '{model_name}'")
            return
        
        print(f"\n📊 VERSIONES DEL MODELO '{model_name}' ({len(versions)}):\n")
        
        # Preparar datos para la tabla
        data = []
        for v in versions:
            # Obtener métricas del run
            run = client.get_run(v.run_id)
            accuracy = run.data.metrics.get('accuracy', 'N/A')
            f1 = run.data.metrics.get('f1_macro', 'N/A')
            
            # Obtener aliases
            try:
                model_details = client.get_registered_model(model_name)
                aliases = [alias for alias, ver in model_details.aliases.items() if ver == v.version]
                alias_str = ', '.join(aliases) if aliases else '-'
            except:
                alias_str = '-'
            
            data.append([
                v.version,
                v.current_stage,
                f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy,
                f"{f1:.4f}" if isinstance(f1, float) else f1,
                alias_str,
                v.creation_timestamp,
                v.run_id[:8] + '...'
            ])
        
        headers = ['Version', 'Stage', 'Accuracy', 'F1', 'Aliases', 'Created', 'Run ID']
        print(tabulate(data, headers=headers, tablefmt='grid'))
        
        if detailed:
            print("\n📋 DETALLES POR VERSIÓN:")
            for v in versions:
                print(f"\n🔹 Version {v.version} ({v.current_stage}):")
                print(f"   Run ID: {v.run_id}")
                print(f"   Descripción: {v.description or 'Sin descripción'}")
                print(f"   Tags: {v.tags}")
                print(f"   Source: {v.source}")
                
    except Exception as e:
        print(f"❌ Error listando versiones: {e}")


def promote_model(client, model_name, version, stage):
    """Promover un modelo a un stage específico"""
    valid_stages = ['None', 'Staging', 'Production', 'Archived']
    
    if stage not in valid_stages:
        print(f"❌ Stage inválido. Debe ser uno de: {valid_stages}")
        return
    
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == 'Production')
        )
        print(f"✅ Modelo {model_name} v{version} promovido a {stage}")
        
        if stage == 'Production':
            print(f"📌 Versiones anteriores en Production archivadas automáticamente")
        
    except Exception as e:
        print(f"❌ Error promoviendo modelo: {e}")


def set_alias(client, model_name, alias, version):
    """Asignar un alias a una versión específica"""
    try:
        client.set_registered_model_alias(model_name, alias, version)
        print(f"✅ Alias '{alias}' asignado a {model_name} v{version}")
    except Exception as e:
        print(f"❌ Error asignando alias: {e}")


def delete_alias(client, model_name, alias):
    """Eliminar un alias"""
    try:
        client.delete_registered_model_alias(model_name, alias)
        print(f"✅ Alias '{alias}' eliminado de {model_name}")
    except Exception as e:
        print(f"❌ Error eliminando alias: {e}")


def compare_versions(client, model_name, version1, version2):
    """Comparar dos versiones de un modelo"""
    try:
        v1 = client.get_model_version(model_name, version1)
        v2 = client.get_model_version(model_name, version2)
        
        run1 = client.get_run(v1.run_id)
        run2 = client.get_run(v2.run_id)
        
        print(f"\n📊 COMPARACIÓN: {model_name} v{version1} vs v{version2}\n")
        
        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        data = []
        
        for metric in metrics:
            val1 = run1.data.metrics.get(metric, 'N/A')
            val2 = run2.data.metrics.get(metric, 'N/A')
            
            if isinstance(val1, float) and isinstance(val2, float):
                diff = val2 - val1
                diff_str = f"{diff:+.4f}" if diff != 0 else "="
                data.append([
                    metric,
                    f"{val1:.4f}",
                    f"{val2:.4f}",
                    diff_str
                ])
            else:
                data.append([metric, val1, val2, 'N/A'])
        
        headers = ['Métrica', f'v{version1}', f'v{version2}', 'Diferencia']
        print(tabulate(data, headers=headers, tablefmt='grid'))
        
        print(f"\nℹ️  Versión 1: Stage={v1.current_stage}, Created={v1.creation_timestamp}")
        print(f"ℹ️  Versión 2: Stage={v2.current_stage}, Created={v2.creation_timestamp}")
        
    except Exception as e:
        print(f"❌ Error comparando versiones: {e}")


def get_best_model(client, model_name, metric='accuracy'):
    """Obtener la mejor versión según una métrica"""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"📦 No hay versiones para comparar")
            return
        
        best_version = None
        best_value = -float('inf')
        
        for v in versions:
            try:
                run = client.get_run(v.run_id)
                value = run.data.metrics.get(metric, -float('inf'))
                
                if value > best_value:
                    best_value = value
                    best_version = v
            except:
                continue
        
        if best_version:
            print(f"\n🏆 MEJOR MODELO por '{metric}':")
            print(f"   Modelo: {model_name}")
            print(f"   Versión: {best_version.version}")
            print(f"   Stage: {best_version.current_stage}")
            print(f"   {metric}: {best_value:.4f}")
            print(f"   Run ID: {best_version.run_id}")
            print(f"   Created: {best_version.creation_timestamp}")
        else:
            print(f"❌ No se pudo determinar el mejor modelo")
            
    except Exception as e:
        print(f"❌ Error buscando mejor modelo: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="🏷️ Model Registry Manager - Gestionar modelos MLflow",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # List models
    subparsers.add_parser('list', help='Listar modelos registrados')
    
    # List versions
    parser_versions = subparsers.add_parser('versions', help='Listar versiones de un modelo')
    parser_versions.add_argument('model_name', help='Nombre del modelo')
    parser_versions.add_argument('--detailed', '-d', action='store_true', help='Mostrar detalles completos')
    
    # Promote
    parser_promote = subparsers.add_parser('promote', help='Promover modelo a un stage')
    parser_promote.add_argument('model_name', help='Nombre del modelo')
    parser_promote.add_argument('version', type=int, help='Versión del modelo')
    parser_promote.add_argument('stage', choices=['Staging', 'Production', 'Archived', 'None'],
                                help='Stage destino')
    
    # Set alias
    parser_alias = subparsers.add_parser('alias', help='Asignar alias a una versión')
    parser_alias.add_argument('model_name', help='Nombre del modelo')
    parser_alias.add_argument('alias', help='Nombre del alias (ej: champion, challenger)')
    parser_alias.add_argument('version', type=int, help='Versión del modelo')
    
    # Delete alias
    parser_delalias = subparsers.add_parser('delalias', help='Eliminar un alias')
    parser_delalias.add_argument('model_name', help='Nombre del modelo')
    parser_delalias.add_argument('alias', help='Nombre del alias a eliminar')
    
    # Compare
    parser_compare = subparsers.add_parser('compare', help='Comparar dos versiones')
    parser_compare.add_argument('model_name', help='Nombre del modelo')
    parser_compare.add_argument('version1', type=int, help='Primera versión')
    parser_compare.add_argument('version2', type=int, help='Segunda versión')
    
    # Best
    parser_best = subparsers.add_parser('best', help='Encontrar mejor modelo')
    parser_best.add_argument('model_name', help='Nombre del modelo')
    parser_best.add_argument('--metric', default='accuracy', help='Métrica para comparar')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config()
    print(f"🔗 MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    client = MlflowClient()
    
    if args.command == 'list':
        list_registered_models(client)
    
    elif args.command == 'versions':
        list_model_versions(client, args.model_name, args.detailed)
    
    elif args.command == 'promote':
        promote_model(client, args.model_name, args.version, args.stage)
    
    elif args.command == 'alias':
        set_alias(client, args.model_name, args.alias, args.version)
    
    elif args.command == 'delalias':
        delete_alias(client, args.model_name, args.alias)
    
    elif args.command == 'compare':
        compare_versions(client, args.model_name, args.version1, args.version2)
    
    elif args.command == 'best':
        get_best_model(client, args.model_name, args.metric)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
