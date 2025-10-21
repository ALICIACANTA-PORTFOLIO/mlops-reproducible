import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar src al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent / "src"))

class TestDataValidation:
    """Tests básicos de validación de datos"""
    
    def test_raw_data_exists(self):
        """Verifica que el archivo de datos raw existe"""
        raw_path = Path("data/raw/ObesityDataSet_raw_and_data_sinthetic.csv")
        assert raw_path.exists(), f"Archivo de datos raw no encontrado: {raw_path}"
    
    def test_raw_data_structure(self):
        """Verifica la estructura básica del dataset raw"""
        raw_path = Path("data/raw/ObesityDataSet_raw_and_data_sinthetic.csv")
        if raw_path.exists():
            df = pd.read_csv(raw_path)
            
            # Verificar que no esté vacío
            assert len(df) > 0, "El dataset está vacío"
            
            # Verificar columnas esperadas
            expected_cols = [
                "Age", "Height", "Weight", "Gender", "family_history_with_overweight",
                "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", 
                "CALC", "MTRANS", "NObeyesdad"
            ]
            
            for col in expected_cols:
                assert col in df.columns, f"Columna {col} no encontrada en el dataset"

    def test_params_file_exists(self):
        """Verifica que params.yaml existe y es válido"""
        import yaml
        params_path = Path("params.yaml")
        assert params_path.exists(), "params.yaml no encontrado"
        
        with open(params_path, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        
        # Verificar secciones críticas
        assert "data" in params, "Sección 'data' no encontrada en params.yaml"
        assert "model" in params, "Sección 'model' no encontrada en params.yaml"
        assert "split" in params, "Sección 'split' no encontrada en params.yaml"