"""
Data processing module for obesity classification.
Handles data loading, validation, and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Professional data processor for obesity classification dataset."""
    
    def __init__(self, config: dict):
        """Initialize data processor with configuration."""
        
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Define categorical columns
        self.categorical_columns = [
            'Gender', 'family_history_with_overweight', 'FAVC', 
            'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'
        ]
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load obesity dataset from CSV file."""
        
        if data_path is None:
            data_path = self.config.get('raw_path', 'data/raw/ObesityDataSet_raw_and_data_sinthetic.csv')
        
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info(f"Loading data from: {data_path}")
        
        df = pd.read_csv(data_path)
        
        logger.info(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate dataset structure and quality."""
        
        # Check for required columns
        expected_columns = {
            'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
            'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF',
            'TUE', 'CALC', 'MTRANS', 'NObeyesdad'
        }
        
        missing_columns = expected_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Dataset contains {missing_values} missing values")
        
        # Validate target column
        if 'NObeyesdad' not in df.columns:
            raise ValueError("Target column 'NObeyesdad' not found")
        
        # Check data types and ranges
        self._validate_ranges(df)
        
        logger.info("Data validation completed successfully")
    
    def _validate_ranges(self, df: pd.DataFrame) -> None:
        """Validate data ranges for numerical columns."""
        
        validations = {
            'Age': (0, 120),
            'Height': (0.5, 2.5),
            'Weight': (10, 500),
            'FCVC': (0, 5),
            'NCP': (0, 10),
            'CH2O': (0, 5),
            'FAF': (0, 5),
            'TUE': (0, 5)
        }
        
        for column, (min_val, max_val) in validations.items():
            if column in df.columns:
                out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
                if len(out_of_range) > 0:
                    logger.warning(f"Column '{column}': {len(out_of_range)} values out of range [{min_val}, {max_val}]")
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """Preprocess dataset for machine learning."""
        
        logger.info("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Encode categorical variables
        for col in self.categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        # Separate features and target
        target_column = 'NObeyesdad'
        feature_columns = [col for col in df_processed.columns if col != target_column]
        
        X = df_processed[feature_columns].values
        y = df_processed[target_column].values
        
        # Encode target if it's categorical
        if not np.issubdtype(y.dtype, np.number):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
            self.label_encoders['target'] = target_encoder
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Preprocessing completed: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        
        return X_scaled, y, feature_columns
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: Optional[float] = None, 
                   random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        
        test_size = test_size or self.config.get('test_size', 0.2)
        random_state = random_state or self.config.get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training set: {X_train.shape[0]} samples")
        logger.info(f"  Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> list:
        """Get list of feature names after preprocessing."""
        
        return [col for col in self.categorical_columns + 
                ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
                if col != 'NObeyesdad']
    
    def save_preprocessed_data(self, X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray,
                              feature_names: list, output_dir: str = "data/processed/") -> None:
        """Save preprocessed data to files."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save data arrays
        np.save(output_path / "X_train.npy", X_train)
        np.save(output_path / "X_test.npy", X_test) 
        np.save(output_path / "y_train.npy", y_train)
        np.save(output_path / "y_test.npy", y_test)
        
        # Save feature names
        with open(output_path / "feature_names.txt", 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        # Save preprocessing objects
        import joblib
        joblib.dump(self.scaler, output_path / "scaler.pkl")
        joblib.dump(self.label_encoders, output_path / "label_encoders.pkl")
        
        logger.info(f"Preprocessed data saved to: {output_path}")
    
    def load_preprocessed_data(self, data_dir: str = "data/processed/") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Load preprocessed data from files."""
        
        data_path = Path(data_dir)
        
        X_train = np.load(data_path / "X_train.npy")
        X_test = np.load(data_path / "X_test.npy")
        y_train = np.load(data_path / "y_train.npy") 
        y_test = np.load(data_path / "y_test.npy")
        
        # Load feature names
        with open(data_path / "feature_names.txt", 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Load preprocessing objects
        import joblib
        self.scaler = joblib.load(data_path / "scaler.pkl")
        self.label_encoders = joblib.load(data_path / "label_encoders.pkl")
        
        logger.info(f"Preprocessed data loaded from: {data_path}")
        
        return X_train, X_test, y_train, y_test, feature_names