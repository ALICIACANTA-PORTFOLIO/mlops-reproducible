"""
Feature engineering module for obesity classification.
Handles feature creation, selection, and transformation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for obesity classification."""
    
    def __init__(self, config: dict):
        """Initialize feature engineer with configuration."""
        
        self.config = config
        self.feature_selector = None
        self.pca = None
        self.engineered_features = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from existing data."""
        
        logger.info("Creating additional features...")
        
        df_features = df.copy()
        
        # BMI calculation
        if 'Height' in df.columns and 'Weight' in df.columns:
            df_features['BMI'] = df_features['Weight'] / (df_features['Height'] ** 2)
            self.engineered_features.append('BMI')
            
        # Weight-to-Height ratio
        if 'Height' in df.columns and 'Weight' in df.columns:
            df_features['Weight_Height_Ratio'] = df_features['Weight'] / df_features['Height']
            self.engineered_features.append('Weight_Height_Ratio')
            
        # Age groups
        if 'Age' in df.columns:
            df_features['Age_Group'] = pd.cut(
                df_features['Age'], 
                bins=[0, 18, 30, 45, 60, 100],
                labels=['Child', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior']
            ).astype(str)
            self.engineered_features.append('Age_Group')
            
        # Activity level indicator
        if 'FAF' in df.columns:
            df_features['High_Activity'] = (df_features['FAF'] >= 2).astype(int)
            self.engineered_features.append('High_Activity')
            
        # Eating frequency indicator  
        if 'NCP' in df.columns:
            df_features['Regular_Meals'] = (df_features['NCP'] >= 3).astype(int)
            self.engineered_features.append('Regular_Meals')
            
        # Water consumption adequacy
        if 'CH2O' in df.columns:
            df_features['Adequate_Water'] = (df_features['CH2O'] >= 2).astype(int)
            self.engineered_features.append('Adequate_Water')
            
        # Vegetable consumption adequacy
        if 'FCVC' in df.columns:
            df_features['Adequate_Vegetables'] = (df_features['FCVC'] >= 2).astype(int)
            self.engineered_features.append('Adequate_Vegetables')
            
        logger.info(f"Created {len(self.engineered_features)} additional features")
        
        return df_features
    
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       feature_names: List[str], k: int = 15) -> Tuple[np.ndarray, List[str]]:
        """Select top k features using statistical methods."""
        
        logger.info(f"Selecting top {k} features...")
        
        # Use SelectKBest with f_classif
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        # Get feature scores
        feature_scores = self.feature_selector.scores_
        
        # Log feature selection results
        logger.info("Feature selection results:")
        for i, (name, score) in enumerate(zip(feature_names, feature_scores)):
            selected = "✓" if i in selected_indices else "✗"
            logger.info(f"  {selected} {name}: {score:.4f}")
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        return X_selected, selected_features
    
    def apply_pca(self, X: np.ndarray, n_components: Optional[int] = None, 
                  variance_threshold: float = 0.95) -> np.ndarray:
        """Apply PCA for dimensionality reduction."""
        
        if n_components is None:
            # Find number of components for desired variance
            pca_temp = PCA()
            pca_temp.fit(X)
            
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        logger.info(f"Applying PCA with {n_components} components...")
        
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        explained_variance = self.pca.explained_variance_ratio_.sum()
        logger.info(f"PCA explained variance: {explained_variance:.4f}")
        
        return X_pca
    
    def get_feature_importance(self, model, feature_names: List[str]) -> List[Tuple[str, float]]:
        """Get feature importance from trained model."""
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Sort features by importance
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("Feature importance ranking:")
            for i, (name, importance) in enumerate(feature_importance[:10]):
                logger.info(f"  {i+1}. {name}: {importance:.4f}")
                
            return feature_importance
        else:
            logger.warning("Model does not support feature importance")
            return []
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables."""
        
        logger.info("Creating interaction features...")
        
        df_interactions = df.copy()
        
        # BMI * Activity interaction
        if 'BMI' in df.columns and 'FAF' in df.columns:
            df_interactions['BMI_Activity_Interaction'] = df_interactions['BMI'] * df_interactions['FAF']
            
        # Age * Weight interaction
        if 'Age' in df.columns and 'Weight' in df.columns:
            df_interactions['Age_Weight_Interaction'] = df_interactions['Age'] * df_interactions['Weight']
            
        # Water * Vegetables interaction
        if 'CH2O' in df.columns and 'FCVC' in df.columns:
            df_interactions['Water_Vegetable_Interaction'] = df_interactions['CH2O'] * df_interactions['FCVC']
            
        logger.info("Interaction features created")
        
        return df_interactions
    
    def remove_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features."""
        
        logger.info(f"Removing features with correlation > {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = df.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        logger.info(f"Dropping {len(to_drop)} correlated features: {to_drop}")
        
        return df.drop(columns=to_drop)
    
    def get_engineered_features(self) -> List[str]:
        """Get list of engineered features."""
        return self.engineered_features.copy()
    
    def save_feature_engineering(self, output_dir: str = "models/features/") -> None:
        """Save feature engineering components."""
        
        from pathlib import Path
        import joblib
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save feature selector
        if self.feature_selector:
            joblib.dump(self.feature_selector, output_path / "feature_selector.pkl")
            
        # Save PCA
        if self.pca:
            joblib.dump(self.pca, output_path / "pca.pkl")
            
        # Save engineered features list
        with open(output_path / "engineered_features.txt", 'w') as f:
            for feature in self.engineered_features:
                f.write(f"{feature}\n")
                
        logger.info(f"Feature engineering components saved to: {output_path}")
    
    def load_feature_engineering(self, input_dir: str = "models/features/") -> None:
        """Load feature engineering components."""
        
        from pathlib import Path
        import joblib
        
        input_path = Path(input_dir)
        
        # Load feature selector
        feature_selector_path = input_path / "feature_selector.pkl"
        if feature_selector_path.exists():
            self.feature_selector = joblib.load(feature_selector_path)
            
        # Load PCA
        pca_path = input_path / "pca.pkl"  
        if pca_path.exists():
            self.pca = joblib.load(pca_path)
            
        # Load engineered features list
        engineered_features_path = input_path / "engineered_features.txt"
        if engineered_features_path.exists():
            with open(engineered_features_path, 'r') as f:
                self.engineered_features = [line.strip() for line in f.readlines()]
                
        logger.info(f"Feature engineering components loaded from: {input_path}")