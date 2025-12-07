#%% Imports and Configuration
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

#%% Main Recommendation Engine Class
class EngagementRecommender:
    """
    Machine Learning recommendation engine for customer engagement optimization.
    
    Uses ensemble methods (Random Forest, XGBoost) to predict optimal engagement 
    strategies based on historical data patterns and customer characteristics.
    """
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize recommender system.
        
        Args:
            model_type: 'xgboost', 'random_forest', or 'both'
        """
        self.model_type = model_type
        self.xgb_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.pca = None
        self.use_pca = False
        
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load and validate engagement data."""
        if filepath and os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                print(f"[OK] Loaded {len(data)} records from {filepath}")
                return data
            except Exception as e:
                print(f"[WARN] Error loading data: {e}")
                return self._generate_sample_data()
        return self._generate_sample_data()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate realistic synthetic data for demonstration purposes."""
        np.random.seed(42)
        n_samples = 2000
        
        # Create correlated features for realism
        base_score = np.random.uniform(0.3, 0.95, n_samples)
        engagement_history = np.random.poisson(3, n_samples)
        
        data = pd.DataFrame({
            'account_size': np.random.choice(['SMB', 'Mid-Market', 'Enterprise'], n_samples, p=[0.4, 0.35, 0.25]),
            'industry': np.random.choice(['Financial', 'Healthcare', 'Manufacturing', 'Technology', 'Retail'], n_samples),
            'engagement_history': engagement_history,
            'adoption_score': np.clip(base_score + (engagement_history / 20), 0, 1),
            'time_since_last': np.random.exponential(60, n_samples).astype(int),
            'service_count': np.random.poisson(8, n_samples),
            'revenue': np.random.lognormal(11, 1.2, n_samples),
            'geographic_region': np.random.choice(['US East', 'US West', 'US Central', 'International'], n_samples),
            'account_age_months': np.random.exponential(24, n_samples).astype(int),
        })
        
        # Create success probability based on features
        success_prob = (
            data['adoption_score'] * 0.4 +
            (data['engagement_history'] / 10) * 0.3 +
            (data['account_size'] == 'Enterprise').astype(int) * 0.15 +
            (data['service_count'] / 20) * 0.1 +
            np.random.normal(0, 0.1, n_samples)
        )
        success_prob = np.clip(success_prob, 0, 1)
        data['success'] = (success_prob > np.random.uniform(0, 1, n_samples)).astype(int)
        
        print(f"[OK] Generated {len(data)} synthetic records")
        print(f"  Success rate: {data['success'].mean():.1%}")
        return data
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Engineer features and prepare for modeling."""
        df = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['account_size', 'industry', 'geographic_region']
        if fit:
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str).fillna('Unknown'))
                    self.label_encoders[col] = le
        else:
            for col in categorical_cols:
                if col in df.columns and col in self.label_encoders:
                    # Handle unseen categories
                    df[col] = df[col].astype(str).fillna('Unknown')
                    known_classes = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(lambda x: x if x in known_classes else self.label_encoders[col].classes_[0])
                    df[col] = self.label_encoders[col].transform(df[col])
        
        # Feature engineering - ALWAYS create same features
        if 'time_since_last' in df.columns:
            df['recency_score'] = np.exp(-df['time_since_last'] / 180)
        else:
            df['recency_score'] = 0.5  # Default
        
        if 'engagement_history' in df.columns:
            # For prediction, use a fixed normalization factor if not fitted
            if fit:
                max_engagement = df['engagement_history'].max() + 1
                df['engagement_intensity'] = df['engagement_history'] / max_engagement
            else:
                # Use reasonable default max for prediction
                df['engagement_intensity'] = df['engagement_history'] / 10.0
        else:
            df['engagement_intensity'] = 0.0
        
        if 'service_count' in df.columns and 'revenue' in df.columns:
            df['revenue_per_service'] = df['revenue'] / (df['service_count'] + 1)
        else:
            df['revenue_per_service'] = 0.0
        
        # Define consistent feature columns (must match between fit and transform)
        base_numeric_cols = ['engagement_history', 'adoption_score', 'time_since_last', 
                            'service_count', 'revenue', 'account_age_months']
        engineered_cols = ['recency_score', 'engagement_intensity', 'revenue_per_service']
        encoded_cols = [f'{col}_encoded' if col in categorical_cols else col 
                       for col in categorical_cols if col in df.columns]
        
        # For prediction, use saved feature names if available
        if not fit and hasattr(self, 'feature_names') and len(self.feature_names) > 0:
            feature_cols = [col for col in self.feature_names if col in df.columns]
            # Add missing columns with zeros
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            feature_cols = self.feature_names
        else:
            # During fit, select all numeric features
            feature_cols = [col for col in base_numeric_cols + engineered_cols + encoded_cols 
                          if col in df.columns and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
            # Remove success column if present
            feature_cols = [col for col in feature_cols if col != 'success']
            # Store for future use
            self.feature_names = sorted(feature_cols)
        
        # Ensure all feature columns exist
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        
        # Select only the features we trained on
        X = df[self.feature_names].fillna(0).values
        y = df['success'].values if 'success' in df.columns else None
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def train_models(self, df: pd.DataFrame, use_pca: bool = False, n_components: int = 10) -> Dict:
        """Train Random Forest and/or XGBoost models."""
        print("\n" + "="*60)
        print("TRAINING ML MODELS")
        print("="*60)
        
        # Prepare features
        X, y = self.prepare_features(df, fit=True)
        
        # PCA for dimensionality reduction (optional)
        if use_pca:
            self.pca = PCA(n_components=n_components)
            X = self.pca.fit_transform(X)
            self.use_pca = True
            print(f"[OK] Applied PCA: {X.shape[1]} components, {self.pca.explained_variance_ratio_.sum():.1%} variance explained")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"[OK] Train set: {len(X_train)}, Test set: {len(X_test)}")
        
        results = {}
        
        # Train XGBoost
        if self.model_type in ['xgboost', 'both']:
            print("\n--- Training XGBoost Model ---")
            self.xgb_model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            
            self.xgb_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.xgb_model.predict(X_test)
            y_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]
            
            results['xgboost'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'feature_importance': dict(zip(self.feature_names, self.xgb_model.feature_importances_))
            }
            
            print(f"[OK] Accuracy: {results['xgboost']['accuracy']:.3f}")
            print(f"[OK] Precision: {results['xgboost']['precision']:.3f}")
            print(f"[OK] Recall: {results['xgboost']['recall']:.3f}")
            print(f"[OK] F1-Score: {results['xgboost']['f1']:.3f}")
            print(f"[OK] ROC-AUC: {results['xgboost']['roc_auc']:.3f}")
        
        # Train Random Forest
        if self.model_type in ['random_forest', 'both']:
            print("\n--- Training Random Forest Model ---")
            self.rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.rf_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.rf_model.predict(X_test)
            y_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]
            
            results['random_forest'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'feature_importance': dict(zip(self.feature_names, self.rf_model.feature_importances_))
            }
            
            print(f"[OK] Accuracy: {results['random_forest']['accuracy']:.3f}")
            print(f"[OK] Precision: {results['random_forest']['precision']:.3f}")
            print(f"[OK] Recall: {results['random_forest']['recall']:.3f}")
            print(f"[OK] F1-Score: {results['random_forest']['f1']:.3f}")
            print(f"[OK] ROC-AUC: {results['random_forest']['roc_auc']:.3f}")
        
        return results
    
    def predict(self, data: Dict, use_model: str = 'best') -> Dict:
        """Make prediction for a single engagement opportunity."""
        # Select model
        if use_model == 'best' or use_model == 'xgboost':
            if self.xgb_model is None:
                raise ValueError("XGBoost model not trained. Call train_models() first.")
            model = self.xgb_model
        elif use_model == 'random_forest':
            if self.rf_model is None:
                raise ValueError("Random Forest model not trained. Call train_models() first.")
            model = self.rf_model
        else:
            raise ValueError(f"Unknown model: {use_model}")
        
        # Ensure all required fields are present with defaults
        required_fields = ['account_size', 'industry', 'engagement_history', 'adoption_score', 
                          'time_since_last', 'service_count', 'revenue', 'geographic_region', 
                          'account_age_months']
        
        complete_data = {}
        for field in required_fields:
            complete_data[field] = data.get(field, 
                'Enterprise' if field == 'account_size' or field == 'industry' or field == 'geographic_region'
                else 0 if field in ['engagement_history', 'service_count', 'account_age_months', 'time_since_last']
                else 0.5 if field == 'adoption_score'
                else 50000 if field == 'revenue'
                else 0)
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([complete_data])
        
        # Prepare features
        X, _ = self.prepare_features(input_df, fit=False)
        
        # Apply PCA if used during training
        if self.use_pca and self.pca:
            X = self.pca.transform(X)
        
        # Predict
        probability = model.predict_proba(X)[0, 1]
        prediction = model.predict(X)[0]
        
        # Generate recommendation
        if probability > 0.7:
            strategy = "High-touch engagement recommended - High success probability"
            confidence = 'High'
        elif probability > 0.4:
            strategy = "Standard engagement approach - Moderate success probability"
            confidence = 'Medium'
        else:
            strategy = "Low-priority engagement - Monitor for changes before investing resources"
            confidence = 'Low'
        
        return {
            'success_probability': float(probability),
            'prediction': int(prediction),
            'strategy': strategy,
            'confidence': confidence,
            'model_used': use_model
        }
    
    def batch_predict(self, df: pd.DataFrame, use_model: str = 'best') -> pd.DataFrame:
        """Make predictions for multiple opportunities."""
        # Select model
        if use_model == 'best' or use_model == 'xgboost':
            model = self.xgb_model
        elif use_model == 'random_forest':
            model = self.rf_model
        else:
            raise ValueError(f"Unknown model: {use_model}")
        
        X, _ = self.prepare_features(df, fit=False)
        
        if self.use_pca and self.pca:
            X = self.pca.transform(X)
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        result_df = df.copy()
        result_df['predicted_success'] = predictions
        result_df['success_probability'] = probabilities
        
        return result_df
    
    def perform_pca_analysis(self, df: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, Dict]:
        """Perform PCA for visualization."""
        X, y = self.prepare_features(df, fit=True)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        explained_variance = {
            f'PC{i+1}': float(variance)
            for i, variance in enumerate(pca.explained_variance_ratio_)
        }
        
        return X_pca, explained_variance
    
    def save_models(self, directory: str = 'models/trained'):
        """Save trained models and preprocessors."""
        os.makedirs(directory, exist_ok=True)
        
        if self.xgb_model:
            joblib.dump(self.xgb_model, f'{directory}/xgb_model.pkl')
            print(f"[OK] Saved XGBoost model to {directory}/xgb_model.pkl")
        
        if self.rf_model:
            joblib.dump(self.rf_model, f'{directory}/rf_model.pkl')
            print(f"[OK] Saved Random Forest model to {directory}/rf_model.pkl")
        
        joblib.dump(self.scaler, f'{directory}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{directory}/label_encoders.pkl')
        
        if self.pca:
            joblib.dump(self.pca, f'{directory}/pca.pkl')
        
        print(f"[OK] Models saved to {directory}/")
    
    def load_models(self, directory: str = 'models/trained'):
        """Load trained models and preprocessors."""
        if os.path.exists(f'{directory}/xgb_model.pkl'):
            self.xgb_model = joblib.load(f'{directory}/xgb_model.pkl')
            print(f"[OK] Loaded XGBoost model")
        
        if os.path.exists(f'{directory}/rf_model.pkl'):
            self.rf_model = joblib.load(f'{directory}/rf_model.pkl')
            print(f"[OK] Loaded Random Forest model")
        
        self.scaler = joblib.load(f'{directory}/scaler.pkl')
        self.label_encoders = joblib.load(f'{directory}/label_encoders.pkl')
        
        if os.path.exists(f'{directory}/pca.pkl'):
            self.pca = joblib.load(f'{directory}/pca.pkl')
            self.use_pca = True

#%% Main Execution & Training Script
if __name__ == "__main__":
    print("="*60)
    print("ML ENGAGEMENT RECOMMENDER - TRAINING SCRIPT")
    print("="*60)
    
    # Initialize recommender with both models
    recommender = EngagementRecommender(model_type='both')
    
    # Load or generate data
    data = recommender.load_data()
    
    # Train models
    results = recommender.train_models(data, use_pca=False)
    
    # Save models
    recommender.save_models()
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.1%}")
        print(f"  Precision: {metrics['precision']:.1%}")
        print(f"  Recall:    {metrics['recall']:.1%}")
        print(f"  F1-Score:  {metrics['f1']:.1%}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
    
    # Test prediction
    print("\n" + "="*60)
    print("SAMPLE PREDICTION")
    print("="*60)
    sample_data = {
        'account_size': 'Enterprise',
        'industry': 'Financial',
        'engagement_history': 5,
        'adoption_score': 0.75,
        'time_since_last': 30,
        'service_count': 8,
        'revenue': 50000,
        'geographic_region': 'US East',
        'account_age_months': 24
    }
    
    prediction = recommender.predict(sample_data, use_model='xgboost')
    print(f"\nPrediction Result:")
    print(f"  Success Probability: {prediction['success_probability']:.1%}")
    print(f"  Prediction: {'Success' if prediction['prediction'] == 1 else 'Failure'}")
    print(f"  Strategy: {prediction['strategy']}")
    print(f"  Confidence: {prediction['confidence']}")
    
    print("\n[SUCCESS] Training complete!")
