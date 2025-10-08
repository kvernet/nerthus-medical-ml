import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
#from sklearn.metrics import precision_score, recall_score, f1_score

from .utils import setup_logging, ensure_directory

class NerthusML:
    """
    Machine learning pipeline for BBPS classification using extracted image features.
    """
    
    def __init__(self, models_dir: str = "outputs/models", results_dir: str = "outputs/results"):
        """
        Initialize the ML pipeline.
        
        Args:
            models_dir: Directory to save trained models
            results_dir: Directory to save ML results and reports
        """
        self.logger = setup_logging()
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.models = {}
        self.results = {}
        
        # Create directories
        ensure_directory(models_dir)
        ensure_directory(results_dir)
        
        self.logger.info("NerthusML pipeline initialized")
    
    def load_features_data(self, features_path: str = "outputs/image_features.csv") -> pd.DataFrame:
        """
        Load the extracted image features for ML training.
        
        Args:
            features_path: Path to the features CSV file
            
        Returns:
            DataFrame with features and target
        """
        self.logger.info(f"Loading features data from {features_path}")
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        df = pd.read_csv(features_path)
        
        # Basic validation
        required_cols = ['bbps_class']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        return df
      
    def prepare_features_target(self, df: pd.DataFrame, target_col: str = 'bbps_class') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for ML.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of the target column
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        self.logger.info("Preparing features and target for ML...")
        
        # Separate features and target
        X = df.drop(columns=[target_col], errors='ignore')
        y = df[target_col]
        
        # Convert target to integers (XGBoost requires this)
        try:
            y = y.astype(int)
        except ValueError:
            # If conversion fails, try mapping
            unique_classes = sorted(y.unique())
            class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
            y = y.map(class_mapping)
            self.logger.info(f"Target mapping: {class_mapping}")
        
        # Remove non-feature columns
        non_feature_cols = ['file_path', 'file_name', 'Unnamed: 0']
        X = X.drop(columns=[col for col in non_feature_cols if col in X.columns], errors='ignore')
        
        # Handle missing values
        missing_cols = X.columns[X.isnull().any()].tolist()
        if missing_cols:
            self.logger.warning(f"Columns with missing values: {missing_cols}")
            # Simple imputation with median for numeric columns
            for col in missing_cols:
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].median())
        
        self.logger.info(f"Final feature matrix: {X.shape}")
        self.logger.info(f"Target distribution:\n{y.value_counts().sort_index()}")
        
        return X, y
    
    def analyze_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform basic feature analysis before modeling.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Dictionary with feature analysis results
        """
        self.logger.info("Analyzing features for ML...")
        
        analysis = {
            'feature_names': X.columns.tolist(),
            'n_features': len(X.columns),
            'n_samples': len(X),
            'target_distribution': y.value_counts().to_dict(),
            'feature_stats': {}
        }
        
        # Basic feature statistics
        for col in X.columns:
            analysis['feature_stats'][col] = {
                'dtype': str(X[col].dtype),
                'missing': X[col].isnull().sum(),
                'mean': X[col].mean() if pd.api.types.is_numeric_dtype(X[col]) else None,
                'std': X[col].std() if pd.api.types.is_numeric_dtype(X[col]) else None
            }
        
        # Correlation with target (for numeric targets)
        if pd.api.types.is_numeric_dtype(y):
            correlations = {}
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    correlation = X[col].corr(y)
                    correlations[col] = correlation
            
            # Sort by absolute correlation
            sorted_correlations = dict(sorted(correlations.items(), 
                                            key=lambda x: abs(x[1]), 
                                            reverse=True))
            analysis['target_correlations'] = sorted_correlations
        
        return analysis
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train multiple classifiers and evaluate performance.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
        """
        self.logger.info("Training multiple classifiers...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Define models to train
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'XGBoost': xgb.XGBClassifier(random_state=random_state),
            'SVM': SVC(random_state=random_state),
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000)
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Store model and results
            results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_test': y_test,
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            self.logger.info(f"  {name} accuracy: {results[name]['accuracy']:.3f}")
        
        self.results = results
        return results
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model based on accuracy.
        
        Returns:
            Tuple of (model_name, model_object)
        """
        if not self.results:
            raise ValueError("No models trained yet. Call train_models() first.")
        
        best_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.results[best_name]['model']
        
        self.logger.info(f"Best model: {best_name} with accuracy: {self.results[best_name]['accuracy']:.3f}")
        return best_name, best_model
    
    def save_model(self, model, model_name: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model object
            model_name: Name for the model file
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        self.logger.info(f"Model saved to: {model_path}")
    
    def generate_report(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Generate comprehensive ML performance report with visualizations.
        
        Args:
            X: Features DataFrame
            y: Target Series
        """
        self.logger.info("Generating comprehensive ML report...")
        
        if not self.results:
            self.train_models(X, y)
        
        # 1. Model comparison visualization
        self._plot_model_comparison()
        
        # 2. Confusion matrix for best model
        best_name, best_model = self.get_best_model()
        best_result = self.results[best_name]
        self._plot_confusion_matrix(best_result['y_test'], best_result['y_pred'], best_name)
        
        # 3. Feature importance analysis
        self._plot_feature_importance(best_model, X.columns, best_name)
        
        # 4. Generate text report
        self._generate_text_report()
    
    def _plot_model_comparison(self):
        """Create model performance comparison plot."""
        plt.figure(figsize=(10, 6))
        models = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in models]
        
        bars = plt.bar(models, accuracies, color=['#2E8B57', '#4682B4', '#B22222', '#FF8C00'])
        plt.title('Model Performance Comparison on BBPS Classification', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Create confusion matrix visualization."""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['0', '1', '2', '3'], 
                   yticklabels=['0', '1', '2', '3'])
        plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {accuracy_score(y_true, y_pred):.3f}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted BBPS', fontsize=12)
        plt.ylabel('True BBPS', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, model, feature_names, model_name):
        """Plot feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot top 15 features
            top_n = min(15, len(feature_names))
            plt.barh(range(top_n), importances[indices[:top_n]][::-1])
            plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
            plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Feature Importance', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_text_report(self):
        """Generate detailed text report of ML results."""
        report_path = os.path.join(self.results_dir, 'ml_performance_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("NERTHUS MEDICAL ML PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("MODEL PERFORMANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            for name, result in self.results.items():
                f.write(f"{name}: {result['accuracy']:.3f} accuracy\n")
            
            best_name, _ = self.get_best_model()
            best_result = self.results[best_name]
            
            f.write(f"\nBEST MODEL: {best_name} ({best_result['accuracy']:.3f} accuracy)\n")
            f.write("-" * 30 + "\n")
            
            f.write("\nDETAILED CLASSIFICATION REPORT:\n")
            f.write("-" * 35 + "\n")
            report = best_result['classification_report']
            for class_label in ['0', '1', '2', '3']:
                if class_label in report:
                    f.write(f"BBPS Class {class_label}:\n")
                    f.write(f"  Precision: {report[class_label]['precision']:.3f}\n")
                    f.write(f"  Recall:    {report[class_label]['recall']:.3f}\n")
                    f.write(f"  F1-Score:  {report[class_label]['f1-score']:.3f}\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write(f"  Accuracy:  {report['accuracy']:.3f}\n")
            f.write(f"  Macro Avg: {report['macro avg']['f1-score']:.3f}\n")
            f.write(f"  Weighted:  {report['weighted avg']['f1-score']:.3f}\n")
        
        self.logger.info(f"ML performance report saved to: {report_path}")