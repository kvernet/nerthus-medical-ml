import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import json

from .utils import setup_logging, ensure_directory

class NerthusML:
    """
    Machine learning pipeline for BBPS classification using extracted image features.
    """
    
    def __init__(self,
                 output_dir: str = "outputs/ml/",
                 random_state: int = 32):
        """
        Initialize the ML pipeline.
        
        Args:
            output_dir: Output directory
            random_state: Random sate
        """
        self.logger = setup_logging()
        self.models_dir = f"{output_dir}/models"
        self.results_dir = f"{output_dir}/results"
        self.models = {}
        self.results = {}
        self.random_state = random_state
        
        # Create directories
        ensure_directory(self.models_dir)
        ensure_directory(self.results_dir)
        
        self.logger.info("NerthusML pipeline ML initialized")
    
    def load_features_data(self,
                           features_path: str = "outputs/analysis/image_features.csv"
    ) -> pd.DataFrame:
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
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
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
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        self.logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Define models to train
        models = self.get_models()
        
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
    
    def get_models(self, n_estimators: int = 100, max_iter: int = 1000) -> Dict[str, Any]:
        return {
            'Random Forest': RandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state),
            'SVM': SVC(random_state=self.random_state),
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=max_iter)
        }
    
    def robust_validation(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform robust cross-validation to detect overfitting.
        
        Args:
            X: Features DataFrame
            y: Target Series
            cv_folds: Number of cross-validation folds
        """
        self.logger.info("Performing robust cross-validation...")
        
        models = self.get_models()
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_results = {}
        for name, model in models.items():
            self.logger.info(f"Cross-validating {name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            
            cv_results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores,
                'train_test_gap': None  # Will be calculated after train/test split
            }
            
            self.logger.info(f"  {name} CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Compare with train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            gap = train_score - test_score
            
            cv_results[name]['train_test_gap'] = gap
            cv_results[name]['train_score'] = train_score
            cv_results[name]['test_score'] = test_score
            
            self.logger.info(f"  {name} Train/Test: {train_score:.3f}/{test_score:.3f} (gap: {gap:.3f})")
        
        self.cv_results = cv_results
        return cv_results
    
    def plot_validation_results(self):
        """Plot cross-validation and train/test comparison."""
        if not hasattr(self, 'cv_results'):
            raise ValueError("Run robust_validation() first")
        
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Cross-validation results
        models = list(self.cv_results.keys())
        cv_means = [self.cv_results[name]['cv_mean'] for name in models]
        cv_stds = [self.cv_results[name]['cv_std'] for name in models]
        
        bars1 = ax1.bar(models, cv_means, yerr=cv_stds, capsize=5, 
                       color=['#2E8B57', '#4682B4', '#B22222'])
        ax1.set_title('Cross-Validation Performance\n(5-fold Stratified)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, mean, std in zip(bars1, cv_means, cv_stds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Train/Test gap
        train_scores = [self.cv_results[name]['train_score'] for name in models]
        test_scores = [self.cv_results[name]['test_score'] for name in models]
        gaps = [self.cv_results[name]['train_test_gap'] for name in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars2 = ax2.bar(x - width/2, train_scores, width, label='Train', alpha=0.7)
        bars3 = ax2.bar(x + width/2, test_scores, width, label='Test', alpha=0.7)
        
        # Add gap annotations
        for i, gap in enumerate(gaps):
            ax2.text(i, max(train_scores[i], test_scores[i]) + 0.02, 
                    f'gap: {gap:.3f}', ha='center', va='bottom', fontsize=10, color='red')
        
        ax2.set_title('Train vs Test Performance\n(Overfitting Detection)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        ax2.set_ylim(0, 1.0)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'robust_validation.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def overfitting_analysis(self) -> Dict[str, Any]:
        """
        Analyze potential overfitting and provide recommendations.
        """
        if not hasattr(self, 'cv_results'):
            raise ValueError("Run robust_validation() first")
        
        analysis = {
            'overfitting_risk': 'Low',
            'recommendations': [],
            'cv_performance': {},
            'train_test_gaps': {}
        }
        
        for name, result in self.cv_results.items():
            analysis['cv_performance'][name] = {
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'train_test_gap': result['train_test_gap']
            }
            
            analysis['train_test_gaps'][name] = result['train_test_gap']
            
            # Overfitting detection logic
            if result['train_test_gap'] > 0.1:
                analysis['overfitting_risk'] = 'High'
                analysis['recommendations'].append(
                    f"{name}: Large train-test gap ({result['train_test_gap']:.3f}) suggests overfitting"
                )
            elif result['train_test_gap'] > 0.05:
                analysis['overfitting_risk'] = 'Medium'
                analysis['recommendations'].append(
                    f"{name}: Moderate train-test gap ({result['train_test_gap']:.3f})"
                )
        
        if not analysis['recommendations']:
            analysis['recommendations'].append("No significant overfitting detected")
        
        file_path = f"{self.results_dir}/overfitting_analysis.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=4)
        
        self.logger.info(f"ML overfitting analysis report saved to: {file_path}")
        
        return analysis
    
    def run_pipeline(self,
                     features_path: str = "outputs/analysis/image_features.csv",
                     target_col: str = 'bbps_class',
                     test_size: float = 0.2,
                     cv_folds: int = 5) -> Dict[str, Any]:
        """
        Run ML pipeline using EXISTING features.
        
        Args:            
        """
        self.logger.info("Starting ML pipeline...")
        
        # Load existing features
        features_df = self.load_features_data(features_path=features_path)
        
        # Prepare and train
        X, y = self.prepare_features_target(df=features_df, target_col=target_col)
        self.train_models(X=X, y=y, test_size=test_size)
        self.robust_validation(X=X, y=y, cv_folds=cv_folds)
        self.generate_report(X, y)

        # Save the best model
        best_name, best_model = self.get_best_model()
        self.save_model(best_model, best_name)
        
        return self.overfitting_analysis()

def main():
    """Main entry point for the nerthus-ml command."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Nerthus Medical ML Pipeline')
    
    parser.add_argument(
        "-f", "--features_path",
        type=str,
        default="outputs/analysis/image_features.csv",
        help="Features path (default: outputs/analysis/image_features.csv)"
    )

    parser.add_argument(
        "-t", "--target_col",
        type=str,
        default='bbps_class',
        help="Target column (default: bbps_class)"
    )

    parser.add_argument(
        "-e", "--test_size",
        type=float,
        default=0.2,
        help="Test dataset size (default: 0.2)"
    )

    parser.add_argument(
        "-l", "--cv_folds",
        type=int,
        default=5,
        help="cv folds (default: 5)"
    )
    
    parser.add_argument(
        "-s", "--random_state",
        type=int,
        default=42,
        help="Random sate (default: 42)"
    )

    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="outputs/ml/",
        help="Output directory (default: outputs/ml/)"
    )
    
    args = parser.parse_args()
    
    print("Nerthus Medical ML Pipeline")
    print("=" * 40)
    
    ml = NerthusML(
        output_dir=args.output_dir,
        random_state=args.random_state
    )

    analysis = ml.run_pipeline(
        features_path=args.features_path,
        target_col=args.target_col,
        test_size=args.test_size,
        cv_folds=args.cv_folds
    )
    
    best_name = ml.get_best_model()[0]
    best_cv = ml.cv_results[best_name]['cv_mean']
    
    print(f"\n🎯 Best Model: {best_name} ({best_cv:.1%} cross-validated accuracy)")
    print(f"🔍 Overfitting Risk: {analysis['overfitting_risk']}")
    print(f"✅ Pipeline completed! Check {args.output_dir} directory for results.")

if __name__ == "__main__":
    main()