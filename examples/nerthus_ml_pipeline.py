"""
Nerthus ML Pipeline:
    Loading features data
    Preparing features target
    Training the models
    Robust validation
    Generating report
    Getting and saving the best model
"""

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nerthus import NerthusML

def main():
    print("Nerthus Medical ML Pipeline")
    print("=" * 40)
    
    output_dir = "outputs/ml/"
    ml = NerthusML(
        output_dir=output_dir,
        random_state=42
    )

    analysis = ml.run_pipeline(
        features_path="outputs/analysis/image_features.csv",
        target_col='bbps_class',
        test_size=0.2,
        cv_folds=5
    )
    
    best_name = ml.get_best_model()[0]
    best_cv = ml.cv_results[best_name]['cv_mean']
    
    print(f"\n🎯 Best Model: {best_name} ({best_cv:.1%} cross-validated accuracy)")
    print(f"🔍 Overfitting Risk: {analysis['overfitting_risk']}")
    print(f"✅ Pipeline completed! Check {output_dir} directory for results.")

if __name__ == "__main__":
    main()