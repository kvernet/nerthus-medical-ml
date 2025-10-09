"""
Command line interface for Nerthus Medical ML
"""

def main():
    """Main entry point for the nerthus command."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Nerthus Medical ML Pipeline')
    
    parser.add_argument(
        "-f", "--features_path",
        type=str,
        default="outputs/image_features.csv",
        help="Features path (default: outputs/image_features.csv)"
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
        "-d", "--models_dir",
        type=str,
        default="outputs/models",
        help="Directory to save trained models (default: outputs/models)"
    )

    parser.add_argument(
        "-r", "--results_dir",
        type=str,
        default="outputs/results",
        help="Directory to save ML results and reports (default: outputs/results)"
    )

    parser.add_argument(
        "-s", "--random_state",
        type=int,
        default=42,
        help="Random sate (default: 42)"
    )
    
    args = parser.parse_args()
    
    print("Nerthus Medical ML Pipeline")
    print("=" * 40)
    
    from .ml import NerthusML
    ml = NerthusML(
        models_dir=args.models_dir,
        results_dir=args.results_dir,
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
    print(f"✅ Pipeline completed! Check outputs/ directory.")

if __name__ == "__main__":
    main()