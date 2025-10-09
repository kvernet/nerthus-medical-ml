from nerthus import NerthusML

def main():
    ml = NerthusML()

    # Load features
    df = ml.load_features_data()

    # Prepare features target
    X, y = ml.prepare_features_target(df)

    # Train the models
    ml.train_models(X=X, y=y, test_size=0.2)

    # Get & save the best model
    best_name, best_model = ml.get_best_model()
    ml.save_model(best_model, best_name)

    # Genreate report
    ml.generate_report(X, y)

    # Robust validation
    ml.robust_validation(X, y)
    ml.overfitting_analysis()

if __name__ == "__main__":
    main()