"""
Systematic CNN Tuning to Find Optimal Regularization
"""

from nerthus.utils import cnn_generate_text_report, extract_ml_performance, get_data_path
from nerthus.cnn import NerthusCNN

def tune_cnn_dropout():
    """Test different dropout rates to find optimal regularization"""
    output_dir = "outputs/cnn/tunning"

    data_path = get_data_path()
    data_path = f"{data_path}/nerthus-dataset-frames/nerthus-dataset-frames"
    
    dropout_configs = [
        {'name': 'Tiny Regularization', 'dropouts': [0.1, 0.2, 0.3, 0.2]},
        {'name': 'Light Regularization', 'dropouts': [0.2, 0.3, 0.4, 0.3]},
        {'name': 'Medium Regularization', 'dropouts': [0.3, 0.4, 0.4, 0.5]},
        {'name': 'Heavy Regularization', 'dropouts': [0.4, 0.5, 0.5, 0.6]},
        {'name': 'Huge Regularization', 'dropouts': [0.5, 0.6, 0.6, 0.7]},
    ]
    
    results = []
    
    for config in dropout_configs:
        print(f"\n🔧 Testing {config['name']}...")
        
        cnn = NerthusCNN(
            input_shape=(150, 150, 3),
            num_classes=4,
            random_state=42,
            output_dir=output_dir
        )
        
        # Build model with specific dropout rates
        model = cnn.build_tunable_cnn(
            dropout_rates=config['dropouts']
        )
        
        train_gen, val_gen = cnn.create_data_generators(
            data_path=data_path,
            batch_size=32,
            validation_split=0.2
        )
        
        # Quick training (50 epochs) to compare
        history = cnn.train(
            train_generator=train_gen,
            val_generator=val_gen,
            epochs=200
        )
        
        best_val_acc = max(history.history['val_accuracy'])
        final_train_acc = history.history['accuracy'][-1]
        overfitting_gap = final_train_acc - best_val_acc

        # Generate text report
        cnn_name = str.replace(config['name'], ' ', '')
        cnn_generate_text_report(
            accuracy=best_val_acc,
            history=history,
            train_gen=train_gen,
            val_gen=val_gen,
            input_shape=cnn.input_shape,
            report_path=f"{output_dir}/cnn_{cnn_name}_performance_report.txt"
        )

        # Plot training history
        history_path = cnn.plot_training_history(
            file_name=f"nerthus_cnn_{cnn_name}_training_history.png"
        )
        
        # Save & load the best model
        model_path = cnn.save_model(
            model_name=f"nerthus_cnn_{cnn_name}.keras"
        )
        
        results.append({
            'config': config['name'],
            'best_val_accuracy': best_val_acc,
            'overfitting_gap': overfitting_gap,
            'dropouts': config['dropouts'],
            'model': model
        })
        
        print(f"   ✅ Best Val Accuracy: {best_val_acc:.1%}")
        print(f"   📊 Overfitting Gap: {overfitting_gap:.1%}")

        print(f"\n✅ CNN {cnn_name} complete!")
        print(f"   - Model saved: {model_path}")
        print(f"   - Training plot: {history_path}")
    
    return results

def main():
    print("NERTHUS MEDICAL ML - SYSTEMATIC CNN TUNING")
    print("=" * 50)

    ml_results = extract_ml_performance("outputs/ml/results/ml_performance_report.txt")
    _, best_ml_score = max(ml_results.items(), key=lambda x: x[1])
    
    print(f"🎯 Goal: Find optimal regularization to reach {100*best_ml_score}%+ accuracy")
    print("📊 Strategy: Test different dropout configurations\n")
    
    results = tune_cnn_dropout()
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['best_val_accuracy'])
    
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   {best_result['config']}")
    print(f"   Best Validation Accuracy: {best_result['best_val_accuracy']:.1%}")
    print(f"   Dropout Rates: {best_result['dropouts']}")
    print(f"   Overfitting Gap: {best_result['overfitting_gap']:.1%}")

    # Compare with traditional ML
    print(f"\n🔄 Traditional ML validation comparison:")
    print(f"   - Tunning CNN Best: {best_result['best_val_accuracy']:.1%}")
    for model_name, model_accuracy in ml_results.items():
        print(f"   - {model_name} CV Accuracy: {100*model_accuracy:.1f}%")

if __name__ == "__main__":
    main()