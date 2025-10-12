"""
Simple CNN for Nerthus Medical ML
Trains CNN and compares with traditional ML results
"""

from nerthus.utils import cnn_generate_text_report, extract_ml_performance, get_data_path
from nerthus.cnn import NerthusCNN

def main():
    print("NERTHUS MEDICAL ML - SIMPLE CNN")
    print("=" * 40)
    output_dir="outputs/cnn/simple"

    ml_results = extract_ml_performance("outputs/ml/results/ml_performance_report.txt")

    data_path = get_data_path()
    data_path = f"{data_path}/nerthus-dataset-frames/nerthus-dataset-frames"
    
    # Initialize CNN
    cnn = NerthusCNN(
        input_shape=(150, 150, 3),  # Slightly larger for better performance
        num_classes=4,
        random_state=42,
        output_dir=output_dir
    )
    
    # Build model
    print("🤖 Building simple CNN architecture...")
    cnn.build_simple_cnn()
    
    # Create data generators
    print("📊 Loading data...")
    train_gen, val_gen = cnn.create_data_generators(
        data_path=data_path,
        batch_size=32,
        validation_split=0.2
    )
    
    # Train model & load the best model (not the final overfitted one)
    print("🚀 Training CNN...")
    history = cnn.train(
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=200
    )

    # Evaluate model
    print("📈 Evaluating model...")
    _, accuracy = cnn.evaluate(
        test_generator=val_gen
    )

    # Generate text report
    cnn_generate_text_report(
        accuracy=accuracy,
        history=history,
        train_gen=train_gen,
        val_gen=val_gen,
        input_shape=cnn.input_shape,
        report_path=f"{output_dir}/cnn_performance_report.txt"
    )
    
    # Plot training history
    history_path = cnn.plot_training_history(
        file_name="nerthus_cnn_training_history.png"
    )
    
    # Save & load the best model
    model_path = cnn.save_model(
        model_name="nerthus_cnn.keras"
    )
    
    # Results comparison
    print("\n" + "=" * 50)
    print("🏆 PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"🤖 CNN Performance:")
    print(f"   - Validation Accuracy: {accuracy:.1%}")
    print(f"   - Final Training Accuracy: {history.history['accuracy'][-1]:.1%}")
    print(f"   - Best Validation Accuracy: {max(history.history['val_accuracy']):.1%}")
    
    print(f"\n🔄 Traditional ML:")
    for model_name, model_accuracy in ml_results.items():
        print(f"   - {model_name} CV Accuracy: {100*model_accuracy:.1f}%")
    
    print(f"\n📊 Dataset Info:")
    print(f"   - Training samples: {train_gen.samples}")
    print(f"   - Validation samples: {val_gen.samples}")
    print(f"   - Input shape: {cnn.input_shape}")
    
    print(f"\n✅ CNN pipeline complete!")
    print(f"   - Model saved: {model_path}")
    print(f"   - Training plot: {history_path}")

if __name__ == "__main__":
    main()