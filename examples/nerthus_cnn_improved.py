"""
Improved CNN Pipeline with Regularization
"""

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nerthus.utils import cnn_generate_text_report, extract_ml_performance, get_data_path
from nerthus.cnn import NerthusCNN

def main():
    print("NERTHUS MEDICAL ML - IMPROVED CNN")
    print("=" * 45)
    output_dir="outputs/cnn/improved"

    ml_results = extract_ml_performance("outputs/ml/results/ml_performance_report.txt")
    
    data_path = get_data_path()
    data_path = f"{data_path}/nerthus-dataset-frames/nerthus-dataset-frames"

    # Initialize improved CNN
    cnn = NerthusCNN(
        input_shape=(150, 150, 3),
        num_classes=4,
        random_state=42,
        output_dir=output_dir
    )
    
    # Build improved architecture
    print("🤖 Building improved CNN with regularization...")
    cnn.build_improved_cnn()
    
    # Create data generators with less augmentation
    train_gen, val_gen = cnn.create_data_generators(
        data_path=data_path,
        batch_size=32,
        validation_split=0.2
    )
    
    # Train with improved settings
    print("🚀 Training improved CNN...")
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
    model_path = cnn.save_model(model_name="nerthus_cnn.keras")
    
    # Get best validation accuracy
    best_val_accuracy = max(history.history['val_accuracy'])
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    print(f"\n🎯 TRAINING RESULTS:")
    print(f"   Best Validation Accuracy: {best_val_accuracy:.1%}")
    print(f"   Final Validation Accuracy: {final_val_accuracy:.1%}")
    print(f"   Final Training Accuracy: {history.history['accuracy'][-1]:.1%}")
    
    # Compare with traditional ML
    print(f"\n🔄 Traditional ML validation comparison:")
    for model_name, model_accuracy in ml_results.items():
        print(f"   - {model_name} CV Accuracy: {100*model_accuracy:.1f}%")
    
    print(f"\n✅ CNN improved complete!")
    print(f"   - Model saved: {model_path}")
    print(f"   - Training plot: {history_path}")

if __name__ == "__main__":
    main()