#!/usr/bin/env python3
"""
Complete CNN Pipeline for Nerthus Medical ML
Trains CNN and compares with traditional ML results
"""

from nerthus.utils import ensure_directory
from nerthus.utils import get_data_path
from nerthus.cnn import NerthusCNN
import matplotlib.pyplot as plt

output_dir = "outputs/images"

def main():
    ensure_directory(output_dir)
    
    print("NERTHUS MEDICAL ML - CNN PIPELINE")
    print("=" * 40)

    data_path = get_data_path()
    data_path = f"{data_path}/nerthus-dataset-frames/nerthus-dataset-frames"
    
    # Initialize CNN
    cnn = NerthusCNN(input_shape=(150, 150, 3))  # Slightly larger for better performance
    
    # Build model
    print("🤖 Building CNN architecture...")
    cnn.build_simple_cnn()
    
    # Create data generators
    print("📊 Loading data...")
    train_gen, val_gen = cnn.create_data_generators(
        data_path=data_path,
        batch_size=32,
        validation_split=0.2
    )
    
    # Train model
    print("🚀 Training CNN...")
    history = cnn.train(train_gen, val_gen, epochs=200)
    
    # Evaluate
    print("📈 Evaluating model...")
    loss, accuracy = cnn.evaluate(val_gen)
    
    # Plot training history
    cnn.plot_training_history(f"{output_dir}/cnn_training_history.png")
    
    # Save model
    model_path = cnn.save_model("nerthus_cnn_trained")
    
    # Results comparison
    print("\n" + "=" * 50)
    print("🏆 PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"🤖 CNN Performance:")
    print(f"   - Validation Accuracy: {accuracy:.1%}")
    print(f"   - Final Training Accuracy: {history.history['accuracy'][-1]:.1%}")
    print(f"   - Best Validation Accuracy: {max(history.history['val_accuracy']):.1%}")
    
    print(f"\n🔄 Traditional ML (from previous analysis):")
    print(f"   - Random Forest CV Accuracy: 90.5%")
    print(f"   - XGBoost CV Accuracy: 89.7%")
    
    print(f"\n📊 Dataset Info:")
    print(f"   - Training samples: {train_gen.samples}")
    print(f"   - Validation samples: {val_gen.samples}")
    print(f"   - Input shape: {cnn.input_shape}")
    
    print(f"\n💡 Insights:")
    if accuracy > 0.70:
        print("   ✅ CNN shows strong potential with more training!")
    else:
        print("   🔄 CNN may need architecture tuning or more data.")
    
    print(f"\n✅ CNN pipeline complete!")
    print(f"   - Model saved: {model_path}")
    print(f"   - Training plot: {output_dir}/cnn_training_history.png")

if __name__ == "__main__":
    main()