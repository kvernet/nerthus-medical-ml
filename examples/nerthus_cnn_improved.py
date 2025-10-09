#!/usr/bin/env python3
"""
Improved CNN Pipeline with Regularization
"""

from nerthus.utils import get_data_path
from nerthus.cnn import NerthusCNN

def main():
    print("NERTHUS MEDICAL ML - IMPROVED CNN PIPELINE")
    print("=" * 45)
    
    data_path = get_data_path()
    data_path = f"{data_path}/nerthus-dataset-frames/nerthus-dataset-frames"

    # Initialize improved CNN
    cnn = NerthusCNN(input_shape=(150, 150, 3))
    
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
    history = cnn.train(train_gen, val_gen, epochs=100)
    
    # Get best validation accuracy
    best_val_accuracy = max(history.history['val_accuracy'])
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    print(f"\n🎯 TRAINING RESULTS:")
    print(f"   Best Validation Accuracy: {best_val_accuracy:.1%}")
    print(f"   Final Validation Accuracy: {final_val_accuracy:.1%}")
    print(f"   Final Training Accuracy: {history.history['accuracy'][-1]:.1%}")
    
    # Compare with traditional ML
    print(f"\n🏆 COMPARISON WITH TRADITIONAL ML:")
    print(f"   Improved CNN Best: {best_val_accuracy:.1%}")
    print(f"   Random Forest: 90.5%")
    
    if best_val_accuracy > 0.905:
        print("   🎉 CNN OUTPERFORMS TRADITIONAL ML!")
    else:
        print("   🔄 CNN shows competitive performance")
    
    print(f"\n✅ Improved CNN pipeline complete!")

if __name__ == "__main__":
    main()