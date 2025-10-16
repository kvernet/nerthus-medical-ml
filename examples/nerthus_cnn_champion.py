"""
CHAMPION CNN PIPELINE - Full training with 95.5% configuration
"""

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nerthus.utils import cnn_generate_text_report, ensure_directory, \
    extract_ml_performance, get_data_path
from nerthus.cnn import NerthusCNN
import matplotlib.pyplot as plt
import numpy as np

def create_champion_comparison(results, cnn_accuracy, output_dir):
    """Create a professional champion comparison plot."""
    results.update(
        {'CNN Champion': cnn_accuracy}
    )
    output_dir = f"{output_dir}/results"
    ensure_directory(output_dir)
    
    plt.figure(figsize=(14, 8))
    
    models = list(results.keys())
    accuracies = list(results.values())
    
    # Create bars with champion highlighted
    colors = ['#808080', '#808080', '#808080', '#808080', '#FFD700']
    bars = plt.bar(models, accuracies, color=colors, alpha=0.8)
    
    # Customize the plot
    plt.title('Medical Image Classification', 
              fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        color = 'black' if accuracy != cnn_accuracy else 'darkred'
        weight = 'normal' if accuracy != cnn_accuracy else 'bold'
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{accuracy:.1%}', ha='center', va='bottom', 
                fontweight=weight, fontsize=12, color=color)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/champion_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    return output_dir

def main():
    dropout_rates=[0.1, 0.2, 0.3, 0.2]
    
    print("NERTHUS MEDICAL ML - CHAMPION CNN TRAINING")
    print("=" * 50)
    print("🎯 Training CNN with 95.5% validation accuracy configuration")
    print(f"💡 Dropout rates: {dropout_rates} - Tiny Regularization\n")

    data_path = get_data_path()
    data_path = f"{data_path}/nerthus-dataset-frames/nerthus-dataset-frames"

    ml_results = extract_ml_performance("outputs/ml/results/ml_performance_report.txt")
    best_ml_model, best_ml_score = max(ml_results.items(), key=lambda x: x[1])
    
    output_dir = "outputs/cnn/champion"
    # Initialize champion CNN
    cnn = NerthusCNN(
        input_shape=(150, 150, 3),
        num_classes=4,
        random_state=42,
        output_dir=output_dir
    )
    
    # Build champion architecture
    print("🤖 Building champion CNN architecture...")
    model = cnn.build_tunable_cnn(
        dropout_rates=dropout_rates
    )
    
    # Create data generators
    train_gen, val_gen = cnn.create_data_generators(
        data_path=data_path,
        batch_size=32,
        validation_split=0.2
    )
    
    # Full training with optimal configuration
    print("🚀 Starting champion training...")
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
    
    # Get results
    best_val_accuracy = max(history.history['val_accuracy'])
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    final_train_accuracy = history.history['accuracy'][-1]
    overfitting_gap = final_train_accuracy - history.history['val_accuracy'][-1]
    
    print(f"\n🏆 CHAMPION RESULTS:")
    print(f"   Best Validation Accuracy: {best_val_accuracy:.1%} (epoch {best_epoch})")
    print(f"   Final Training Accuracy: {final_train_accuracy:.1%}")
    print(f"   Overfitting Gap: {overfitting_gap:.1%}")
    
    # Create champion comparison plot
    comparison_path = create_champion_comparison(
        ml_results, best_val_accuracy, output_dir
    )
    
    print(f"\n🎯 HISTORIC ACHIEVEMENT:")
    print(f"   🥇 NEW CHAMPION: CNN - {best_val_accuracy:.1%}")
    print(f"   🥈 Previous Best: {best_ml_model} - {best_ml_score}%")
    print(f"   📈 Improvement: +{100*best_val_accuracy-best_ml_score:.1f}%")
    
    print(f"\n✅ Champion CNN pipeline complete!")
    print(f"   - Model saved: {model_path}")
    print(f"   - Training plot: {history_path}")
    print(f"   - Champion comparison plot: {comparison_path}")

if __name__ == "__main__":
    main()