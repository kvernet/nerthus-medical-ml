"""
Plot comparison
"""

from nerthus.utils import ensure_directory, extract_cnn_performance, extract_ml_performance
import matplotlib.pyplot as plt
import numpy as np

def main():
    output_dir = "outputs/cnn"
    ensure_directory(output_dir)

    ml_results = extract_ml_performance("outputs/ml/results/ml_performance_report.txt")
    cnn_results = extract_cnn_performance(
        "outputs/cnn/cnn_performance_report.txt")

    # Create plot with smaller size and DPI
    plt.figure(figsize=(8, 6))
    models = list(ml_results.keys())
    accuracies = [round(100*accur,1) for accur in ml_results.values()]

    cnn_accuracies = list(cnn_results.values())

    models.append("CNN")
    accuracies.append(cnn_accuracies[1])

    colors = ['#2E8B57', '#4682B4', '#B22222', '#FF8C00', '#6A5ACD']
    bars = plt.bar(models, accuracies, color=colors[:len(models)])

    plt.title('Medical Image Classification:\nTraditional ML vs Deep Learning', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    if len(models) > 1:
        plt.axvline(x=len(models)-1.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

    plt.tight_layout()

    # Save with optimization
    save_path = f"{output_dir}/ml_vs_cnn_comparison.png"
    plt.savefig(save_path,
                dpi=150,
                bbox_inches='tight',
                pil_kwargs={"compress_level": 9})

    plt.close()
    print(f"ML vs CNN comparison saved: {save_path}")

if __name__ == "__main__":
    main()