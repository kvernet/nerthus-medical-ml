import os
import logging
import pandas as pd
from typing import Dict, Any

def cnn_generate_text_report(
        accuracy, history, train_gen,
        val_gen, input_shape, report_path):
    with open(report_path, 'w') as f:
        f.write("NERTHUS MEDICAL CNN PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("MODEL PERFORMANCE SUMMARY:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Validation Accuracy: {accuracy:.1%}\n")
        f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.1%}\n")
        f.write(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.1%}\n")
        
        f.write(f"Training samples: {train_gen.samples}\n")
        f.write(f"Validation samples: {val_gen.samples}\n")
        f.write(f"Input shape: {input_shape}\n")

def get_data_path() -> str:
    import kagglehub
    return kagglehub.dataset_download("waltervanhuissteden/the-nerthus-dataset")

def ensure_directory(directory: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(directory, exist_ok=True)

def extract_cnn_performance(file_path: str) -> Dict[str, Any]:
    result = {}
    with open(file_path, 'r') as file:
        for line in file:
            if "Final Training Accuracy" in line:
                value = float(line.strip().split(":")[1].replace('%', '').strip())
                result.update({"Final Training Accuracy": value})
            elif "Best Validation Accuracy" in line:
                value = float(line.strip().split(":")[1].replace('%', '').strip())
                result.update({"Best Validation Accuracy": value})
    return result

def extract_ml_performance(file_path: str) -> Dict[str, Any]:
    results = {}
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Ensure there are enough lines
        if len(lines) < 9:
            raise ValueError("File does not contain enough lines to extract model performance.")

        # Extract only the 4 lines with model performance (lines 6 to 9, 0-based index 5 to 8)
        performance_lines = lines[5:9]

        for idx, line in enumerate(performance_lines, start=6):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                model_part, rest = line.split(":", 1)
                model_name = model_part.strip()
                score_str, metric = rest.strip().split(maxsplit=1)
                score = float(score_str)
                results.update({model_name: score})
            except ValueError as e:
                print(f"Warning: Failed to parse line {idx}: '{line}'. Error: {e}")
                continue

    except Exception as e:
        print(f"File {file_path} doesn't exist.")
    
    return results

def save_dataframe(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """Save DataFrame with proper directory creation."""
    ensure_directory(os.path.dirname(filepath))
    df.to_csv(filepath, **kwargs)

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("nerthus")
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger