from typing import Dict, Any

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

def get_features(feature_file):
    import numpy as np
    import pandas as pd
    df = pd.read_csv(feature_file)
    features = df.columns
    # Remove non-feature columns
    non_feature_cols = ['Unnamed: 0', 'file_path', 'bbps_class', 'file_name']
    numeric_features = [f for f in features if f not in non_feature_cols]
    return numeric_features

def extract_dataset_info(file_path):
    import re
    
    total_images = 0
    class_counts = {}

    with open(file_path, 'r') as f:
        for line in f:
            # Extract total images
            if line.strip().startswith("Total images:"):
                match = re.search(r"Total images:\s*(\d+)", line)
                if match:
                    total_images = int(match.group(1))

            # Extract class image counts
            class_match = re.search(r"- Class (\d+):\s*(\d+)\s*images", line)
            if class_match:
                class_id = int(class_match.group(1))
                image_count = int(class_match.group(2))
                class_counts[class_id] = image_count

    return total_images, class_counts
