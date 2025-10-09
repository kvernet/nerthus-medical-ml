import os
import logging
import pandas as pd

def get_data_path() -> str:
    import kagglehub
    return kagglehub.dataset_download("waltervanhuissteden/the-nerthus-dataset")

def ensure_directory(directory: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(directory, exist_ok=True)

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