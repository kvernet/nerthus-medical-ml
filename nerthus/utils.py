import os
import logging

def ensure_directory(directory: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(directory, exist_ok=True)

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