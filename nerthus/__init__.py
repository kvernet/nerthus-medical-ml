"""
Nerthus Medical ML - Automated bowel preparation quality assessment
"""

__version__ = "0.1.0"
__author__ = "Kinson VERNET"
__email__ = "kinson.vernet@gmail.com"

from .processor import ImageProcessor
from .utils import ensure_directory, setup_logging

__all__ = ["ImageProcessor", "ensure_directory", "setup_logging"]