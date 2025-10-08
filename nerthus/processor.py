import os
import cv2
import numpy as np
from scipy import stats
from typing import List, Dict, Optional
from skimage import feature, measure
import logging

class ImageProcessor:
    """
    Processor for medical image data from the Nerthus dataset.
    Handles image loading, feature extraction, and medical image analysis.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.logger = logging.getLogger("nerthus")
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def discover_image_files(self) -> Dict[str, List[str]]:
        """
        Discover image files organized by BBPS quality classes (0-3).
        
        Returns:
            Dictionary with BBPS class labels as keys and list of image paths as values
        """
        self.logger.info("Discovering image files...")
        
        image_files = {}
        
        # Look for directories that correspond to BBPS classes (0-3)
        for item in os.listdir(self.data_path):
            item_path = os.path.join(self.data_path, item)
            
            if os.path.isdir(item_path) and item in ['0', '1', '2', '3']:
                class_images = []
                
                # Look for image files in this directory
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in self.image_extensions):
                            file_path = os.path.join(root, file)
                            class_images.append(file_path)
                
                if class_images:
                    image_files[item] = class_images
                    self.logger.info(f"Found {len(class_images)} images in class {item}")
        
        # If no organized structure found in root, check subdirectories
        if not image_files:
            self.logger.info("Searching in subdirectories...")
            for root, dirs, files in os.walk(self.data_path):
                for dir_name in dirs:
                    if dir_name in ['0', '1', '2', '3']:
                        dir_path = os.path.join(root, dir_name)
                        class_images = []
                        
                        for file in os.listdir(dir_path):
                            file_path = os.path.join(dir_path, file)
                            if any(file.lower().endswith(ext) for ext in self.image_extensions):
                                class_images.append(file_path)
                        
                        if class_images:
                            image_files[dir_name] = class_images
                            self.logger.info(f"Found {len(class_images)} images in class {dir_name}")
        
        if not image_files:
            self.logger.warning(f"No image files found in {self.data_path}")
            # Try to find any images in the directory structure
            all_images = []
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self.image_extensions):
                        file_path = os.path.join(root, file)
                        all_images.append(file_path)
            
            if all_images:
                image_files['all'] = all_images
                self.logger.info(f"Found {len(all_images)} images in total")
        
        return image_files
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image using PIL to avoid Qt issues.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array or None if loading fails
        """
        try:
            from PIL import Image
            # Use PIL to load image (no Qt dependencies)
            pil_image = Image.open(image_path)
            # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert to numpy array
            image_array = np.array(pil_image)
            return image_array
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def extract_image_metadata(self, image_path: str) -> Dict:
        """
        Extract basic metadata from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing image metadata
        """
        try:
            image = self.load_image(image_path)
            if image is None:
                return {'file_path': image_path, 'error': 'Could not load image'}
            
            metadata = {
                'file_path': image_path,
                'file_name': os.path.basename(image_path),
                'file_size': os.path.getsize(image_path),
                'width': image.shape[1],
                'height': image.shape[0],
                'channels': image.shape[2] if len(image.shape) > 2 else 1,
                'aspect_ratio': image.shape[1] / image.shape[0],
                'file_extension': os.path.splitext(image_path)[1].lower()
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return {
                'file_path': image_path,
                'file_name': os.path.basename(image_path),
                'error': str(e)
            }
    
    def analyze_image_features(self, image: np.ndarray) -> Dict:
        """
        Analyze features of a medical image for bowel preparation quality assessment.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Dictionary containing image features
        """
        try:
            # Convert to grayscale for some analyses
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Basic intensity statistics
            features = {
                'mean_intensity': np.mean(gray),
                'std_intensity': np.std(gray),
                'min_intensity': np.min(gray),
                'max_intensity': np.max(gray),
                'median_intensity': np.median(gray),
                'image_entropy': measure.shannon_entropy(gray),
            }
            
            # Histogram analysis
            hist, bins = np.histogram(gray.flatten(), bins=256, range=[0, 256])
            hist = hist / hist.sum()  # Normalize
            features['histogram_skewness'] = stats.skew(hist)
            features['histogram_kurtosis'] = stats.kurtosis(hist)
            
            # Texture analysis using GLCM
            try:
                glcm = feature.graycomatrix(gray, [1], [0], symmetric=True, normed=True)
                features['contrast'] = feature.graycoprops(glcm, 'contrast')[0, 0]
                features['homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]
                features['energy'] = feature.graycoprops(glcm, 'energy')[0, 0]
                features['correlation'] = feature.graycoprops(glcm, 'correlation')[0, 0]
            except Exception as e:
                self.logger.debug(f"GLCM analysis failed: {e}")
                features.update({'contrast': 0, 'homogeneity': 0, 'energy': 0, 'correlation': 0})
            
            # Edge analysis (important for bowel texture)
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            # Color analysis in different spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # HSV analysis
            features['hue_mean'] = np.mean(hsv[:, :, 0])
            features['saturation_mean'] = np.mean(hsv[:, :, 1])
            features['value_mean'] = np.mean(hsv[:, :, 2])
            
            # LAB analysis (perceptually uniform)
            features['l_mean'] = np.mean(lab[:, :, 0])
            features['a_mean'] = np.mean(lab[:, :, 1])
            features['b_mean'] = np.mean(lab[:, :, 2])
            
            # Medical image specific features
            # Blob analysis for potential polyps or features
            try:
                # Simple blob detection
                blobs = feature.blob_dog(gray, max_sigma=30, threshold=0.1)
                features['blob_count'] = len(blobs)
            except:
                features['blob_count'] = 0
            
            # Sharpness measure (important for quality assessment)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            features['sharpness'] = sharpness
            
            # Local binary patterns for texture
            try:
                lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
                features['lbp_entropy'] = measure.shannon_entropy(lbp.astype(np.uint8))
            except:
                features['lbp_entropy'] = 0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error analyzing image features: {e}")
            return {}
    
    def create_image_montage(self, images: List[np.ndarray], output_path: str, 
                           titles: Optional[List[str]] = None, max_images: int = 12) -> None:
        """
        Create a montage of multiple images.
        
        Args:
            images: List of images as numpy arrays
            output_path: Path to save the montage
            titles: Optional titles for each image
            max_images: Maximum number of images to include in montage
        """
        if not images:
            return
        
        # Limit number of images
        images = images[:max_images]
        
        # Calculate grid dimensions
        n_images = len(images)
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        # Resize images to have the same size
        target_size = (200, 200)  # width, height
        resized_images = []
        
        for img in images:
            resized_img = cv2.resize(img, target_size)
            resized_images.append(resized_img)
        
        # Create montage grid
        montage = np.zeros((n_rows * target_size[1], n_cols * target_size[0], 3), dtype=np.uint8)
        
        for i, img in enumerate(resized_images):
            row = i // n_cols
            col = i % n_cols
            y_start = row * target_size[1]
            y_end = y_start + target_size[1]
            x_start = col * target_size[0]
            x_end = x_start + target_size[0]
            montage[y_start:y_end, x_start:x_end] = img
        
        # Add titles if provided
        if titles and len(titles) >= len(images):
            title_height = 30
            montage_with_titles = np.zeros((montage.shape[0] + title_height, montage.shape[1], 3), dtype=np.uint8)
            montage_with_titles[title_height:] = montage
            
            # Add text for each image
            for i, title in enumerate(titles[:len(images)]):
                row = i // n_cols
                col = i % n_cols
                x_pos = col * target_size[0] + 10
                y_pos = 20
                cv2.putText(montage_with_titles, title, 
                          (x_pos, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            montage = montage_with_titles
        
        # Save montage
        cv2.imwrite(output_path, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    
    def save_sample_images(self, image_files: Dict[str, List[str]], 
                          samples_per_class: int = 5, 
                          output_dir: str = "sample_images") -> None:
        """
        Save sample images from each class for visualization.
        
        Args:
            image_files: Dictionary of image files by class
            samples_per_class: Number of samples to save per class
            output_dir: Directory to save sample images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for class_label, images in image_files.items():
            class_dir = os.path.join(output_dir, f"class_{class_label}")
            os.makedirs(class_dir, exist_ok=True)
            
            # Sample images from this class
            sample_images = images[:samples_per_class]
            
            for i, image_path in enumerate(sample_images):
                image = self.load_image(image_path)
                if image is not None:
                    # Resize for consistency
                    image_resized = cv2.resize(image, (300, 300))
                    output_path = os.path.join(class_dir, f"sample_{i+1}_{os.path.basename(image_path)}")
                    cv2.imwrite(output_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))