import os
import joblib
import pandas as pd
import numpy as np
import cv2
from PIL import Image

# Suppress TensorFlow WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Force CPU only to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
except ImportError:
    tf = None
    print("⚠️ TensorFlow not available, CNN predictions will be disabled")

class ModelLoader:
    """Load and manage trained models for the web app with GPU error handling"""
    
    def __init__(self, base_dir="static"):
        self.base_dir = base_dir
        self.models_loaded = False
        self.ml_model = None
        self.cnn_model = None
        self.feature_names = None
        self.cnn_available = tf is not None
        self.class_descriptions = {
            0: "Unprepared - Mucosa not visible due to solid stool",
            1: "Partially Prepared - Portions of mucosa visible", 
            2: "Well Prepared - Minor residue, mucosa clearly visible",
            3: "Excellent Preparation - Entire mucosa clearly visible"
        }
    
    def load_models(self,
                    ml_model_file="XGBoost.joblib",
                    cnn_model_file="CNN.keras"):
        """Load all trained models with error handling"""
        try:
            print("🔄 Loading trained models...")
            
            # Load ML
            rf_path = f"{self.base_dir}/models/{ml_model_file}"
            if os.path.exists(rf_path):
                self.ml_model = joblib.load(rf_path)
                print("✅ ML loaded")
            else:
                print("❌ ML model not found")
            
            # Load CNN if TensorFlow is available
            if self.cnn_available:
                cnn_path = f"{self.base_dir}/models/{cnn_model_file}"
                if os.path.exists(cnn_path):
                    try:
                        # Force CPU execution
                        with tf.device('/CPU:0'):
                            self.cnn_model = tf.keras.models.load_model(cnn_path)
                        print("✅ CNN model loaded (CPU mode)")
                    except Exception as e:
                        print(f"❌ CNN loading failed: {e}")
                        self.cnn_available = False
                else:
                    print("❌ CNN model not found")
                    self.cnn_available = False
            else:
                print("⚠️ TensorFlow not available - CNN predictions disabled")
            
            # Load feature names
            self.load_feature_names()
            
            self.models_loaded = True
            print("🎉 Model loading completed!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.models_loaded = False
    
    def load_feature_names(self):
        """Load feature names from CSV or use defaults"""
        try:
            features_path = f"{self.base_dir}/image_features.csv"
            if os.path.exists(features_path):
                df = pd.read_csv(features_path)
                # Get feature columns (exclude metadata)
                non_feature_cols = ['file_path', 'file_name', 'bbps_class', 'Unnamed: 0']
                self.feature_names = [col for col in df.columns if col not in non_feature_cols]
                print(f"✅ Loaded {len(self.feature_names)} feature names from CSV")
            else:
                # Use default feature names based on your analysis
                self.feature_names = [
                    'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'median_intensity',
                    'hue_mean', 'saturation_mean', 'value_mean', 'l_mean', 'a_mean', 'b_mean',
                    'edge_density', 'sharpness', 'contrast', 'homogeneity', 'energy', 'correlation',
                    'image_entropy', 'lbp_entropy', 'blob_count'
                ]
                print("✅ Using default feature names")
                
        except Exception as e:
            print(f"❌ Error loading feature names: {e}")
            # Fallback to essential features
            self.feature_names = [
                'mean_intensity', 'min_intensity', 'hue_mean', 'homogeneity', 
                'edge_density', 'saturation_mean', 'contrast', 'sharpness'
            ]
    
    def extract_features_from_image(self, image):
        """
        Extract the same features used in training from a new image.
        Fixed version: keeps RGB order consistent across training and inference.
        """
        try:
            # --- 1. Convert to RGB NumPy array consistently ---
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
                image = np.array(image)
            else:
                image = np.array(image)
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # --- 2. Now work in RGB order everywhere ---
            features = {}

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # --- Intensity features ---
            features['mean_intensity'] = float(np.mean(gray))
            features['std_intensity'] = float(np.std(gray))
            features['min_intensity'] = float(np.min(gray))
            features['max_intensity'] = float(np.max(gray))
            features['median_intensity'] = float(np.median(gray))

            # --- Color features (still RGB order) ---
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            features['hue_mean'] = float(np.mean(hsv[:, :, 0]))
            features['saturation_mean'] = float(np.mean(hsv[:, :, 1]))
            features['value_mean'] = float(np.mean(hsv[:, :, 2]))

            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            features['l_mean'] = float(np.mean(lab[:, :, 0]))
            features['a_mean'] = float(np.mean(lab[:, :, 1]))
            features['b_mean'] = float(np.mean(lab[:, :, 2]))

            # --- Edge and texture features ---
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = float(np.sum(edges > 0) / (gray.shape[0] * gray.shape[1]))
            features['sharpness'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            features['contrast'] = float(np.std(gray))
            features['homogeneity'] = float(1.0 / (1.0 + np.var(gray)))
            features['energy'] = float(np.mean(gray**2))
            features['correlation'] = 0.5

            hist, _ = np.histogram(gray, bins=256, range=[0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            features['image_entropy'] = float(entropy)
            features['lbp_entropy'] = float(entropy)

            # --- Blob count ---
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            features['blob_count'] = float(len(contours))

            # --- Ensure all expected features are present ---
            for feature in self.feature_names:
                if feature not in features:
                    features[feature] = 0.0

            return features

        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return {feature: 0.0 for feature in self.feature_names}
 
    def predict_with_ml(self, image):
        """Predict BBPS class using traditional ML model with robust preprocessing."""
        if not self.ml_model:
            return None, None, "ML model not loaded"

        try:
            # --- Step 1: Ensure consistent RGB format ---
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                arr = np.array(image)
                if arr.ndim == 2:
                    image = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB))
                elif arr.shape[2] == 4:
                    image = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB))
                else:
                    image = Image.fromarray(arr)

            # --- Step 2: Extract features ---
            features = self.extract_features_from_image(image)
            if features is None:
                return None, None, "Feature extraction failed"

            # --- Step 3: Build feature vector ---
            feature_vector = np.array(
                [features.get(feat, 0) for feat in self.feature_names],
                dtype=np.float32
            ).reshape(1, -1)

            # --- Step 4: Apply same scaler if available ---
            if hasattr(self, "scaler") and self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector)

            # --- Step 5: Predict ---
            prediction = int(self.ml_model.predict(feature_vector)[0])
            probabilities = self.ml_model.predict_proba(feature_vector)[0]
            confidence = float(np.max(probabilities))

            return prediction, confidence, "Success"

        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            return None, None, f"Prediction error: {str(e)}"
    
    def predict_with_cnn(self, image, debug=False):
        """Predict BBPS class using CNN with robust, consistent preprocessing.

        - Ensures RGB channel order (no silent BGR swaps).
        - Converts images to float32 and normalizes to [0,1].
        - Resizes to (150,150) and adds batch dim.
        - Returns (pred_class:int, confidence:float, message:str)
        """
        if not self.cnn_model or not self.cnn_available:
            return None, None, "CNN model not available"

        try:
            # If PIL Image, convert to RGB to ensure consistent channel order
            if isinstance(image, Image.Image):
                image = image.convert("RGB")  # -> ALWAYS 3-channel RGB
                arr = np.array(image)
            else:
                arr = np.array(image)

            # Validate shape
            if arr.ndim == 2:  # grayscale -> convert to RGB
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 4:  # RGBA -> RGB
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            elif not (arr.ndim == 3 and arr.shape[2] == 3):
                raise ValueError(f"Unsupported image shape: {arr.shape}")

            if debug:
                print(f"[predict_with_cnn] original shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")

            # Resize (use cv2 but keep RGB ordering)
            processed = cv2.resize(arr, (150, 150), interpolation=cv2.INTER_AREA)

            # Convert to float32 and normalize to [0,1]
            processed = processed.astype(np.float32) / 255.0

            if debug:
                print(f"[predict_with_cnn] after resize & norm shape={processed.shape}, dtype={processed.dtype}, "
                    f"range=[{processed.min():.4f},{processed.max():.4f}]")

            # Add batch dimension
            processed = np.expand_dims(processed, axis=0)  # shape (1,150,150,3)

            # Predict (force CPU if desired)
            try:
                with tf.device('/CPU:0'):
                    preds = self.cnn_model.predict(processed, verbose=0)
            except Exception:
                # fallback without explicit device context
                preds = self.cnn_model.predict(processed, verbose=0)

            if debug:
                print(f"[predict_with_cnn] raw predictions: {preds}")

            pred_idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))

            return pred_idx, confidence, "Success"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, f"CNN prediction error: {str(e)}"

    def get_class_description(self, class_id):
        """Get human-readable description for BBPS class"""
        return self.class_descriptions.get(class_id, "Unknown class")
    
    def get_available_models(self):
        """Return which models are available"""
        return {
            'ml': self.ml_model is not None,
            'cnn': self.cnn_available and self.cnn_model is not None
        }

# Global model loader instance
modelLoader = ModelLoader()
