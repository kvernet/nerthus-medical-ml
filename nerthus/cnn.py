import os
# Force CPU only to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from .utils import setup_logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from .processor import ImageProcessor
from .utils import ensure_directory

# Configure TensorFlow to use CPU and be less verbose
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

class NerthusCNN:
    """
    CNN-based classifier for bowel preparation quality assessment.
    Uses simplified architecture to avoid GPU issues.
    """
    
    def __init__(self, input_shape=(150, 150, 3), num_classes=4, models_dir="outputs/models"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models_dir = models_dir
        self.model = None
        self.history = None
        self.logger = setup_logging()

        ensure_directory(models_dir)
        
        self.logger.info("NerthusCNN initialized (CPU mode)")
    
    def build_simple_cnn(self):
        """
        Build a very simple CNN to test the pipeline.
        """
        self.logger.info("Building simple CNN architecture...")
        
        model = keras.Sequential([
            # First simple block
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            
            # Second simple block  
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third simple block
            layers.Conv2D(64, (3, 3), activation='relu'), 
            layers.MaxPooling2D((2, 2)),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Simple compiler
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.logger.info("Simple CNN built successfully")
        return model
    
    def build_custom_cnn(self):
        """
        Build custom CNN - fallback to simple version for now.
        """
        return self.build_simple_cnn()
    
    def create_data_generators(self, data_path: str, batch_size: int = 32, 
                             validation_split: float = 0.2):
        """
        Create data generators for training and validation.
        
        Args:
            data_path: Path to the dataset root directory
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        """
        self.logger.info("Creating data generators...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Train generator
        train_generator = train_datagen.flow_from_directory(
            data_path,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='sparse',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            data_path,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation',
            shuffle=False
        )
        
        self.logger.info(f"Training samples: {train_generator.samples}")
        self.logger.info(f"Validation samples: {val_generator.samples}")
        self.logger.info(f"Class indices: {train_generator.class_indices}")
        
        return train_generator, val_generator

    def train(self, train_generator, val_generator, epochs: int = 20):  # Reduced for CPU
        """
        Train the CNN model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of training epochs
        """
        self.logger.info("Starting CNN training...")
        
        # Simple callbacks for CPU training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,  # Reduced for faster training
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Less aggressive
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=None,
            verbose=1
        )
        
        self.logger.info("CNN training completed")
        return self.history
    
    def evaluate(self, test_generator):
        """
        Evaluate the trained model.
        
        Args:
            test_generator: Test data generator
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        self.logger.info("Evaluating CNN model...")
        
        results = self.model.evaluate(test_generator, verbose=0)
        loss, accuracy = results[0], results[1]
        
        self.logger.info(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
        return loss, accuracy
    
    def plot_training_history(self, save_path: str = "cnn_training_history.png"):
        """
        Plot training history for visualization.
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training history plot saved to: {save_path}")
    
    def save_model(self, model_name: str = "nerthus_cnn"):
        """
        Save the trained model using modern Keras format.
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Use modern .keras format instead of .h5
        model_path = os.path.join(self.models_dir, f"{model_name}.keras")
        self.model.save(model_path)
        self.logger.info(f"Model saved to: {model_path}")
        return model_path
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained model.
        """
        self.model = keras.models.load_model(model_path)
        self.logger.info(f"Model loaded from: {model_path}")
        return self.model