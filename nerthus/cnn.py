import os
# Suppress TensorFlow WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Force CPU only to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import matplotlib.pyplot as plt
from .utils import setup_logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from .utils import cnn_generate_text_report, ensure_directory

# Configure TensorFlow to use CPU and be less verbose
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

class NerthusCNN:
    """
    CNN-based classifier for bowel preparation quality assessment.
    Uses simplified architecture to avoid GPU issues.
    """
    
    def __init__(self,
                 input_shape=(150, 150, 3),
                 num_classes=4,
                 random_state=32,
                 output_dir="outputs/cnn"
    ):
        # Set random seeds globally
        np.random.seed(random_state)
        random.seed(random_state)
        tf.random.set_seed(random_state)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.random_state = random_state
        self.models_dir = f"{output_dir}/models"
        self.results_dir = f"{output_dir}/results"
        self.model = None
        self.history = None
        self.logger = setup_logging()

        ensure_directory(self.models_dir)
        ensure_directory(self.results_dir)
        
        self.logger.info("Nerthus CNN (CPU mode)")
    
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
    
    def build_improved_cnn(self):
        """
        Build CNN with regularization to prevent overfitting.
        """
        self.logger.info("Building improved CNN with regularization...")
        
        model = keras.Sequential([
            # First block with regularization
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),  # Increased dropout
            
            # Second block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),  # Increased dropout
            
            # Third block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Fourth block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),  # Better than Flatten for regularization
            layers.Dropout(0.5),
            
            # Classifier with L2 regularization
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile with lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Lower LR
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.logger.info("Improved CNN built successfully")
        return model
    
    def build_tunable_cnn(self, dropout_rates=[0.3, 0.4, 0.4, 0.5]):
        """
        Build CNN with customizable dropout rates.
        """
        model = keras.Sequential([
            # First block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rates[0]),
            
            # Second block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rates[1]),
            
            # Third block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rates[2]),
            
            # Classifier
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rates[3]),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.logger.info("Tunable CNN with customizable dropout rates built successfully")
        return model

    def create_data_generators(self,
                               data_path: str,
                               batch_size: int = 32,
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
            seed=self.random_state,
            shuffle=True
        )
        
        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            data_path,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation',
            seed=self.random_state,
            shuffle=False
        )
        
        self.logger.info(f"Training samples: {train_generator.samples}")
        self.logger.info(f"Validation samples: {val_generator.samples}")
        self.logger.info(f"Class indices: {train_generator.class_indices}")
        
        return train_generator, val_generator

    def train(self, train_generator, val_generator, epochs: int = 100):
        """
        Train the CNN model with improved early stopping.
        """
        self.logger.info("Starting CNN training...")
        
        # Improved callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',  # Stop based on validation accuracy
                patience=15,  # Wait 15 epochs after best
                restore_best_weights=True,  # Restore weights from best epoch
                mode='max'  # Maximize validation accuracy
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,  # More patience for LR reduction
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.models_dir, 'best_cnn_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load the best model (not the final overfitted one)
        self.model = keras.models.load_model(
            os.path.join(self.models_dir, 'best_cnn_model.keras')
        )
        
        self.logger.info("CNN training completed with best weights restored")
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
    
    def plot_training_history(self, file_name: str = "cnn_training_history.png"):
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
        
        save_path = f"{self.results_dir}/{file_name}"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training history plot saved to: {save_path}")
        return save_path
    
    def save_model(self, model_name: str = "nerthus_cnn.keras"):
        """
        Save the trained model using modern Keras format.
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Use modern .keras format instead of .h5
        model_path = os.path.join(self.models_dir, f"{model_name}")
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


def main():
    """Main entry point for the nerthus-cnn command."""
    from .utils import get_data_path
    import argparse
    
    parser = argparse.ArgumentParser(description='Nerthus Medical CNN Pipeline')
    
    parser.add_argument(
        "-c", "--num_classes",
        type=int,
        default=4,
        help="Number of classes (default: 4)"
    )

    parser.add_argument(
        "-s", "--random_state",
        type=int,
        default=42,
        help="Random state (default: 42)"
    )

    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )

    parser.add_argument(
        "-a", "--validation_split",
        type=float,
        default=0.2,
        help="Validation split (default: 0.2)"
    )

    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=200,
        help="Number of epochs (default: 200)"
    )

    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="outputs/cnn",
        help="Output directory (default: outputs/cnn)"
    )

    args = parser.parse_args()
    
    print("NERTHUS MEDICAL ML - IMPROVED CNN PIPELINE")
    print("=" * 45)
    
    data_path = get_data_path()
    data_path = f"{data_path}/nerthus-dataset-frames/nerthus-dataset-frames"

    # Initialize improved CNN
    cnn = NerthusCNN(
        input_shape=(150, 150, 3),
        num_classes=args.num_classes,
        random_state=args.random_state,
        output_dir=args.output_dir
    )
    
    # Build improved architecture
    print("🤖 Building tunable CNN with regularization...")
    #cnn.build_improved_cnn()
    model = cnn.build_tunable_cnn(
        dropout_rates=[0.1, 0.2, 0.3, 0.2]
    )
    
    # Create data generators with less augmentation
    train_gen, val_gen = cnn.create_data_generators(
        data_path=data_path,
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )
    
    # Train with improved settings
    print("🚀 Training tunable CNN...")
    history = cnn.train(
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=args.epochs
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
        report_path=f"{args.output_dir}/cnn_performance_report.txt"
    )

    # Plot training history
    history_path = cnn.plot_training_history(
        file_name="nerthus_cnn_training_history.png"
    )
    
    # Save & load the best model
    model_path = cnn.save_model(model_name="nerthus_cnn.keras")
    
    # Get best validation accuracy
    best_val_accuracy = max(history.history['val_accuracy'])
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    print(f"\n🎯 TRAINING RESULTS:")
    print(f"   Best Validation Accuracy: {best_val_accuracy:.1%}")
    print(f"   Final Validation Accuracy: {final_val_accuracy:.1%}")
    print(f"   Final Training Accuracy: {history.history['accuracy'][-1]:.1%}")

    print(f"\n✅ CNN complete!")
    print(f"   - Model saved: {model_path}")
    print(f"   - Training plot: {history_path}")

if __name__ == "__main__":
    main()