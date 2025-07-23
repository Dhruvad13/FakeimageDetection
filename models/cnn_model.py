"""
CNN Model for Fake Image Detection
This module contains a custom CNN architecture for detecting fake images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

class CNNModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Initialize CNN Model for Fake Image Detection
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of classes (2 for fake/real)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the CNN architecture"""
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fifth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
            
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and metrics"""

        if self.model is None:
            self.build_model()

    # Use Adam optimizer with the given learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile with all required metrics
        self.model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
        return self.model
    
    def get_data_generators(self, train_dir, validation_dir, batch_size=32, 
                          image_size=(224, 224), augment=True):
        """
        Create data generators for training and validation
        
        Args:
            train_dir: Directory containing training data
            validation_dir: Directory containing validation data
            batch_size: Batch size for training
            image_size: Size to resize images to
            augment: Whether to apply data augmentation
        """
        
        if augment:
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest',
                validation_split=0.2
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def get_callbacks(self, model_name='cnn_model', monitor='val_accuracy', 
                     patience=10, save_best_only=True):
        """Get callbacks for training"""
        
        callbacks_list = [
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=f'saved_models/{model_name}_best.h5',
                monitor=monitor,
                save_best_only=save_best_only,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(f'saved_models/{model_name}_training_log.csv')
        ]
        
        return callbacks_list
    
    def train(self, train_generator, validation_generator, epochs=50, 
              callbacks_list=None, verbose=1):
        """
        Train the CNN model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            callbacks_list: List of callbacks to use during training
            verbose: Verbosity level
        """
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if callbacks_list is None:
            callbacks_list = self.get_callbacks()
        
        # Calculate steps per epoch
        steps_per_epoch = len(train_generator)
        validation_steps = len(validation_generator)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        self.history = history
        return history
    
    def evaluate(self, test_generator, verbose=1):
        """Evaluate the model on test data"""
        
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(
            test_generator,
            verbose=verbose
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy
    
    def predict(self, images, threshold=0.5):
        """
        Make predictions on images
        
        Args:
            images: Input images (numpy array or single image)
            threshold: Threshold for binary classification
        
        Returns:
            predictions: Array of predictions
            probabilities: Array of prediction probabilities
        """
        
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        # Make predictions
        probabilities = self.model.predict(images)
        
        # Get class predictions
        predictions = np.argmax(probabilities, axis=1)
        
        # Get confidence scores
        confidence_scores = np.max(probabilities, axis=1)
        
        return predictions, probabilities, confidence_scores
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        
        if self.history is None:
            raise ValueError("No training history found. Train the model first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CNN Model Training History', fontsize=16, fontweight='bold')
        
        # Plot training & validation accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot training & validation loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Plot validation accuracy vs loss
        axes[1, 1].scatter(self.history.history['val_loss'], 
                          self.history.history['val_accuracy'], 
                          alpha=0.6)
        axes[1, 1].set_title('Validation Accuracy vs Loss')
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, test_generator, class_names=None):
        """Plot confusion matrix"""
        
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        # Get predictions
        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        
        # Get true labels
        y_true = test_generator.classes
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names or ['Fake', 'Real'],
                   yticklabels=class_names or ['Fake', 'Real'])
        plt.title('Confusion Matrix - CNN Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                  target_names=class_names or ['Fake', 'Real']))
        
        return cm
    
    def save_model(self, filepath):
        """Save the complete model"""
        
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")
        
        return self.model
    
    def get_model_summary(self):
        """Get model summary"""
        
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        return self.model.summary()
    
    def visualize_feature_maps(self, image, layer_names=None):
        """Visualize feature maps for a given image"""
        
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        if layer_names is None:
            # Get some convolutional layer names
            layer_names = [layer.name for layer in self.model.layers 
                          if 'conv' in layer.name][:8]  # First 8 conv layers
        
        # Create model for feature extraction
        layer_outputs = [self.model.get_layer(name).output for name in layer_names]
        activation_model = models.Model(inputs=self.model.input, outputs=layer_outputs)
        
        # Get activations
        activations = activation_model.predict(np.expand_dims(image, axis=0))
        
        # Plot feature maps
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        fig.suptitle('Feature Maps Visualization', fontsize=16)
        
        for i, (layer_name, activation) in enumerate(zip(layer_names, activations)):
            if i >= 8:  # Only show first 8 layers
                break
            
            # Get first feature map
            feature_map = activation[0, :, :, 0]
            
            row = i // 4
            col = i % 4
            
            axes[row, col].imshow(feature_map, cmap='viridis')
            axes[row, col].set_title(f'{layer_name}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage and utility functions
def create_cnn_model():
    """Create and return a CNN model instance"""
    
    cnn = CNNModel(input_shape=(224, 224, 3), num_classes=2)
    model = cnn.build_model()
    compiled_model = cnn.compile_model(learning_rate=0.001)
    
    return cnn

def train_cnn_model(train_dir, epochs=50, batch_size=32):
    """Train a CNN model with given parameters"""
    
    # Create CNN model
    cnn = create_cnn_model()
    
    # Get data generators
    train_gen, val_gen = cnn.get_data_generators(
        train_dir=train_dir,
        validation_dir=train_dir,  # Using same dir with split
        batch_size=batch_size,
        image_size=(224, 224),
        augment=True
    )
    
    # Get callbacks
    callbacks_list = cnn.get_callbacks(
        model_name='cnn_fake_detector',
        monitor='val_accuracy',
        patience=10
    )
    
    # Train model
    history = cnn.train(
        train_generator=train_gen,
        validation_generator=val_gen,
        epochs=epochs,
        callbacks_list=callbacks_list
    )
    
    return cnn, history

if __name__ == "__main__":
    # Example usage
    print("🚀 CNN Model for Fake Image Detection")
    print("=" * 50)
    
    # Create model
    cnn_model = create_cnn_model()
    
    # Print model summary
    print("\n📊 Model Architecture:")
    print(cnn_model.get_model_summary())
    
    print("\n✅ CNN Model created successfully!")
    print("Use train_cnn_model() to start training.")