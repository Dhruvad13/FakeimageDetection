import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os

class ResNetFakeImageDetector:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Initialize ResNet model for fake image detection
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of classes (2 for fake/real)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self, fine_tune=True, dropout_rate=0.5):
        """
        Build ResNet50 model with custom classification head
        
        Args:
            fine_tune: Whether to fine-tune the pre-trained weights
            dropout_rate: Dropout rate for regularization
        """
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=self.input_shape)
        
        # Preprocessing layers
        x = layers.Rescaling(1./255)(inputs)
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Global pooling and classification layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate/2)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Fine-tuning: unfreeze top layers of base model
        if fine_tune:
            base_model.trainable = True
            # Fine-tune from this layer onwards
            fine_tune_at = 100
            
            # Freeze all layers before fine_tune_at
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            
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

        
        print(f"Model built successfully with {self.model.count_params():,} parameters")
        return self.model
    
    def get_callbacks(self, model_save_path='models/saved_models/best_resnet_model.h5'):
        """
        Get training callbacks
        
        Args:
            model_save_path: Path to save the best model
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, train_generator, validation_generator, epochs=50, 
              model_save_path='models/saved_models/best_resnet_model.h5'):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            model_save_path: Path to save the best model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Get callbacks
        callbacks = self.get_callbacks(model_save_path)
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def predict_image(self, image_path):
        """
        Predict if an image is fake or real
        
        Args:
            image_path: Path to the image file
            
        Returns:
            prediction: Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Build and train the model first.")
        
        # Load and preprocess image
        image = keras.preprocessing.image.load_img(
            image_path, 
            target_size=self.input_shape[:2]
        )
        image_array = keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image_array, verbose=0)
        confidence = np.max(predictions[0])
        predicted_class = np.argmax(predictions[0])
        
        # Class mapping (0: fake, 1: real)
        class_labels = ['Fake', 'Real']
        predicted_label = class_labels[predicted_class]
        
        # Calculate fake percentage
        fake_percentage = predictions[0][0] * 100
        real_percentage = predictions[0][1] * 100
        
        result = {
            'predicted_class': predicted_label,
            'confidence': float(confidence),
            'fake_percentage': float(fake_percentage),
            'real_percentage': float(real_percentage),
            'raw_predictions': predictions[0].tolist()
        }
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Predict multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            predictions: List of prediction results
        """
        predictions = []
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                result['image_path'] = image_path
                predictions.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
        
        return predictions
    
    def evaluate_model(self, test_generator):
        """
        Evaluate the model on test data
        
        Args:
            test_generator: Test data generator
            
        Returns:
            evaluation_results: Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Build and train the model first.")
        
        # Evaluate the model
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            test_generator,
            steps=len(test_generator),
            verbose=1
        )
        
        # Calculate F1 score
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        evaluation_results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'f1_score': f1_score
        }
        
        print(f"Test Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        
        return evaluation_results
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """
        Get model summary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self.model.summary()
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
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
        
        # Plot training & validation precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot training & validation recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage
if __name__ == "__main__":
    # Initialize the model
    detector = ResNetFakeImageDetector(input_shape=(224, 224, 3), num_classes=2)
    
    # Build the model
    model = detector.build_model(fine_tune=True, dropout_rate=0.5)
    
    # Print model summary
    print(detector.get_model_summary())
    
    # Example prediction (uncomment when you have a trained model)
    # result = detector.predict_image('path/to/your/image.jpg')
    # print(f"Prediction: {result['predicted_class']}")
    # print(f"Confidence: {result['confidence']:.2f}")
    # print(f"Fake Percentage: {result['fake_percentage']:.2f}%")
    # print(f"Real Percentage: {result['real_percentage']:.2f}%")