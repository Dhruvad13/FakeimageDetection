import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import argparse
import json
from datetime import datetime

# Custom imports
from models.cnn_model import CNNModel
from models.resnet_model import ResNetFakeImageDetector
from utils.data_preprocessing import DataPreprocessor
from utils.visualization import visualizer
from utils.metrics import MetricsCalculator

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train fake image detection model')
    parser.add_argument('--model', type=str, default='cnn', 
                        choices=['cnn', 'resnet', 'advanced_cnn'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='models/saved_models')
    
    # ✅ Add this line:
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with fewer samples')

    return parser.parse_args()


def create_callbacks(model_save_path):
    return [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy',
                        save_best_only=True, save_weights_only=False, mode='max', verbose=1)
    ]


    
def train_model(args):
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print(f"\nTraining model: {args.model} | Image Size: {args.img_size}x{args.img_size} | "
          f"Batch Size: {args.batch_size} | LR: {args.learning_rate} | Epochs: {args.epochs}")

    preprocessor = DataPreprocessor(img_size=(args.img_size, args.img_size))
    train_gen, val_gen, test_gen = preprocessor.prepare_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size)
    )

    num_classes = 2
    class_names = ['real', 'fake']
    input_shape = (args.img_size, args.img_size, 3)

    # Build model
    if args.model == 'cnn':
        model_builder = CNNModel(input_shape=input_shape, num_classes=num_classes)
        model = model_builder.build_model()
        model_builder.compile_model(args.learning_rate)
        trained_model = model_builder.model
    elif args.model == 'resnet':
        model_builder = ResNetFakeImageDetector(input_shape=input_shape, num_classes=num_classes)
        model = model_builder.build_model()
        model_builder.compile_model(args.learning_rate)
        trained_model = model_builder.model

    # Re-compile for metrics
    trained_model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    trained_model.summary()

    # Save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model}_fake_detection_{timestamp}.h5"
    model_save_path = os.path.join(args.save_dir, model_name)

    # Callbacks
    callbacks = create_callbacks(model_save_path)

    # Train
    print("\nTraining...")
    history = trained_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy, test_precision, test_recall = trained_model.evaluate(test_gen, verbose=1)

    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    # Predict and calculate metrics
    y_true = []
    y_pred = []
    y_pred_proba = []

    for X_batch, y_batch in test_gen:
        preds = trained_model.predict(X_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        y_pred_proba.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    # Metrics
    metrics = MetricsCalculator.calculate_basic_metrics(y_true, y_pred, y_pred_proba)

    print("\nClassification Report:")
    MetricsCalculator().print_classification_report(y_true, y_pred, class_names)

    # Visualizations
    print("\nPlotting visualizations...")
    visualizer.plot_training_history(history.history)
    visualizer.plot_confusion_matrix(y_true, y_pred, class_names)

    # Save training info
    training_info = {
        'model_type': args.model,
        'timestamp': timestamp,
        'model_path': model_save_path,
        'training_params': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'img_size': args.img_size
        },
        'results': {
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_loss': float(test_loss)
        },
        'data_info': {
            'num_classes': num_classes,
            'class_names': class_names
        }
    }

    info_path = os.path.join('logs', f'training_info_{timestamp}.json')
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=4)

    print(f"\n✅ Training complete. Model saved to: {model_save_path}")
    print(f"ℹ️ Training info saved to: {info_path}")
    
    return trained_model, history, metrics


def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    args = parse_arguments()
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    model, history, metrics = train_model(args)
    print("\n✔️ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
