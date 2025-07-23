import os
import numpy as np
import cv2
import argparse
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


from utils.visualization import visualize_prediction

class FakeImagePredictor:
    def __init__(self, model_path, img_size=(224, 224)):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path (str): Path to the trained model
            img_size (tuple): Image size for preprocessing
        """
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.class_names = ['fake', 'real']  # Default class names
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            print(f"Loading model from {self.model_path}")
            self.model = load_model(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            np.array: Preprocessed image array
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image from {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # If it's already an array
                image = image_path
            
            # Resize image
            image = cv2.resize(image, self.img_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def predict_single_image(self, image_path, return_probabilities=True):
        """
        Predict if a single image is fake or real
        
        Args:
            image_path (str): Path to the image
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results
        """
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get prediction results
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        
        # Calculate fake probability
        fake_probability = predictions[0][0] if self.class_names[0] == 'fake' else predictions[0][1]
        real_probability = predictions[0][1] if self.class_names[0] == 'fake' else predictions[0][0]
        
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'fake_probability': float(fake_probability),
            'real_probability': float(real_probability),
            'is_fake': predicted_class == 'fake'
        }
        
        if return_probabilities:
            result['class_probabilities'] = {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Predict multiple images
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error predicting {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def analyze_image_features(self, image_path):
        """
        Analyze image features that might indicate manipulation
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            dict: Analysis results
        """
        try:
            # Load original image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Basic image statistics
            height, width, channels = image.shape
            
            # Color analysis
            mean_color = np.mean(image_rgb, axis=(0, 1))
            std_color = np.std(image_rgb, axis=(0, 1))
            
            # Edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            # Noise analysis
            noise_level = np.std(gray)
            
            # Compression artifacts (simplified)
            blur_level = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            analysis = {
                'image_dimensions': (width, height),
                'mean_color_rgb': mean_color.tolist(),
                'color_std_rgb': std_color.tolist(),
                'edge_density': float(edge_density),
                'noise_level': float(noise_level),
                'blur_level': float(blur_level)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing image features: {e}")
            return None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict fake/real images')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--image_path', type=str,
                       help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str,
                       help='Directory containing images for batch prediction')
    parser.add_argument('--output_file', type=str, default='predictions.json',
                       help='Output file for predictions')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size for preprocessing')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of predictions')
    
    return parser.parse_args()

def main():
    """Main prediction function"""
    args = parse_arguments()
    
    # Initialize predictor
    predictor = FakeImagePredictor(
        model_path=args.model_path,
        img_size=(args.img_size, args.img_size)
    )
    
    results = []
    
    if args.image_path:
        # Single image prediction
        print(f"Predicting single image: {args.image_path}")
        
        result = predictor.predict_single_image(args.image_path)
        analysis = predictor.analyze_image_features(args.image_path)
        
        result['image_path'] = args.image_path
        result['feature_analysis'] = analysis
        
        results.append(result)
        
        # Print results
        print(f"\nPrediction Results:")
        print(f"Image: {args.image_path}")
        print(f"Predicted Class: {result['predicted_class'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Fake Probability: {result['fake_probability']:.4f}")
        print(f"Real Probability: {result['real_probability']:.4f}")
        
        if analysis:
            print(f"\nImage Analysis:")
            print(f"Dimensions: {analysis['image_dimensions']}")
            print(f"Edge Density: {analysis['edge_density']:.4f}")
            print(f"Noise Level: {analysis['noise_level']:.2f}")
            print(f"Blur Level: {analysis['blur_level']:.2f}")
    
    elif args.image_dir:
        # Batch prediction
        print(f"Predicting images in directory: {args.image_dir}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for filename in os.listdir(args.image_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(args.image_dir, filename))
        
        print(f"Found {len(image_paths)} images")
        
        # Predict all images
        results = predictor.predict_batch(image_paths)
        
        # Print summary
        fake_count = sum(1 for r in results if r.get('is_fake', False))
        real_count = len(results) - fake_count
        
        print(f"\nBatch Prediction Summary:")
        print(f"Total Images: {len(results)}")
        print(f"Predicted Fake: {fake_count}")
        print(f"Predicted Real: {real_count}")
        print(f"Fake Percentage: {fake_count/len(results)*100:.2f}%")
    
    else:
        print("Please provide either --image_path or --image_dir")
        return
    
    # Save results to JSON
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {args.output_file}")
    
    # Create visualization if requested
    if args.visualize and args.image_path:
        visualize_prediction(args.image_path, results[0])

if __name__ == "__main__":
    main()