"""
Model inference module for QuickDraw sketch classification.
Handles model loading and prediction logic.
"""
import os
import numpy as np
import tensorflow as tf
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class SketchClassifier:
    """QuickDraw sketch classifier"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the classifier with a trained model.
        
        Args:
            model_path: Path to the trained model file. If None, uses default path.
        """
        # Extended class list matching Model-Training.py
        self.class_names = [
            # Animals
            "cat", "dog", "bird", "fish", "bear", "butterfly", "bee", "spider",
            # Buildings & Structures
            "house", "castle", "barn", "bridge", "lighthouse", "church",
            # Transportation
            "car", "airplane", "bicycle", "boat", "train", "truck", "bus",
            # Nature
            "tree", "flower", "sun", "moon", "cloud", "mountain", "river",
            # Common Objects
            "apple", "banana", "book", "chair", "table", "cup", "umbrella",
            # People & Body
            "face", "eye", "hand", "foot",
            # Shapes & Symbols
            "circle", "triangle", "square", "star", "heart",
            # Tools & Items
            "sword", "axe", "hammer", "key", "crown"
        ]
        
        # Default model path
        if model_path is None:
            model_path = os.path.join("saved_models", "quickdraw_house_cat_dog_car.keras")
        
        # Check if model exists
        if not os.path.exists(model_path):
            # Try .h5 format as fallback
            h5_path = model_path.replace(".keras", ".h5")
            if os.path.exists(h5_path):
                model_path = h5_path
                logger.info(f"Using H5 model format: {model_path}")
            else:
                raise FileNotFoundError(
                    f"Model file not found at {model_path}. "
                    "Please train the model first using Model-Training.py"
                )
        
        logger.info(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully!")
        
        # Verify input shape
        self.input_shape = self.model.input_shape[1:]  # (28, 28, 1)
        logger.info(f"Model input shape: {self.input_shape}")
    
    def predict(self, image: np.ndarray, top_k: int = 3) -> List[Dict[str, any]]:
        """
        Make prediction on a preprocessed image.
        
        Args:
            image: Preprocessed image array of shape (1, 28, 28, 1)
            top_k: Number of top predictions to return
        
        Returns:
            List of dictionaries containing class names and confidence scores
        """
        # Validate input shape
        if image.shape != (1, 28, 28, 1):
            raise ValueError(
                f"Expected input shape (1, 28, 28, 1), got {image.shape}. "
                "Please preprocess the image first."
            )
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        
        # Get top k predictions
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "class": self.class_names[idx],
                "confidence": float(predictions[0][idx]),
                "confidence_percent": f"{predictions[0][idx] * 100:.2f}%"
            })
        
        return results
    
    def predict_batch(self, images: np.ndarray, top_k: int = 3) -> List[List[Dict[str, any]]]:
        """
        Make predictions on a batch of preprocessed images.
        
        Args:
            images: Batch of preprocessed images of shape (N, 28, 28, 1)
            top_k: Number of top predictions to return per image
        
        Returns:
            List of prediction results for each image
        """
        # Make predictions
        predictions = self.model.predict(images, verbose=0)
        
        results = []
        for pred in predictions:
            # Get top k predictions for this image
            top_indices = np.argsort(pred)[::-1][:top_k]
            
            image_results = []
            for idx in top_indices:
                image_results.append({
                    "class": self.class_names[idx],
                    "confidence": float(pred[idx]),
                    "confidence_percent": f"{pred[idx] * 100:.2f}%"
                })
            
            results.append(image_results)
        
        return results
