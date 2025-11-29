"""
Utility functions for image preprocessing.
Handles various input formats: bytes, base64, PIL images, etc.
"""
import io
import base64
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image from raw bytes.
    
    Args:
        image_bytes: Raw image bytes (PNG, JPG, etc.)
    
    Returns:
        Preprocessed numpy array of shape (1, 28, 28, 1) normalized to [0, 1]
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        image_array = image_array / 255.0
        
        # Reshape to (1, 28, 28, 1) for model input
        image_array = image_array.reshape(1, 28, 28, 1)
        
        return image_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image from bytes: {e}")
        raise ValueError(f"Failed to process image: {str(e)}")


def preprocess_image_from_base64(base64_string: str) -> np.ndarray:
    """
    Preprocess image from base64 encoded string.
    
    Args:
        base64_string: Base64 encoded image string (with or without data URI prefix)
    
    Returns:
        Preprocessed numpy array of shape (1, 28, 28, 1) normalized to [0, 1]
    """
    try:
        # Remove data URI prefix if present (e.g., "data:image/png;base64,")
        if ',' in base64_string and base64_string.startswith('data:'):
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Use the bytes preprocessing function
        return preprocess_image_from_bytes(image_bytes)
    
    except Exception as e:
        logger.error(f"Error preprocessing image from base64: {e}")
        raise ValueError(f"Failed to process base64 image: {str(e)}")


def preprocess_image_from_array(image_array: np.ndarray) -> np.ndarray:
    """
    Preprocess image from numpy array.
    Handles various input shapes and formats.
    
    Args:
        image_array: Numpy array representing an image
    
    Returns:
        Preprocessed numpy array of shape (1, 28, 28, 1) normalized to [0, 1]
    """
    try:
        # Convert to float32
        image_array = image_array.astype(np.float32)
        
        # Handle different input shapes
        if len(image_array.shape) == 4:  # (batch, height, width, channels)
            # Take first image if batch
            image_array = image_array[0]
        
        if len(image_array.shape) == 3:  # (height, width, channels)
            # If RGB, convert to grayscale
            if image_array.shape[2] == 3:
                # Simple RGB to grayscale conversion
                image_array = 0.299 * image_array[:, :, 0] + \
                             0.587 * image_array[:, :, 1] + \
                             0.114 * image_array[:, :, 2]
            elif image_array.shape[2] == 1:
                image_array = image_array.squeeze(-1)
        
        # Now image_array should be 2D (height, width)
        if len(image_array.shape) != 2:
            raise ValueError(f"Cannot process image with shape {image_array.shape}")
        
        # Resize if needed
        if image_array.shape != (28, 28):
            image_pil = Image.fromarray(image_array.astype(np.uint8))
            image_pil = image_pil.resize((28, 28), Image.Resampling.LANCZOS)
            image_array = np.array(image_pil, dtype=np.float32)
        
        # Normalize to [0, 1] if not already
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        # Reshape to (1, 28, 28, 1)
        image_array = image_array.reshape(1, 28, 28, 1)
        
        return image_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image from array: {e}")
        raise ValueError(f"Failed to process image array: {str(e)}")


def preprocess_stroke_data(strokes: list, canvas_size: int = 256) -> np.ndarray:
    """
    Convert stroke data (list of coordinates) to a 28x28 image.
    Useful if VR application sends raw drawing coordinates.
    
    Args:
        strokes: List of strokes, where each stroke is a list of (x, y) coordinates
                Example: [[(x1, y1), (x2, y2), ...], [(x3, y3), ...]]
        canvas_size: Size of the virtual canvas (default: 256x256)
    
    Returns:
        Preprocessed numpy array of shape (1, 28, 28, 1) normalized to [0, 1]
    """
    try:
        # Create a blank canvas
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        
        # Draw strokes on canvas
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            
            # Draw lines between consecutive points
            for i in range(len(stroke) - 1):
                x1, y1 = stroke[i]
                x2, y2 = stroke[i + 1]
                
                # Simple line drawing (Bresenham's algorithm would be better)
                # For now, use a simple approximation
                points = _interpolate_points(x1, y1, x2, y2)
                for x, y in points:
                    if 0 <= x < canvas_size and 0 <= y < canvas_size:
                        canvas[int(y), int(x)] = 255
        
        # Convert canvas to PIL Image for resizing
        image = Image.fromarray(canvas)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Reshape to (1, 28, 28, 1)
        image_array = image_array.reshape(1, 28, 28, 1)
        
        return image_array
    
    except Exception as e:
        logger.error(f"Error preprocessing stroke data: {e}")
        raise ValueError(f"Failed to process stroke data: {str(e)}")


def _interpolate_points(x1: float, y1: float, x2: float, y2: float, num_points: int = 10) -> list:
    """
    Interpolate points between two coordinates for smooth line drawing.
    
    Args:
        x1, y1: Start coordinates
        x2, y2: End coordinates
        num_points: Number of points to interpolate
    
    Returns:
        List of (x, y) coordinate tuples
    """
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        points.append((x, y))
    return points
