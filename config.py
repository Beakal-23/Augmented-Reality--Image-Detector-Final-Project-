"""
Configuration settings for the QuickDraw API.
Modify these settings based on your deployment needs.
"""
import os
from typing import List


class Settings:
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "QuickDraw Sketch Recognition API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "API for recognizing hand-drawn sketches for VR/AR applications"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False  # Set to True for development
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]  # In production, specify allowed origins
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Model Settings
    MODEL_PATH: str = os.path.join("saved_models", "quickdraw_house_cat_dog_car.keras")
    CLASS_NAMES: List[str] = [
        # Animals (7)
        "cat", "dog", "bird", "fish", "bear", "butterfly", "spider",
        # Buildings & Structures (6)
        "house", "castle", "barn", "bridge", "lighthouse", "church",
        # Transportation (5)
        "car", "airplane", "bicycle", "truck", "train",
        # Nature (6)
        "tree", "flower", "sun", "moon", "cloud", "mountain",
        # Common Objects (7)
        "apple", "banana", "book", "chair", "table", "cup", "umbrella",
        # People & Body (4)
        "face", "eye", "hand", "foot",
        # Shapes (4)
        "circle", "triangle", "square", "star",
        # Tools & Items (5)
        "sword", "axe", "hammer", "key", "crown",
        # Musical Instruments (2)
        "guitar", "piano"
    ]
    
    # Prediction Settings
    DEFAULT_TOP_K: int = 3
    CONFIDENCE_THRESHOLD: float = 0.5  # Minimum confidence for valid predictions
    
    # Image Processing Settings
    INPUT_IMAGE_SIZE: tuple = (28, 28)
    GRAYSCALE: bool = True
    NORMALIZE: bool = True  # Normalize pixel values to [0, 1]
    
    # Logging
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL


# Create a singleton instance
settings = Settings()
