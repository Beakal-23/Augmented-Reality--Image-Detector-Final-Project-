"""
FastAPI application for QuickDraw sketch recognition.
Exposes API endpoints for VR/AR applications to classify drawings.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

from model import SketchClassifier
from utils import preprocess_image_from_bytes, preprocess_image_from_base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="QuickDraw Sketch Recognition API",
    description="API for recognizing hand-drawn sketches (house, cat, dog, car) for VR/AR applications",
    version="1.0.0"
)

# CORS middleware - adjust origins based on your VR application needs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your VR app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model (singleton)
classifier = None


class PredictionRequest(BaseModel):
    """Request model for base64 encoded image"""
    image_base64: str
    top_k: Optional[int] = 3


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[dict]
    success: bool
    message: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    global classifier
    try:
        logger.info("Loading QuickDraw model...")
        classifier = SketchClassifier()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "QuickDraw Sketch Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Predict from uploaded image file (POST)",
            "/predict/base64": "Predict from base64 encoded image (POST)",
            "/classes": "Get list of supported classes (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = classifier is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }


@app.get("/classes")
async def get_classes():
    """Get list of supported drawing classes"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": classifier.class_names,
        "num_classes": len(classifier.class_names)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_from_file(
    file: UploadFile = File(...),
    top_k: int = 3
):
    """
    Predict drawing class from uploaded image file.
    
    Args:
        file: Image file (PNG, JPG, etc.)
        top_k: Number of top predictions to return (default: 3)
    
    Returns:
        PredictionResponse with top predictions and confidence scores
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image_from_bytes(image_bytes)
        
        # Make prediction
        predictions = classifier.predict(processed_image, top_k=top_k)
        
        return PredictionResponse(
            predictions=predictions,
            success=True,
            message="Prediction successful"
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_from_base64(request: PredictionRequest):
    """
    Predict drawing class from base64 encoded image.
    Ideal for VR/AR applications sending image data directly.
    
    Args:
        request: PredictionRequest containing base64 image and optional top_k
    
    Returns:
        PredictionResponse with top predictions and confidence scores
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess image from base64
        processed_image = preprocess_image_from_base64(request.image_base64)
        
        # Make prediction
        predictions = classifier.predict(processed_image, top_k=request.top_k)
        
        return PredictionResponse(
            predictions=predictions,
            success=True,
            message="Prediction successful"
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
