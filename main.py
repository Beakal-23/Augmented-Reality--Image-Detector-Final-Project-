"""
FastAPI application for QuickDraw sketch recognition.
Exposes API endpoints for VR/AR applications to classify drawings.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
import os
import base64
from datetime import datetime
from pathlib import Path
import json

from model import SketchClassifier
from utils import preprocess_image_from_bytes, preprocess_image_from_base64

# Configure comprehensive logging
LOG_DIR = "api_logs"
IMAGES_LOG_DIR = os.path.join(LOG_DIR, "received_images")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMAGES_LOG_DIR, exist_ok=True)

# Setup logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create separate logger for request details
request_logger = logging.getLogger("requests")
request_handler = logging.FileHandler(os.path.join(LOG_DIR, 'requests_detailed.log'))
request_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
request_logger.addHandler(request_handler)
request_logger.setLevel(logging.INFO)

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
    top_k: int = 3,
    http_request: Request = None
):
    """
    Predict drawing class from uploaded image file.
    
    Args:
        file: Image file (PNG, JPG, etc.)
        top_k: Number of top predictions to return (default: 3)
        http_request: FastAPI request object for logging
    
    Returns:
        PredictionResponse with top predictions and confidence scores
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate unique request ID
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    logger.info(f"="*80)
    logger.info(f"[FILE-REQUEST {request_id}] New file upload prediction")
    logger.info(f"[FILE-REQUEST {request_id}] Filename: {file.filename}")
    logger.info(f"[FILE-REQUEST {request_id}] Content-Type: {file.content_type}")
    logger.info(f"[FILE-REQUEST {request_id}] Top K: {top_k}")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        logger.info(f"[FILE-REQUEST {request_id}] File size: {len(image_bytes)} bytes")
        
        # Save uploaded file
        uploaded_file = os.path.join(IMAGES_LOG_DIR, f"uploaded_{request_id}_{file.filename}")
        with open(uploaded_file, 'wb') as f:
            f.write(image_bytes)
        logger.info(f"[FILE-REQUEST {request_id}] File saved to: {uploaded_file}")
        
        # Preprocess image
        logger.info(f"[FILE-REQUEST {request_id}] Preprocessing image...")
        processed_image = preprocess_image_from_bytes(image_bytes)
        logger.info(f"[FILE-REQUEST {request_id}] Preprocessed shape: {processed_image.shape}")
        
        # Make prediction
        logger.info(f"[FILE-REQUEST {request_id}] Running inference...")
        predictions = classifier.predict(processed_image, top_k=top_k)
        
        # Log predictions
        logger.info(f"[FILE-REQUEST {request_id}] PREDICTIONS:")
        for i, pred in enumerate(predictions, 1):
            logger.info(f"[FILE-REQUEST {request_id}]   {i}. {pred['class']}: {pred['confidence_percent']}")
        
        logger.info(f"[FILE-REQUEST {request_id}] ✓ Success")
        logger.info(f"="*80)
        
        return PredictionResponse(
            predictions=predictions,
            success=True,
            message=f"Prediction successful (Request ID: {request_id})"
        )
    
    except Exception as e:
        logger.error(f"[FILE-REQUEST {request_id}] ✗ FAILED: {e}")
        logger.info(f"="*80)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_from_base64(request: PredictionRequest, http_request: Request):
    """
    Predict drawing class from base64 encoded image.
    Ideal for VR/AR applications sending image data directly.
    
    Args:
        request: PredictionRequest containing base64 image and optional top_k
        http_request: FastAPI request object for logging
    
    Returns:
        PredictionResponse with top predictions and confidence scores
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate unique request ID
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Log incoming request details
    logger.info(f"="*80)
    logger.info(f"[REQUEST {request_id}] New prediction request from VR")
    logger.info(f"[REQUEST {request_id}] Client: {http_request.client.host}:{http_request.client.port}")
    logger.info(f"[REQUEST {request_id}] User-Agent: {http_request.headers.get('user-agent', 'Unknown')}")
    logger.info(f"[REQUEST {request_id}] Top K: {request.top_k}")
    
    # Log base64 image details
    base64_length = len(request.image_base64)
    logger.info(f"[REQUEST {request_id}] Base64 image length: {base64_length} characters")
    logger.info(f"[REQUEST {request_id}] Base64 prefix (first 100 chars): {request.image_base64[:100]}...")
    
    # Save base64 string to file for debugging
    base64_log_file = os.path.join(LOG_DIR, f"request_{request_id}_base64.txt")
    with open(base64_log_file, 'w') as f:
        f.write(request.image_base64)
    logger.info(f"[REQUEST {request_id}] Base64 saved to: {base64_log_file}")
    
    try:
        # Decode and save the actual image
        try:
            image_data = base64.b64decode(request.image_base64)
            image_file = os.path.join(IMAGES_LOG_DIR, f"request_{request_id}.png")
            with open(image_file, 'wb') as f:
                f.write(image_data)
            logger.info(f"[REQUEST {request_id}] Decoded image saved to: {image_file}")
            logger.info(f"[REQUEST {request_id}] Decoded image size: {len(image_data)} bytes")
        except Exception as decode_error:
            logger.warning(f"[REQUEST {request_id}] Failed to decode/save image: {decode_error}")
        
        # Preprocess image from base64
        logger.info(f"[REQUEST {request_id}] Preprocessing image...")
        processed_image = preprocess_image_from_base64(request.image_base64)
        logger.info(f"[REQUEST {request_id}] Preprocessed image shape: {processed_image.shape}")
        
        # Make prediction
        logger.info(f"[REQUEST {request_id}] Running model inference...")
        predictions = classifier.predict(processed_image, top_k=request.top_k)
        
        # Log predictions
        logger.info(f"[REQUEST {request_id}] PREDICTIONS:")
        for i, pred in enumerate(predictions, 1):
            logger.info(f"[REQUEST {request_id}]   {i}. {pred['class']}: {pred['confidence_percent']} (confidence: {pred['confidence']:.4f})")
        
        # Save detailed request log as JSON
        request_log = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "client_ip": http_request.client.host,
            "client_port": http_request.client.port,
            "user_agent": http_request.headers.get('user-agent', 'Unknown'),
            "base64_length": base64_length,
            "image_file": image_file if 'image_file' in locals() else None,
            "top_k": request.top_k,
            "predictions": predictions,
            "success": True
        }
        
        json_log_file = os.path.join(LOG_DIR, f"request_{request_id}.json")
        with open(json_log_file, 'w') as f:
            json.dump(request_log, f, indent=2)
        logger.info(f"[REQUEST {request_id}] Full request log saved to: {json_log_file}")
        
        logger.info(f"[REQUEST {request_id}] ✓ Prediction completed successfully")
        logger.info(f"="*80)
        
        return PredictionResponse(
            predictions=predictions,
            success=True,
            message=f"Prediction successful (Request ID: {request_id})"
        )
    
    except Exception as e:
        logger.error(f"[REQUEST {request_id}] ✗ Prediction FAILED")
        logger.error(f"[REQUEST {request_id}] Error: {str(e)}")
        logger.error(f"[REQUEST {request_id}] Error type: {type(e).__name__}")
        logger.info(f"="*80)
        
        # Save error log
        error_log = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "error_type": type(e).__name__,
            "base64_length": base64_length,
            "success": False
        }
        error_log_file = os.path.join(LOG_DIR, f"request_{request_id}_ERROR.json")
        with open(error_log_file, 'w') as f:
            json.dump(error_log, f, indent=2)
        
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
