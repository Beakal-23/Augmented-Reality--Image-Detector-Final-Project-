# QuickDraw Sketch Recognition API for VR/AR

A FastAPI-based REST API for real-time sketch recognition, designed for integration with VR/AR applications. The API uses a CNN model trained on Google's QuickDraw dataset to classify hand-drawn sketches of houses, cats, dogs, and cars.

## 🎯 Project Overview

This API is specifically designed for VR applications where users draw sketches with a virtual brush. The drawings are sent to the API for classification, enabling interactive experiences where the system recognizes what users are drawing in real-time.

### Supported Classes

The model recognizes **55 common drawing objects**:

**Animals**: cat, dog, bird, fish, bear, butterfly, bee, spider  
**Buildings**: house, castle, barn, bridge, lighthouse, church  
**Transportation**: car, airplane, bicycle, boat, train, truck, bus  
**Nature**: tree, flower, sun, moon, cloud, mountain, river  
**Objects**: apple, banana, book, chair, table, cup, umbrella  
**Body Parts**: face, eye, hand, foot  
**Shapes**: circle, triangle, square, star, heart  
**Tools**: sword, axe, hammer, key, crown

_You can easily add more classes by editing `Model-Training.py`. Google's QuickDraw has 345+ classes available!_

## 🚀 Quick Start

### 🐳 Docker (Recommended) - Works on Windows, macOS & Linux

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

**Windows:**

```cmd
# Train model (first time, takes 10-30 min)
train_docker.bat

# Start API
start_api_docker.bat
```

**macOS/Linux:**

```bash
# Train model (first time, takes 10-30 min)
./train_docker.sh

# Start API
./start_api_docker.sh
```

**Or manually:**

```bash
# Build and train
docker-compose --profile training up quickdraw-training

# Start API
docker-compose up -d

# Check status
curl http://localhost:8000/health
```

### 🐍 Local Python Installation (Alternative)

**Prerequisites:** Python 3.10+

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (first time, 10-30 min)
python Model-Training.py

# Start API
python main.py
```

The API will be available at `http://localhost:8000`

## 📡 API Endpoints

### 1. Root Endpoint

```
GET /
```

Returns API information and available endpoints.

**Response:**

```json
{
  "message": "QuickDraw Sketch Recognition API",
  "version": "1.0.0",
  "endpoints": {
    "/health": "Health check",
    "/predict": "Predict from uploaded image file (POST)",
    "/predict/base64": "Predict from base64 encoded image (POST)",
    "/classes": "Get list of supported classes (GET)"
  }
}
```

### 2. Health Check

```
GET /health
```

Check if the API and model are ready.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Get Supported Classes

```
GET /classes
```

Returns the list of drawing classes the model can recognize.

**Response:**

```json
{
  "classes": ["house", "cat", "dog", "car"],
  "num_classes": 4
}
```

### 4. Predict from Image File

```
POST /predict
```

Upload an image file for classification.

**Parameters:**

- `file`: Image file (PNG, JPG, etc.)
- `top_k`: (optional) Number of top predictions to return (default: 3)

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/predict?top_k=3" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@drawing.png"
```

**Response:**

```json
{
  "predictions": [
    {
      "class": "cat",
      "confidence": 0.8542,
      "confidence_percent": "85.42%"
    },
    {
      "class": "dog",
      "confidence": 0.1234,
      "confidence_percent": "12.34%"
    },
    {
      "class": "house",
      "confidence": 0.0224,
      "confidence_percent": "2.24%"
    }
  ],
  "success": true,
  "message": "Prediction successful"
}
```

### 5. Predict from Base64 Image (Recommended for VR)

```
POST /predict/base64
```

Send a base64-encoded image for classification. **This is the recommended endpoint for VR/AR applications.**

**Request Body:**

```json
{
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "top_k": 3
}
```

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/predict/base64" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANS...",
    "top_k": 3
  }'
```

**Response:**

```json
{
  "predictions": [
    {
      "class": "house",
      "confidence": 0.9123,
      "confidence_percent": "91.23%"
    },
    {
      "class": "car",
      "confidence": 0.0567,
      "confidence_percent": "5.67%"
    },
    {
      "class": "cat",
      "confidence": 0.031,
      "confidence_percent": "3.10%"
    }
  ],
  "success": true,
  "message": "Prediction successful"
}
```

## 🎮 Integration with VR/AR (C# Unity)

Here's an example of how to call the API from C# in Unity:

```csharp
using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

[Serializable]
public class PredictionRequest
{
    public string image_base64;
    public int top_k = 3;
}

[Serializable]
public class PredictionResult
{
    public string @class;
    public float confidence;
    public string confidence_percent;
}

[Serializable]
public class PredictionResponse
{
    public PredictionResult[] predictions;
    public bool success;
    public string message;
}

public class SketchRecognition : MonoBehaviour
{
    private const string API_URL = "http://localhost:8000/predict/base64";

    public IEnumerator RecognizeDrawing(Texture2D drawingTexture)
    {
        // Convert texture to PNG bytes
        byte[] imageBytes = drawingTexture.EncodeToPNG();

        // Convert to base64
        string base64Image = "data:image/png;base64," + Convert.ToBase64String(imageBytes);

        // Create request
        PredictionRequest request = new PredictionRequest
        {
            image_base64 = base64Image,
            top_k = 3
        };

        string jsonData = JsonUtility.ToJson(request);

        // Send request
        using (UnityWebRequest www = UnityWebRequest.Post(API_URL, ""))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                PredictionResponse response = JsonUtility.FromJson<PredictionResponse>(www.downloadHandler.text);

                Debug.Log($"Top prediction: {response.predictions[0].@class} " +
                         $"({response.predictions[0].confidence_percent})");

                // Handle the prediction result
                OnPredictionReceived(response);
            }
            else
            {
                Debug.LogError($"API Error: {www.error}");
            }
        }
    }

    private void OnPredictionReceived(PredictionResponse response)
    {
        // Your logic here - e.g., spawn the recognized object
        string recognizedClass = response.predictions[0].@class;
        float confidence = response.predictions[0].confidence;

        if (confidence > 0.7f) // 70% confidence threshold
        {
            Debug.Log($"Recognized: {recognizedClass}");
            // Spawn or trigger appropriate VR action
        }
    }
}
```

## 📁 Project Structure

```
Augmented-Reality--Image-Detector-Final-Project-/
├── main.py                 # FastAPI application
├── model.py               # Model loading and inference logic
├── utils.py               # Image preprocessing utilities
├── Model-Training.py      # Script to train the CNN model
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── saved_models/         # Directory for trained models
│   └── quickdraw_house_cat_dog_car.keras
└── quickdraw_npy/        # Downloaded dataset (created during training)
```

## 🛠️ Development

### Interactive API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Docker Commands

```bash
# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Stop
docker-compose down

# Rebuild after changes
docker-compose up -d --build
```

### Running Locally

```bash
# Development mode with auto-reload
uvicorn main:app --reload

# Different port
uvicorn main:app --port 8080
```

### Training with More Classes

To add more classes, edit `Model-Training.py`:

```python
CLASS_NAMES = ["house", "cat", "dog", "car", "tree", "sun", "flower"]
```

Then retrain the model:

```bash
python Model-Training.py
```

### Adjusting Training Parameters

In `Model-Training.py`, you can modify:

- `MAX_ITEMS_PER_CLASS`: Number of samples per class (default: 2000)
- `epochs`: Training epochs (default: 20)
- `batch_size`: Batch size (default: 128)

## 🔧 Configuration

### CORS Settings

By default, the API accepts requests from any origin (`allow_origins=["*"]`). For production, update `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://your-vr-app-domain.com"],  # Specify your VR app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Model Path

To use a different model path, modify `model.py`:

```python
classifier = SketchClassifier(model_path="path/to/your/model.keras")
```

## 📊 Model Performance

The CNN model achieves approximately:

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~90%
- **Inference Time**: ~20-50ms per image

Note: Actual performance depends on training parameters and hardware.

## 🐛 Troubleshooting

### Model Not Found

**Solution**: Train the model first

```bash
# Docker
./train_docker.sh  # or train_docker.bat on Windows

# Local
python Model-Training.py
```

### Port Already in Use

**Solution**: Change port in `docker-compose.yml` or stop conflicting process

```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Connection Refused from Unity

1. Check API is running: `curl http://localhost:8000/health`
2. If Unity on different machine, use your IP: `http://192.168.1.x:8000`
3. Check firewall allows port 8000

### Docker Not Starting

1. Ensure Docker Desktop is running
2. Check: `docker info`
3. Restart Docker Desktop

## 📝 License

This project uses Google's QuickDraw dataset, which is available under the Creative Commons Attribution 4.0 International license.

## 🙏 Acknowledgments

- Google Creative Lab for the [QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset)
- FastAPI framework
- TensorFlow/Keras team

## 📧 Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Happy Drawing! 🎨🖌️**
