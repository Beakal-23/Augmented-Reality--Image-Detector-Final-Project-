#!/usr/bin/env python3
"""
Upload QuickDraw model to Hugging Face Hub.
Requires: pip install huggingface_hub
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import json

# Configuration
REPO_NAME = "quickdraw-sketch-recognition"  # Change this to your preferred name
USERNAME = None  # Will prompt if not set
MODEL_DIR = "saved_models"
README_PATH = "MODEL_CARD.md"


def create_model_card():
    """Create a comprehensive model card for Hugging Face."""
    model_card = """---
license: mit
tags:
- image-classification
- sketch-recognition
- quickdraw
- vr
- ar
- drawing
- cnn
datasets:
- quickdraw
language:
- en
library_name: tensorflow
pipeline_tag: image-classification
---

# QuickDraw Sketch Recognition Model

A CNN-based sketch recognition model trained on Google's QuickDraw dataset for VR/AR applications.

## Model Description

This model recognizes hand-drawn sketches across 46 different classes. It's specifically designed for real-time sketch recognition in Virtual Reality (VR) and Augmented Reality (AR) applications where users draw with controllers or hand tracking.

**Model Type:** Convolutional Neural Network (CNN)  
**Framework:** TensorFlow/Keras  
**Input:** 28x28 grayscale images  
**Output:** 46 classes with confidence scores  
**Validation Accuracy:** 84.89%

## Supported Classes

The model can recognize the following 46 classes:

### Animals (7)
cat, dog, bird, fish, bear, butterfly, spider

### Buildings & Structures (6)
house, castle, barn, bridge, lighthouse, church

### Transportation (5)
car, airplane, bicycle, truck, train

### Nature (6)
tree, flower, sun, moon, cloud, mountain

### Common Objects (7)
apple, banana, book, chair, table, cup, umbrella

### Body Parts (4)
face, eye, hand, foot

### Shapes (4)
circle, triangle, square, star

### Tools & Items (5)
sword, axe, hammer, key, crown

### Musical Instruments (2)
guitar, piano

## Model Architecture

```
Sequential Model:
├── Conv2D(32, 3x3, ReLU) + MaxPooling2D(2x2)
├── Conv2D(64, 3x3, ReLU) + MaxPooling2D(2x2)
├── Conv2D(128, 3x3, ReLU)
├── Flatten
├── Dense(128, ReLU)
├── Dropout(0.3)
└── Dense(46, Softmax)
```

**Total Parameters:** ~500K  
**Training Time:** ~15-20 minutes on CPU  
**Inference Time:** <100ms per image

## Training Details

- **Dataset:** Google QuickDraw (5000 samples per class)
- **Total Training Samples:** 230,000
- **Validation Split:** 20%
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Early Stopping:** Patience of 5 epochs
- **Data Augmentation:** None (QuickDraw pre-normalized)
- **Best Epoch:** 10/20

## Usage

### Python/TensorFlow

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('quickdraw_house_cat_dog_car.keras')

# Prepare image (28x28 grayscale)
img = Image.open('drawing.png').convert('L').resize((28, 28))
img_array = np.array(img) / 255.0
img_array = img_array.reshape(1, 28, 28, 1)

# Predict
predictions = model.predict(img_array)
class_names = ['cat', 'dog', 'bird', ...]  # Full list in model files

top_3 = np.argsort(predictions[0])[-3:][::-1]
for idx in top_3:
    print(f"{class_names[idx]}: {predictions[0][idx]*100:.2f}%")
```

### FastAPI Server

A complete FastAPI server is available in the repository for production deployment:

```bash
# Using Docker
docker compose up -d quickdraw-api

# Access at http://localhost:8001
```

API endpoints:
- `POST /predict/base64` - Predict from base64 encoded image
- `POST /predict` - Predict from uploaded file
- `GET /classes` - List all supported classes
- `GET /health` - Health check

### Unity/VR Integration

C# integration example for Unity:

```csharp
using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class QuickDrawAPI : MonoBehaviour
{
    private string apiUrl = "http://your-server:8001/predict/base64";
    
    public IEnumerator RecognizeDrawing(Texture2D drawing)
    {
        byte[] imageBytes = drawing.EncodeToPNG();
        string base64 = System.Convert.ToBase64String(imageBytes);
        
        string json = JsonUtility.ToJson(new ImageData { 
            image_base64 = base64, 
            top_k = 3 
        });
        
        using (UnityWebRequest request = UnityWebRequest.Post(apiUrl, json, "application/json"))
        {
            yield return request.SendWebRequest();
            
            if (request.result == UnityWebRequest.Result.Success)
            {
                var response = JsonUtility.FromJson<PredictionResponse>(request.downloadHandler.text);
                Debug.Log($"Top prediction: {response.predictions[0].class_name}");
            }
        }
    }
}
```

## Model Files

This repository contains three formats:

1. **quickdraw_house_cat_dog_car.keras** - Native Keras format (recommended)
2. **quickdraw_house_cat_dog_car.h5** - Legacy HDF5 format
3. **quickdraw_house_cat_dog_car.onnx** - ONNX format for cross-platform inference

## Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | 84.89% |
| Top-3 Accuracy | ~95% |
| Average Inference Time (CPU) | 50-80ms |
| Average Inference Time (GPU) | 10-20ms |
| Model Size | 2.9 MB (Keras) |

## Limitations

- Works best with centered, single-object drawings
- Performance may vary with highly abstract or ambiguous sketches
- Trained on simplified 28x28 images (QuickDraw format)
- Some classes may be confused due to visual similarity (e.g., house vs barn)

## Use Cases

✅ **Ideal for:**
- VR/AR drawing applications
- Educational games
- Quick sketch recognition
- Real-time drawing feedback
- Art/creativity tools

❌ **Not suitable for:**
- Photo-realistic image classification
- Complex scene understanding
- Multi-object detection
- Fine-grained sketch details

## Citation

If you use this model, please cite the QuickDraw dataset:

```bibtex
@article{quickdraw,
  title={The Quick, Draw! Dataset},
  author={Ha, David and Eck, Douglas},
  journal={https://github.com/googlecreativelab/quickdraw-dataset},
  year={2017}
}
```

## License

MIT License - Free for commercial and non-commercial use.

## Repository

Full code, training scripts, and deployment tools: [GitHub Repository](https://github.com/Beakal-23/Augmented-Reality--Image-Detector-Final-Project-)

## Contact

For questions or issues, please open an issue on the GitHub repository.
"""
    
    with open(README_PATH, 'w') as f:
        f.write(model_card)
    
    print(f"✓ Model card created: {README_PATH}")


def upload_to_hf(username=None, repo_name=REPO_NAME, token=None):
    """Upload model to Hugging Face Hub."""
    
    # Initialize API
    api = HfApi()
    
    # Get username if not provided
    if username is None:
        username = input("Enter your Hugging Face username: ").strip()
    
    # Get token if not provided
    if token is None:
        print("\nTo get your Hugging Face token:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'write' permission")
        print("3. Copy and paste it below\n")
        token = input("Enter your Hugging Face token: ").strip()
    
    repo_id = f"{username}/{repo_name}"
    
    try:
        # Create repository
        print(f"\nCreating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="model",
            exist_ok=True
        )
        print(f"✓ Repository created/verified: https://huggingface.co/{repo_id}")
        
        # Upload README
        print("\nUploading model card...")
        upload_file(
            path_or_fileobj=README_PATH,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token
        )
        print("✓ Model card uploaded")
        
        # Upload model files
        print("\nUploading model files...")
        model_files = [
            "quickdraw_house_cat_dog_car.keras",
            "quickdraw_house_cat_dog_car.h5",
            "quickdraw_house_cat_dog_car.onnx"
        ]
        
        for model_file in model_files:
            file_path = os.path.join(MODEL_DIR, model_file)
            if os.path.exists(file_path):
                print(f"  Uploading {model_file}...")
                upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=model_file,
                    repo_id=repo_id,
                    token=token
                )
                print(f"  ✓ {model_file} uploaded")
            else:
                print(f"  ⚠ {model_file} not found, skipping")
        
        # Create and upload config
        config = {
            "model_type": "cnn",
            "framework": "tensorflow",
            "task": "image-classification",
            "input_shape": [28, 28, 1],
            "num_classes": 46,
            "classes": [
                "cat", "dog", "bird", "fish", "bear", "butterfly", "spider",
                "house", "castle", "barn", "bridge", "lighthouse", "church",
                "car", "airplane", "bicycle", "truck", "train",
                "tree", "flower", "sun", "moon", "cloud", "mountain",
                "apple", "banana", "book", "chair", "table", "cup", "umbrella",
                "face", "eye", "hand", "foot",
                "circle", "triangle", "square", "star",
                "sword", "axe", "hammer", "key", "crown",
                "guitar", "piano"
            ]
        }
        
        config_path = "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\nUploading model config...")
        upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=repo_id,
            token=token
        )
        print("✓ Config uploaded")
        
        print(f"\n{'='*80}")
        print(f"✓ SUCCESS! Model uploaded to Hugging Face")
        print(f"{'='*80}")
        print(f"\nModel URL: https://huggingface.co/{repo_id}")
        print(f"\nTo use the model:")
        print(f"  from huggingface_hub import hf_hub_download")
        print(f"  model_path = hf_hub_download(repo_id='{repo_id}', filename='quickdraw_house_cat_dog_car.keras')")
        print(f"  model = tf.keras.models.load_model(model_path)")
        print()
        
        return repo_id
        
    except Exception as e:
        print(f"\n✗ Error uploading to Hugging Face: {e}")
        return None


if __name__ == "__main__":
    print("="*80)
    print("QuickDraw Model - Hugging Face Upload")
    print("="*80)
    print()
    
    # Check if model files exist
    model_files_exist = any(
        os.path.exists(os.path.join(MODEL_DIR, f)) 
        for f in ["quickdraw_house_cat_dog_car.keras", "quickdraw_house_cat_dog_car.h5"]
    )
    
    if not model_files_exist:
        print("✗ Error: Model files not found in saved_models/")
        print("Please train the model first using: docker compose --profile training up")
        exit(1)
    
    # Create model card
    create_model_card()
    
    # Upload to Hugging Face
    print("\nReady to upload to Hugging Face Hub")
    proceed = input("Continue? (y/n): ").strip().lower()
    
    if proceed == 'y':
        upload_to_hf()
    else:
        print("Upload cancelled. Model card saved to MODEL_CARD.md")
