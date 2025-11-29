"""
Test script for the QuickDraw Sketch Recognition API.
Run this to verify the API is working correctly.
"""
import requests
import base64
from pathlib import Path
import json

API_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("✓ Health check passed")


def test_get_classes():
    """Test get classes endpoint"""
    print("\n=== Testing Get Classes ===")
    response = requests.get(f"{API_URL}/classes")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Classes: {data['classes']}")
    print(f"Number of classes: {data['num_classes']}")
    assert response.status_code == 200
    assert len(data['classes']) == 4
    print("✓ Get classes passed")


def test_predict_base64():
    """Test prediction with base64 encoded image"""
    print("\n=== Testing Base64 Prediction ===")
    
    # Create a simple test image (28x28 white square)
    try:
        from PIL import Image
        import numpy as np
        import io
        
        # Create a simple drawing-like image
        img = Image.new('L', (28, 28), color=255)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Convert to base64
        base64_image = base64.b64encode(img_bytes.read()).decode('utf-8')
        base64_image = f"data:image/png;base64,{base64_image}"
        
        # Make request
        payload = {
            "image_base64": base64_image,
            "top_k": 3
        }
        
        response = requests.post(f"{API_URL}/predict/base64", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data['success']}")
            print(f"Predictions:")
            for pred in data['predictions']:
                print(f"  - {pred['class']}: {pred['confidence_percent']}")
            print("✓ Base64 prediction passed")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"⚠ Could not test base64 prediction: {e}")


def test_predict_file():
    """Test prediction with file upload"""
    print("\n=== Testing File Upload Prediction ===")
    
    try:
        from PIL import Image
        import tempfile
        
        # Create a temporary test image
        img = Image.new('L', (28, 28), color=255)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name
        
        # Upload file
        with open(tmp_path, 'rb') as f:
            files = {'file': ('test.png', f, 'image/png')}
            response = requests.post(f"{API_URL}/predict?top_k=3", files=files)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data['success']}")
            print(f"Predictions:")
            for pred in data['predictions']:
                print(f"  - {pred['class']}: {pred['confidence_percent']}")
            print("✓ File upload prediction passed")
        else:
            print(f"Error: {response.text}")
            
        # Cleanup
        Path(tmp_path).unlink()
        
    except Exception as e:
        print(f"⚠ Could not test file upload: {e}")


def main():
    """Run all tests"""
    print("=" * 50)
    print("QuickDraw API Test Suite")
    print("=" * 50)
    
    try:
        test_health()
        test_get_classes()
        test_predict_base64()
        test_predict_file()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed!")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("\n⚠ Error: Could not connect to API")
        print("Make sure the API is running: python main.py")
    except Exception as e:
        print(f"\n⚠ Error: {e}")


if __name__ == "__main__":
    main()
