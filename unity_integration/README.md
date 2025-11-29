# Unity Integration Guide

## Overview

This folder contains Unity C# scripts for integrating the QuickDraw Sketch Recognition API with your VR/AR Unity project.

## Files

- `QuickDrawAPI.cs` - Main API integration script

## Installation

1. **Copy the script to your Unity project:**

   ```
   Copy QuickDrawAPI.cs to: Assets/Scripts/
   ```

2. **Attach to a GameObject:**

   - Create an empty GameObject in your scene (e.g., "QuickDrawManager")
   - Attach the `QuickDrawAPI` component to it

3. **Configure the component:**
   - **API URL**: Set to your API endpoint (default: `http://localhost:8001/predict/base64`)
   - **Top K**: Number of predictions to return (1-4)
   - **Confidence Threshold**: Minimum confidence to accept prediction (0-1)

## Usage Examples

### Example 1: Recognize Drawing from Texture

```csharp
public class DrawingPanel : MonoBehaviour
{
    private QuickDrawAPI quickDrawAPI;
    public Texture2D drawingTexture;

    void Start()
    {
        quickDrawAPI = FindObjectOfType<QuickDrawAPI>();
    }

    public void OnSubmitDrawing()
    {
        quickDrawAPI.RecognizeDrawing(drawingTexture);
    }
}
```

### Example 2: Recognize from RenderTexture (VR Drawing)

```csharp
public class VRDrawingCanvas : MonoBehaviour
{
    private QuickDrawAPI quickDrawAPI;
    public RenderTexture drawingRenderTexture;

    void Start()
    {
        quickDrawAPI = FindObjectOfType<QuickDrawAPI>();
    }

    public void OnVRButtonPressed()
    {
        // User finished drawing in VR
        quickDrawAPI.RecognizeDrawing(drawingRenderTexture);
    }
}
```

### Example 3: Using Events

```csharp
public class GameManager : MonoBehaviour
{
    private QuickDrawAPI quickDrawAPI;

    void Start()
    {
        quickDrawAPI = FindObjectOfType<QuickDrawAPI>();

        // Subscribe to prediction events
        quickDrawAPI.OnPredictionReceived.AddListener(HandlePrediction);
        quickDrawAPI.OnPredictionError.AddListener(HandleError);
    }

    void HandlePrediction(QuickDrawAPI.PredictionResponse response)
    {
        string topClass = response.predictions[0].@class;
        float confidence = response.predictions[0].confidence;

        Debug.Log($"Recognized: {topClass} with {confidence:P} confidence");

        // Spawn appropriate 3D object
        SpawnObject(topClass);
    }

    void HandleError(string error)
    {
        Debug.LogError($"Prediction error: {error}");
        // Show error UI to user
    }

    void SpawnObject(string className)
    {
        // Your logic to spawn recognized object
        GameObject prefab = Resources.Load<GameObject>($"Prefabs/{className}");
        if (prefab != null)
        {
            Instantiate(prefab, transform.position, Quaternion.identity);
        }
    }
}
```

### Example 4: Custom Drawing Recognition Handler

```csharp
public class ObjectSpawner : MonoBehaviour
{
    public GameObject housePrefab;
    public GameObject catPrefab;
    public GameObject dogPrefab;
    public GameObject carPrefab;

    private QuickDrawAPI quickDrawAPI;

    void Start()
    {
        quickDrawAPI = FindObjectOfType<QuickDrawAPI>();
        quickDrawAPI.OnPredictionReceived.AddListener(OnDrawingRecognized);
    }

    void OnDrawingRecognized(QuickDrawAPI.PredictionResponse response)
    {
        if (response.predictions.Length == 0) return;

        var topPrediction = response.predictions[0];

        // Only spawn if confidence is high enough
        if (topPrediction.confidence < 0.7f)
        {
            Debug.Log("Confidence too low, please draw again");
            return;
        }

        GameObject prefabToSpawn = null;

        switch (topPrediction.@class)
        {
            case "house":
                prefabToSpawn = housePrefab;
                break;
            case "cat":
                prefabToSpawn = catPrefab;
                break;
            case "dog":
                prefabToSpawn = dogPrefab;
                break;
            case "car":
                prefabToSpawn = carPrefab;
                break;
        }

        if (prefabToSpawn != null)
        {
            // Spawn in front of the camera/player
            Vector3 spawnPos = Camera.main.transform.position +
                             Camera.main.transform.forward * 2f;
            Instantiate(prefabToSpawn, spawnPos, Quaternion.identity);
        }
    }
}
```

## API Configuration

### Local Development

```csharp
apiUrl = "http://localhost:8000/predict/base64"
```

### Network/Production

```csharp
apiUrl = "http://your-server-ip:8000/predict/base64"
```

For production deployment, ensure:

1. API server is accessible from VR device
2. Firewall allows connections on port 8000
3. CORS is properly configured in the API

## Troubleshooting

### "Connection refused" error

- Ensure the API server is running
- Check that the API URL is correct
- Verify network connectivity between VR device and API server

### "Failed to parse response" error

- Check API server logs for errors
- Ensure the model is properly loaded

### Low confidence predictions

- Ensure drawings are clear and recognizable
- Try increasing the drawing canvas resolution
- Consider retraining the model with more samples

## Performance Tips

1. **Optimize texture size**: Convert textures to appropriate size before sending
2. **Async operations**: Use coroutines to prevent blocking the main thread
3. **Caching**: Cache the QuickDrawAPI reference instead of finding it repeatedly
4. **Error handling**: Implement proper error handling for network issues

## Additional Resources

- QuickDraw API Documentation: See README.md in project root
- Unity Networking: https://docs.unity3d.com/Manual/UnityWebRequest.html
- VR Best Practices: https://docs.unity3d.com/Manual/VROverview.html
