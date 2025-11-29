using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// QuickDraw Sketch Recognition Integration for Unity VR/AR
/// This script provides methods to send drawings to the API and receive predictions
/// </summary>
public class QuickDrawAPI : MonoBehaviour
{
    [Header("API Configuration")]
    [Tooltip("URL of the QuickDraw API endpoint")]
    public string apiUrl = "http://localhost:8000/predict/base64";

    [Tooltip("Number of top predictions to return")]
    [Range(1, 4)]
    public int topK = 3;

    [Tooltip("Minimum confidence threshold (0-1)")]
    [Range(0f, 1f)]
    public float confidenceThreshold = 0.7f;

    [Header("Events")]
    public UnityEngine.Events.UnityEvent<PredictionResponse> OnPredictionReceived;
    public UnityEngine.Events.UnityEvent<string> OnPredictionError;

    #region Data Classes

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

    #endregion

    #region Public Methods

    /// <summary>
    /// Recognize a drawing from a Texture2D
    /// </summary>
    /// <param name="drawingTexture">The texture containing the drawing</param>
    public void RecognizeDrawing(Texture2D drawingTexture)
    {
        StartCoroutine(RecognizeDrawingCoroutine(drawingTexture));
    }

    /// <summary>
    /// Recognize a drawing from a RenderTexture (useful for VR drawing panels)
    /// </summary>
    /// <param name="renderTexture">The RenderTexture containing the drawing</param>
    public void RecognizeDrawing(RenderTexture renderTexture)
    {
        Texture2D texture = ConvertRenderTextureToTexture2D(renderTexture);
        RecognizeDrawing(texture);
        Destroy(texture); // Cleanup
    }

    /// <summary>
    /// Check if the API is healthy
    /// </summary>
    public void CheckAPIHealth()
    {
        StartCoroutine(CheckHealthCoroutine());
    }

    #endregion

    #region Coroutines

    private IEnumerator RecognizeDrawingCoroutine(Texture2D drawingTexture)
    {
        // Convert texture to PNG bytes
        byte[] imageBytes = drawingTexture.EncodeToPNG();

        // Convert to base64
        string base64Image = "data:image/png;base64," + Convert.ToBase64String(imageBytes);

        // Create request payload
        PredictionRequest request = new PredictionRequest
        {
            image_base64 = base64Image,
            top_k = topK
        };

        string jsonData = JsonUtility.ToJson(request);

        // Send POST request
        using (UnityWebRequest www = UnityWebRequest.Post(apiUrl, ""))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                try
                {
                    PredictionResponse response = JsonUtility.FromJson<PredictionResponse>(www.downloadHandler.text);

                    if (response.success)
                    {
                        Debug.Log($"[QuickDraw] Top prediction: {response.predictions[0].@class} " +
                                 $"({response.predictions[0].confidence_percent})");

                        // Invoke event
                        OnPredictionReceived?.Invoke(response);

                        // Process based on confidence
                        ProcessPrediction(response);
                    }
                    else
                    {
                        string error = $"Prediction failed: {response.message}";
                        Debug.LogError($"[QuickDraw] {error}");
                        OnPredictionError?.Invoke(error);
                    }
                }
                catch (Exception e)
                {
                    string error = $"Failed to parse response: {e.Message}";
                    Debug.LogError($"[QuickDraw] {error}");
                    OnPredictionError?.Invoke(error);
                }
            }
            else
            {
                string error = $"API request failed: {www.error}";
                Debug.LogError($"[QuickDraw] {error}");
                OnPredictionError?.Invoke(error);
            }
        }
    }

    private IEnumerator CheckHealthCoroutine()
    {
        string healthUrl = apiUrl.Replace("/predict/base64", "/health");

        using (UnityWebRequest www = UnityWebRequest.Get(healthUrl))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"[QuickDraw] API Health: {www.downloadHandler.text}");
            }
            else
            {
                Debug.LogError($"[QuickDraw] Health check failed: {www.error}");
            }
        }
    }

    #endregion

    #region Private Methods

    private void ProcessPrediction(PredictionResponse response)
    {
        if (response.predictions.Length == 0)
        {
            Debug.LogWarning("[QuickDraw] No predictions returned");
            return;
        }

        PredictionResult topPrediction = response.predictions[0];

        // Check if confidence meets threshold
        if (topPrediction.confidence >= confidenceThreshold)
        {
            Debug.Log($"[QuickDraw] Recognized: {topPrediction.@class} " +
                     $"with {topPrediction.confidence_percent} confidence");

            // Handle recognized class
            HandleRecognizedClass(topPrediction.@class, topPrediction.confidence);
        }
        else
        {
            Debug.Log($"[QuickDraw] Low confidence ({topPrediction.confidence_percent}). " +
                     $"Drawing not clear enough.");
        }
    }

    private void HandleRecognizedClass(string className, float confidence)
    {
        // Override this method or use events to handle different recognized classes
        // Example: Spawn appropriate 3D object in VR scene

        switch (className.ToLower())
        {
            case "house":
                // Spawn house model
                Debug.Log("[QuickDraw] Would spawn house here");
                break;

            case "cat":
                // Spawn cat model
                Debug.Log("[QuickDraw] Would spawn cat here");
                break;

            case "dog":
                // Spawn dog model
                Debug.Log("[QuickDraw] Would spawn dog here");
                break;

            case "car":
                // Spawn car model
                Debug.Log("[QuickDraw] Would spawn car here");
                break;

            default:
                Debug.LogWarning($"[QuickDraw] Unknown class: {className}");
                break;
        }
    }

    private Texture2D ConvertRenderTextureToTexture2D(RenderTexture renderTexture)
    {
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = renderTexture;

        Texture2D texture = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
        texture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        texture.Apply();

        RenderTexture.active = currentRT;
        return texture;
    }

    #endregion

    #region Unity Lifecycle

    private void Start()
    {
        // Optional: Check API health on start
        if (Application.isPlaying)
        {
            CheckAPIHealth();
        }
    }

    #endregion
}
