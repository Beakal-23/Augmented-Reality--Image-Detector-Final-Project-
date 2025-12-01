# API Logging Guide

## Overview

The FastAPI server includes comprehensive logging to help you debug and monitor VR drawing submissions. All requests are logged with detailed information including:

- Request metadata (timestamp, client IP, user-agent)
- Base64 image data (saved to file)
- Decoded PNG images (saved for inspection)
- Model predictions with confidence scores
- Complete request/response cycle

## Log Directory Structure

```
api_logs/
├── api.log                              # Main application log
├── requests_detailed.log                 # Detailed request log
├── received_images/                      # Decoded PNG images from requests
│   ├── request_20251201_034629_866032.png
│   └── ...
├── request_20251201_034629_866032.json   # Detailed JSON log
├── request_20251201_034629_866032_base64.txt  # Base64 image data
└── ...
```

## Using the Log Viewer

A helper script `view_logs.py` is provided to easily view and analyze logged requests.

### List Recent Requests

```bash
python3 view_logs.py list [limit]
```

Example:

```bash
$ python3 view_logs.py list 5

================================================================================
RECENT API REQUESTS (Last 5)
================================================================================

[1] Request ID: 20251201_034629_866032
    Time: 2025-12-01 03:46:29
    Client: 192.168.65.1:23596
    User-Agent: Unity-VR-Drawing-App/2.0
    Base64 Length: 124 chars
    Saved Image: api_logs/received_images/request_20251201_034629_866032.png
    Top Prediction: mountain (99.91%)
    Success: True
```

### View Detailed Request Information

```bash
python3 view_logs.py view <request_id>
```

Example:

```bash
$ python3 view_logs.py view 20251201_034629_866032

================================================================================
REQUEST DETAILS: 20251201_034629_866032
================================================================================

Timestamp: 2025-12-01T03:46:29.947824
Client: 192.168.65.1:23596
User-Agent: Unity-VR-Drawing-App/2.0
Base64 Length: 124 characters
Top K: 3

PREDICTIONS:
  1. mountain        - 99.91% (confidence: 0.9991)
  2. hand            - 0.04%  (confidence: 0.0004)
  3. banana          - 0.03%  (confidence: 0.0003)

Image saved at: api_logs/received_images/request_20251201_034629_866032.png
```

### Decode Base64 to Image

```bash
python3 view_logs.py decode <request_id>
```

This will decode the base64 data and save it as a PNG file for inspection.

### View Statistics

```bash
python3 view_logs.py stats
```

Shows aggregated statistics about all logged requests including top prediction distribution.

## Viewing Images

### On macOS:

```bash
open api_logs/received_images/request_<ID>.png
```

### On Linux:

```bash
xdg-open api_logs/received_images/request_<ID>.png
```

### Using Python:

```python
from PIL import Image
img = Image.open('api_logs/received_images/request_<ID>.png')
img.show()
```

## Console Logs

The API also logs to console in real-time. To view:

```bash
docker logs -f quickdraw-api
```

Example output:

```
2025-12-01 03:46:29,947 - main - INFO - ================================================================================
2025-12-01 03:46:29,947 - main - INFO - [REQUEST 20251201_034629_866032] New prediction request from VR
2025-12-01 03:46:29,947 - main - INFO - [REQUEST 20251201_034629_866032] Client: 192.168.65.1:23596
2025-12-01 03:46:29,947 - main - INFO - [REQUEST 20251201_034629_866032] User-Agent: Unity-VR-Drawing-App/2.0
2025-12-01 03:46:29,947 - main - INFO - [REQUEST 20251201_034629_866032] Top K: 3
2025-12-01 03:46:29,947 - main - INFO - [REQUEST 20251201_034629_866032] Base64 image length: 124 characters
2025-12-01 03:46:29,947 - main - INFO - [REQUEST 20251201_034629_866032] Decoded image saved to: api_logs/received_images/request_20251201_034629_866032.png
2025-12-01 03:46:29,947 - main - INFO - [REQUEST 20251201_034629_866032] PREDICTIONS:
2025-12-01 03:46:29,947 - main - INFO - [REQUEST 20251201_034629_866032]   1. mountain: 99.91% (confidence: 0.9991)
2025-12-01 03:46:29,947 - main - INFO - [REQUEST 20251201_034629_866032]   2. hand: 0.04% (confidence: 0.0004)
2025-12-01 03:46:29,947 - main - INFO - [REQUEST 20251201_034629_866032]   3. banana: 0.03% (confidence: 0.0003)
2025-12-01 03:46:29,947 - main - INFO - [REQUEST 20251201_034629_866032] ✓ Prediction completed successfully
```

## Debugging VR Drawings

### Common Issues

1. **Wrong predictions**: Check the saved PNG to see what the model actually received

   ```bash
   open api_logs/received_images/request_<ID>.png
   ```

2. **Base64 decoding errors**: The base64 data is saved to `request_<ID>_base64.txt` for manual inspection

3. **Image preprocessing issues**: Check the logs for "Preprocessed image shape" - should be `(1, 28, 28, 1)`

### Understanding What's Being Captured

The saved PNG images show exactly what the VR application sent. Compare these with what you drew in VR to understand:

- Is the entire scene being captured or just the drawing panel?
- Are the drawings properly centered?
- Is the image resolution appropriate?
- Are there any artifacts or unwanted elements?

## For Demo Purposes

The extensive logging is perfect for demonstrations:

1. **Live monitoring**: Run `docker logs -f quickdraw-api` during the demo to show real-time inference
2. **Post-analysis**: Use `view_logs.py` to review what was drawn and predicted
3. **Visual verification**: Open saved images to verify the model's input

## Log Retention

Logs accumulate over time. To clean up:

```bash
# Remove all logs
rm -rf api_logs/*

# Remove logs older than 7 days
find api_logs -type f -mtime +7 -delete
```

## JSON Log Format

Each request is saved as a JSON file with complete details:

```json
{
  "request_id": "20251201_034629_866032",
  "timestamp": "2025-12-01T03:46:29.947824",
  "client_ip": "192.168.65.1",
  "client_port": 23596,
  "user_agent": "Unity-VR-Drawing-App/2.0",
  "base64_length": 124,
  "image_file": "api_logs/received_images/request_20251201_034629_866032.png",
  "top_k": 3,
  "predictions": [
    {
      "class": "mountain",
      "confidence": 0.9991400241851807,
      "confidence_percent": "99.91%"
    }
  ],
  "success": true
}
```

This format makes it easy to programmatically analyze request patterns and model performance.
