#!/usr/bin/env python3
"""
Helper script to view and analyze API request logs.
Useful for debugging VR drawing submissions and understanding what's being sent.
"""
import os
import json
import base64
from pathlib import Path
from PIL import Image
import io
from datetime import datetime

LOG_DIR = "api_logs"
IMAGES_DIR = os.path.join(LOG_DIR, "received_images")


def list_recent_requests(limit=10):
    """List recent API requests with their details."""
    json_files = sorted(
        Path(LOG_DIR).glob("request_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    print(f"\n{'='*80}")
    print(f"RECENT API REQUESTS (Last {limit})")
    print(f"{'='*80}\n")
    
    for i, json_file in enumerate(json_files[:limit], 1):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        timestamp = datetime.fromisoformat(data['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"[{i}] Request ID: {data['request_id']}")
        print(f"    Time: {timestamp}")
        print(f"    Client: {data['client_ip']}:{data['client_port']}")
        print(f"    User-Agent: {data.get('user_agent', 'N/A')}")
        print(f"    Base64 Length: {data['base64_length']} chars")
        
        if 'image_file' in data and data['image_file']:
            print(f"    Saved Image: {data['image_file']}")
        
        print(f"    Top Prediction: {data['predictions'][0]['class']} ({data['predictions'][0]['confidence_percent']})")
        print(f"    Success: {data['success']}")
        print()


def view_request(request_id):
    """View detailed information about a specific request."""
    json_file = os.path.join(LOG_DIR, f"request_{request_id}.json")
    
    if not os.path.exists(json_file):
        print(f"Error: Request {request_id} not found!")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"REQUEST DETAILS: {request_id}")
    print(f"{'='*80}\n")
    
    print(f"Timestamp: {data['timestamp']}")
    print(f"Client: {data['client_ip']}:{data['client_port']}")
    print(f"User-Agent: {data.get('user_agent', 'N/A')}")
    print(f"Base64 Length: {data['base64_length']} characters")
    print(f"Top K: {data.get('top_k', 'N/A')}")
    
    print(f"\nPREDICTIONS:")
    for i, pred in enumerate(data['predictions'], 1):
        print(f"  {i}. {pred['class']:15s} - {pred['confidence_percent']:6s} (confidence: {pred['confidence']:.4f})")
    
    # Show image if available
    if 'image_file' in data and data['image_file']:
        image_path = data['image_file']
        if os.path.exists(image_path):
            print(f"\nImage saved at: {image_path}")
            img = Image.open(image_path)
            print(f"Image size: {img.size}")
            print(f"Image mode: {img.mode}")
            print(f"\nTo view the image, run:")
            print(f"  open {image_path}  # macOS")
            print(f"  xdg-open {image_path}  # Linux")
    
    # Show base64 file location
    base64_file = os.path.join(LOG_DIR, f"request_{request_id}_base64.txt")
    if os.path.exists(base64_file):
        print(f"\nBase64 data saved at: {base64_file}")
        print(f"To decode manually:")
        print(f"  base64 -d {base64_file} > decoded_image.png")


def decode_base64_file(request_id, output_path=None):
    """Decode a base64 file to an image."""
    base64_file = os.path.join(LOG_DIR, f"request_{request_id}_base64.txt")
    
    if not os.path.exists(base64_file):
        print(f"Error: Base64 file for request {request_id} not found!")
        return
    
    with open(base64_file, 'r') as f:
        base64_data = f.read()
    
    try:
        image_data = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(image_data))
        
        if output_path is None:
            output_path = f"decoded_{request_id}.png"
        
        img.save(output_path)
        print(f"✓ Image decoded and saved to: {output_path}")
        print(f"  Size: {img.size}")
        print(f"  Mode: {img.mode}")
        
    except Exception as e:
        print(f"✗ Error decoding image: {e}")


def show_statistics():
    """Show statistics about all logged requests."""
    json_files = list(Path(LOG_DIR).glob("request_*.json"))
    
    print(f"\n{'='*80}")
    print(f"API REQUEST STATISTICS")
    print(f"{'='*80}\n")
    
    print(f"Total Requests: {len(json_files)}")
    
    if not json_files:
        print("No requests logged yet.")
        return
    
    # Count predictions by class
    class_counts = {}
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if data['predictions']:
            top_class = data['predictions'][0]['class']
            class_counts[top_class] = class_counts.get(top_class, 0) + 1
    
    print(f"\nTop Predictions Distribution:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name:15s}: {count:3d} ({count/len(json_files)*100:.1f}%)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nQuickDraw API Log Viewer")
        print("=" * 80)
        print("\nUsage:")
        print("  python view_logs.py list [limit]         - List recent requests")
        print("  python view_logs.py view <request_id>    - View specific request details")
        print("  python view_logs.py decode <request_id>  - Decode base64 to image")
        print("  python view_logs.py stats                - Show statistics")
        print("\nExamples:")
        print("  python view_logs.py list 20")
        print("  python view_logs.py view 20251201_033338_661741")
        print("  python view_logs.py decode 20251201_033338_661741")
        print("  python view_logs.py stats")
        print()
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "list":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        list_recent_requests(limit)
    
    elif command == "view":
        if len(sys.argv) < 3:
            print("Error: Please provide a request ID")
            sys.exit(1)
        view_request(sys.argv[2])
    
    elif command == "decode":
        if len(sys.argv) < 3:
            print("Error: Please provide a request ID")
            sys.exit(1)
        decode_base64_file(sys.argv[2])
    
    elif command == "stats":
        show_statistics()
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
