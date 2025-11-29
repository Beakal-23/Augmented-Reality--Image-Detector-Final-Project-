#!/bin/bash

# Train the QuickDraw model using Docker
# This script works on macOS and Linux

echo "=========================================="
echo "QuickDraw Model Training (Docker)"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed."
    echo "Please install Docker Desktop:"
    echo "  macOS: https://www.docker.com/products/docker-desktop/"
    echo "  Linux: curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running."
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Create directories if they don't exist
mkdir -p saved_models
mkdir -p quickdraw_npy

echo "Starting model training..."
echo "This will take 10-30 minutes depending on your hardware."
echo ""

# Train using docker compose
docker compose --profile training up quickdraw-training

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: saved_models/"
echo ""
echo "Next step: Start the API"
echo "  docker-compose up -d"
echo ""
