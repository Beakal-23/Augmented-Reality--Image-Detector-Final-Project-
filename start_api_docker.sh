#!/bin/bash

# Start the QuickDraw API using Docker
# This script works on macOS and Linux

echo "=========================================="
echo "QuickDraw API Starter (Docker)"
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

# Check if model exists
if [ ! -f "saved_models/quickdraw_house_cat_dog_car.keras" ]; then
    echo "⚠️  Warning: Model file not found!"
    echo ""
    echo "You need to train the model first:"
    echo "  ./train_docker.sh"
    echo ""
    read -p "Do you want to train the model now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./train_docker.sh
    else
        echo "Exiting. Please train the model first."
        exit 1
    fi
fi

echo "Starting QuickDraw API..."
echo ""

# Start the API
docker compose up -d

echo ""
echo "=========================================="
echo "API Started Successfully!"
echo "=========================================="
echo ""
echo "API URL: http://localhost:8001"
echo "API Docs: http://localhost:8001/docs"
echo ""
echo "Check health:"
echo "  curl http://localhost:8001/health"
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "Stop the API:"
echo "  docker-compose down"
echo ""
