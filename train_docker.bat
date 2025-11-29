@echo off
REM Train the QuickDraw model using Docker
REM This script works on Windows

echo ==========================================
echo QuickDraw Model Training (Docker)
echo ==========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo X Docker is not installed.
    echo Please install Docker Desktop:
    echo   https://www.docker.com/products/docker-desktop/
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo X Docker is not running.
    echo Please start Docker Desktop and try again.
    exit /b 1
)

echo + Docker is running
echo.

REM Create directories if they don't exist
if not exist saved_models mkdir saved_models
if not exist quickdraw_npy mkdir quickdraw_npy

echo Starting model training...
echo This will take 10-30 minutes depending on your hardware.
echo.

REM Train using docker compose
docker compose --profile training up quickdraw-training

echo.
echo ==========================================
echo Training Complete!
echo ==========================================
echo.
echo Model saved to: saved_models/
echo.
echo Next step: Start the API
echo   docker-compose up -d
echo.
