@echo off
REM Start the QuickDraw API using Docker
REM This script works on Windows

echo ==========================================
echo QuickDraw API Starter (Docker)
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

REM Check if model exists
if not exist "saved_models\quickdraw_house_cat_dog_car.keras" (
    echo WARNING: Model file not found!
    echo.
    echo You need to train the model first:
    echo   train_docker.bat
    echo.
    set /p REPLY="Do you want to train the model now? (y/n) "
    if /i "%REPLY%"=="y" (
        call train_docker.bat
    ) else (
        echo Exiting. Please train the model first.
        exit /b 1
    )
)

echo Starting QuickDraw API...
echo.

REM Start the API
docker compose up -d

echo.
echo ==========================================
echo API Started Successfully!
echo ==========================================
echo.
echo API URL: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Check health:
echo   curl http://localhost:8000/health
echo.
echo View logs:
echo   docker-compose logs -f
echo.
echo Stop the API:
echo   docker-compose down
echo.
