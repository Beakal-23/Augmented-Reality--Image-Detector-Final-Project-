# HTTPS Deployment Guide

## Overview

This guide covers deploying your QuickDraw API with HTTPS for production use, making it accessible from anywhere including VR applications.

## Deployment Options

### Option 1: Hugging Face Spaces (Recommended for Demos)

Free hosting with HTTPS automatically configured.

1. **Upload to Hugging Face Hub:**

   ```bash
   pip install huggingface_hub
   python3 upload_to_huggingface.py
   ```

2. **Create a Hugging Face Space:**

   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Docker" as SDK
   - Name it (e.g., "quickdraw-api")

3. **Add Dockerfile for Space:**
   Create `Dockerfile` in your Space:

   ```dockerfile
   FROM python:3.10-slim

   WORKDIR /app

   RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 7860

   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
   ```

4. **Your API will be available at:**
   ```
   https://YOUR_USERNAME-quickdraw-api.hf.space
   ```

### Option 2: Cloud Platforms (Production-Ready)

#### A. Railway.app (Easy, Free Tier)

1. **Sign up at https://railway.app**

2. **Deploy from GitHub:**

   - Click "New Project" → "Deploy from GitHub"
   - Select your repository
   - Railway auto-detects Docker

3. **Configure:**

   - Set port to `8000`
   - Add environment variables if needed

4. **Get HTTPS URL:**
   - Railway provides: `https://your-app.railway.app`

#### B. Render.com (Free Tier Available)

1. **Sign up at https://render.com**

2. **Create Web Service:**

   - New → Web Service
   - Connect GitHub repository
   - Choose "Docker" environment

3. **Configure:**

   ```yaml
   # render.yaml
   services:
     - type: web
       name: quickdraw-api
       env: docker
       dockerfilePath: ./Dockerfile
       envVars:
         - key: PORT
           value: 8000
   ```

4. **HTTPS URL:**
   - Render provides: `https://your-app.onrender.com`

#### C. Google Cloud Run (Pay-per-use)

1. **Install gcloud CLI:**

   ```bash
   # macOS
   brew install google-cloud-sdk
   ```

2. **Build and push:**

   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID

   # Build image
   docker build -t gcr.io/YOUR_PROJECT_ID/quickdraw-api .

   # Push to Google Container Registry
   docker push gcr.io/YOUR_PROJECT_ID/quickdraw-api
   ```

3. **Deploy to Cloud Run:**

   ```bash
   gcloud run deploy quickdraw-api \
     --image gcr.io/YOUR_PROJECT_ID/quickdraw-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8000
   ```

4. **HTTPS URL:**
   - Cloud Run provides: `https://quickdraw-api-HASH-uc.a.run.app`

#### D. AWS (Elastic Beanstalk or ECS)

1. **Install AWS CLI:**

   ```bash
   pip install awscli
   aws configure
   ```

2. **Deploy with Elastic Beanstalk:**

   ```bash
   # Install EB CLI
   pip install awsebcli

   # Initialize
   eb init -p docker quickdraw-api

   # Create environment
   eb create quickdraw-api-env

   # Deploy
   eb deploy
   ```

3. **Configure HTTPS:**
   - Use AWS Certificate Manager for SSL
   - Configure in EB console

### Option 3: Self-Hosted with Domain

If you have your own server and domain:

#### Using Nginx + Let's Encrypt

1. **Install dependencies:**

   ```bash
   sudo apt update
   sudo apt install nginx certbot python3-certbot-nginx
   ```

2. **Configure Nginx** (`/etc/nginx/sites-available/quickdraw`):

   ```nginx
   server {
       listen 80;
       server_name api.yourdomain.com;

       location / {
           proxy_pass http://localhost:8001;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

3. **Enable site:**

   ```bash
   sudo ln -s /etc/nginx/sites-available/quickdraw /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

4. **Get SSL certificate:**

   ```bash
   sudo certbot --nginx -d api.yourdomain.com
   ```

5. **Start your API:**

   ```bash
   docker compose up -d quickdraw-api
   ```

6. **HTTPS URL:**
   ```
   https://api.yourdomain.com
   ```

#### Using Caddy (Automatic HTTPS)

1. **Install Caddy:**

   ```bash
   sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
   sudo apt update
   sudo apt install caddy
   ```

2. **Configure Caddy** (`/etc/caddy/Caddyfile`):

   ```
   api.yourdomain.com {
       reverse_proxy localhost:8001
   }
   ```

3. **Restart Caddy:**

   ```bash
   sudo systemctl restart caddy
   ```

   Caddy automatically gets and renews SSL certificates!

## Updating Unity/VR App for HTTPS

Once deployed, update your Unity code:

```csharp
public class QuickDrawAPI : MonoBehaviour
{
    // Change to your HTTPS URL
    private string apiUrl = "https://your-app.railway.app/predict/base64";

    // Rest of code stays the same...
}
```

## Testing HTTPS Deployment

```bash
# Test health endpoint
curl https://your-domain.com/health

# Test prediction
curl -X POST https://your-domain.com/predict/base64 \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "YOUR_BASE64_DATA", "top_k": 3}'

# Test from VR/Unity
# Use the /predict/base64 endpoint with your drawing's base64 data
```

## Environment Variables

For production, set these in your deployment platform:

```bash
# Optional configurations
MODEL_PATH=/app/saved_models/quickdraw_house_cat_dog_car.keras
LOG_LEVEL=INFO
CORS_ORIGINS=https://yourvr-app.com
```

## Monitoring

### Railway/Render

- Built-in logs and metrics in dashboard

### Cloud Run

```bash
gcloud logging read "resource.type=cloud_run_revision" --limit 50
```

### Self-hosted

```bash
# View API logs
docker logs -f quickdraw-api

# View Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

## Cost Estimates

| Platform                   | Free Tier                | Paid Plans  |
| -------------------------- | ------------------------ | ----------- |
| Hugging Face Spaces        | ✅ Yes                   | $0/month    |
| Railway                    | 500 hours/month          | $5+/month   |
| Render                     | 750 hours/month          | $7+/month   |
| Google Cloud Run           | 2M requests/month        | Pay per use |
| AWS Elastic Beanstalk      | 750 hours/month (1 year) | $10+/month  |
| Self-hosted (DigitalOcean) | -                        | $6+/month   |

## Security Best Practices

1. **Rate Limiting:**
   Add to `main.py`:

   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address

   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

   @app.post("/predict/base64")
   @limiter.limit("10/minute")  # 10 requests per minute
   async def predict_from_base64(...):
       ...
   ```

2. **API Key Authentication** (optional):

   ```python
   from fastapi import Header, HTTPException

   async def verify_api_key(x_api_key: str = Header(...)):
       if x_api_key != os.getenv("API_KEY"):
           raise HTTPException(status_code=403, detail="Invalid API key")

   @app.post("/predict/base64", dependencies=[Depends(verify_api_key)])
   async def predict_from_base64(...):
       ...
   ```

3. **CORS Configuration:**
   Update `main.py` to restrict origins:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourvr-app.com"],  # Specific domains only
       allow_credentials=True,
       allow_methods=["POST", "GET"],
       allow_headers=["*"],
   )
   ```

## Quick Start: Railway Deployment

1. **Push code to GitHub** (already done!)

2. **Sign up at Railway.app**

3. **New Project → Deploy from GitHub**

4. **Select your repository**

5. **Done!** Your API is live with HTTPS in ~2 minutes

Railway automatically:

- ✅ Builds Docker image
- ✅ Provides HTTPS URL
- ✅ Handles SSL certificates
- ✅ Provides monitoring dashboard
- ✅ Auto-deploys on git push

## Next Steps

1. Choose your deployment platform
2. Run `python3 upload_to_huggingface.py` to share the model
3. Deploy using one of the methods above
4. Update your VR app with the HTTPS URL
5. Test thoroughly before demo!

## Support

For deployment issues:

- Railway: https://docs.railway.app
- Render: https://render.com/docs
- Cloud Run: https://cloud.google.com/run/docs
- Hugging Face: https://huggingface.co/docs/hub
