# Deployment Guide

## Backend Deployment

### Option 1: Render

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.9+

### Option 2: PythonAnywhere

1. Upload your backend files via Files tab
2. Create a new Web app
3. Configure WSGI file:
```python
import sys
path = '/home/yourusername/resume/backend'
if path not in sys.path:
    sys.path.append(path)

from app.main import app as application
```

### Option 3: Docker

1. Build the image:
```bash
cd backend
docker build -t resume-backend .
```

2. Run the container:
```bash
docker run -p 8000:8000 resume-backend
```

3. For production, use docker-compose or Kubernetes

## Frontend Deployment

### Option 1: Netlify

1. Connect GitHub repository
2. Configure:
   - **Build command**: `cd frontend && npm install && npm run build`
   - **Publish directory**: `frontend/dist`
   - **Environment variables**: `VITE_API_URL=https://your-backend-url.com`

### Option 2: Vercel

1. Import GitHub repository
2. Configure:
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

### Option 3: GitHub Pages

1. Build the frontend:
```bash
cd frontend
npm run build
```

2. Deploy `dist` folder to GitHub Pages

## Environment Variables

### Backend
- `PORT`: Server port (default: 8000)
- `MODEL_PATH`: Path to trained ML models (optional)

### Frontend
- `VITE_API_URL`: Backend API URL (default: http://localhost:8000)

## Production Checklist

- [ ] Set CORS origins to specific domains (not "*")
- [ ] Use environment variables for sensitive data
- [ ] Enable HTTPS
- [ ] Set up database for persistent storage
- [ ] Configure logging
- [ ] Set up monitoring and error tracking
- [ ] Train and save ML models
- [ ] Configure rate limiting
- [ ] Set up backup strategy





