"""
Main FastAPI Application
Entry point for the recruitment system backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.routes import router

app = FastAPI(
    title="Hybrid Recruitment System API",
    description="AI-powered resume ranking system with ML, CSP, Search, and Multi-Agent capabilities",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(router, prefix="/api", tags=["api"])

@app.get("/")
def root():
    return {
        "message": "Hybrid Recruitment System API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "resume": "/api/upload-resume, /api/parse-resume, /api/get-resume-features",
            "ml": "/api/predict-fit, /api/candidate-embedding",
            "csp": "/api/evaluate-candidate",
            "search": "/api/rank-candidates",
            "rl": "/api/feedback, /api/improve-ranking",
            "explainability": "/api/shap-report, /api/lime-report",
            "jd": "/api/analyze-job-description"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
