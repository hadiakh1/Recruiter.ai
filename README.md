# Hybrid AI Recruitment System

> **📚 For detailed code architecture and system logic, see [CODE_ARCHITECTURE.md](./CODE_ARCHITECTURE.md)**

A comprehensive AI-powered recruitment platform that combines Machine Learning, Constraint Satisfaction, and Multi-Agent Systems to intelligently rank candidates based on job descriptions.

A comprehensive AI-powered resume ranking system that combines Machine Learning, Constraint Satisfaction Problem (CSP) solving, Best-First Search algorithms, Multi-Agent Systems, Reinforcement Learning, and Explainability features.

## Features

### Core Components

1. **Resume Parsing & Preprocessing**
   - PDF and TXT file support
   - Automatic extraction of skills, experience, education, certifications, and projects
   - Text cleaning and normalization
   - Feature extraction using TF-IDF and BERT embeddings

2. **Machine Learning Models**
   - Multiple classifier models: Logistic Regression, Random Forest, Decision Tree, MLP, BERT-based
   - Cross-validation and hyperparameter tuning
   - Class imbalance handling (SMOTE)
   - Job fit score prediction

3. **Constraint Satisfaction Problem (CSP) Solver**
   - Hard constraints: mandatory skills, required experience, required degree, required certifications
   - Soft constraints: optional skills, preferred experience
   - Eligibility score calculation
   - Constraint violation reporting

4. **Best-First Search Algorithm**
   - Graph-based candidate ranking
   - Heuristic function combining ML scores, skill matches, and eligibility
   - Priority queue implementation
   - NetworkX integration

5. **Multi-Agent System**
   - **Resume-Agent**: Extracts information from resumes
   - **JD-Agent**: Analyzes job descriptions
   - **Match-Agent**: Combines ML and CSP scores
   - **Ranking-Agent**: Produces final ranked list
   - Shared memory for inter-agent communication

6. **Reinforcement Learning (Optional)**
   - Q-Learning implementation
   - HR feedback integration
   - Ranking refinement based on learned policy

7. **Explainability**
   - SHAP (SHapley Additive exPlanations) for global and local explanations
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Human-readable reports
   - Feature importance visualization

8. **RESTful API**
   - Complete REST API with FastAPI
   - All endpoints for resume processing, ML prediction, CSP evaluation, ranking, RL, and explainability

9. **React Frontend**
   - Upload page for resumes and job descriptions
   - Dashboard with ranked candidates
   - Candidate profile view
   - Explainability visualization

## Project Structure

```
backend/
  ├── agents/          # Multi-agent system
  │   ├── base_agent.py
  │   ├── resume_agent.py
  │   ├── jd_agent.py
  │   ├── match_agent.py
  │   └── ranking_agent.py
  ├── ml/              # Machine Learning models
  │   ├── models.py
  │   ├── predictor.py
  │   └── explainability.py
  ├── csp/              # Constraint Satisfaction Problem
  │   └── solver.py
  ├── search/           # Best-First Search
  │   └── best_first.py
  ├── rl/               # Reinforcement Learning
  │   └── q_learning.py
  ├── api/              # REST API routes
  │   └── routes.py
  ├── utils/            # Utilities
  │   ├── resume_parser.py
  │   └── feature_extractor.py
  ├── tests/            # Unit and integration tests
  ├── app/
  │   └── main.py       # FastAPI application
  └── requirements.txt

frontend/
  ├── src/
  │   ├── pages/
  │   │   ├── UploadPage.jsx
  │   │   ├── Dashboard.jsx
  │   │   ├── CandidateProfile.jsx
  │   │   └── ExplainabilityView.jsx
  │   ├── App.jsx
  │   └── main.jsx
  ├── package.json
  └── vite.config.js
```

## Installation

### Backend Setup

1. Create a virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (first run):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

## Running the Application

### Backend

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

### Resume APIs
- `POST /api/upload-resume` - Upload and parse resume
- `POST /api/parse-resume` - Parse resume from file path
- `GET /api/get-resume-features/{candidate_id}` - Get resume features

### ML APIs
- `POST /api/predict-fit` - Predict job fit score
- `POST /api/candidate-embedding` - Generate candidate embedding

### CSP & Search APIs
- `POST /api/evaluate-candidate` - Evaluate candidate using CSP
- `POST /api/rank-candidates` - Rank candidates using Best-First Search

### RL APIs
- `POST /api/feedback` - Provide HR feedback for RL training
- `POST /api/improve-ranking` - Improve ranking using RL

### Explainability APIs
- `POST /api/shap-report` - Generate SHAP explanation
- `POST /api/lime-report` - Generate LIME explanation

### Job Description API
- `POST /api/analyze-job-description` - Analyze job description

## Testing

Run unit tests:
```bash
cd backend
python -m pytest tests/ -v
```

Or use unittest:
```bash
python -m unittest discover tests
```

## Usage Example

1. **Upload Resumes and Job Description**
   - Go to the Upload page
   - Enter job description
   - Upload resume files (PDF or TXT)
   - Click "Process & Rank Candidates"

2. **View Ranked Candidates**
   - Dashboard shows ranked candidates
   - View scores (ML, CSP, Heuristic)
   - Click on a candidate to see details

3. **View Explanations**
   - Go to Explainability page
   - Select a candidate
   - View SHAP and LIME explanations

## Model Training

To train ML models, you'll need training data. The system supports:
- Training multiple models (Logistic Regression, Random Forest, Decision Tree, MLP, BERT)
- Cross-validation
- Model persistence

Example training code:
```python
from ml.models import MLModelTrainer
import numpy as np

trainer = MLModelTrainer()
X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
scores = trainer.train_random_forest(X_train, y_train, X_test, y_test)
trainer.save_model('random_forest', 'models/rf_model.pkl')
```

## Deployment

### Backend Deployment

The backend can be deployed on:
- **Render**: Connect GitHub repo and deploy
- **PythonAnywhere**: Upload files and configure WSGI
- **Docker**: Create Dockerfile and deploy to any container platform

Example Dockerfile:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Deployment

The frontend can be deployed on:
- **Netlify**: Connect GitHub repo
- **Vercel**: Connect GitHub repo
- **GitHub Pages**: Build and deploy static files

Build for production:
```bash
cd frontend
npm run build
```

## Documentation

- API documentation available at `http://localhost:8000/docs` (Swagger UI)
- Alternative docs at `http://localhost:8000/redoc`

## Requirements Met

✅ Backend with proper folder structure  
✅ Resume parsing & preprocessing  
✅ ML models (Logistic Regression, Random Forest, Decision Tree, MLP, BERT)  
✅ CSP module with hard/soft constraints  
✅ Best-First Search algorithm  
✅ 4-agent system with shared memory  
✅ RL adapter (Q-Learning)  
✅ Explainability (SHAP & LIME)  
✅ Complete REST API  
✅ React frontend with all pages  
✅ Unit and integration tests  
✅ Deployment configurations  

## License

This project is provided as-is for educational and demonstration purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
