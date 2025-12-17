# Code Architecture & System Logic

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [File Structure & Relationships](#file-structure--relationships)
4. [Data Flow](#data-flow)
5. [Core Components](#core-components)
6. [AI Agents & Their Roles](#ai-agents--their-roles)
7. [Scoring & Ranking Logic](#scoring--ranking-logic)
8. [Database Schema](#database-schema)
9. [API Endpoints](#api-endpoints)
10. [Frontend Structure](#frontend-structure)

---

## System Overview

This is a **Hybrid AI Recruitment System** that combines:
- **Machine Learning (ML)** for predictive job-fit scoring
- **Constraint Satisfaction Problem (CSP)** for eligibility checking
- **Best-First Search** for intelligent ranking
- **Multi-Agent System** for modular processing

The system processes resumes, analyzes job descriptions, and ranks candidates based on their fit for the position.

---

## Architecture Diagram

```
┌─────────────┐
│   Frontend  │ (React + Vite)
│  (Port 3000)│
└──────┬──────┘
       │ HTTP/REST API
       │
┌──────▼──────────────────────────────────────────────┐
│              Backend API (FastAPI)                  │
│              backend/api/routes.py                  │
│              (Port 8000)                            │
└──────┬──────────────────────────────────────────────┘
       │
       ├─────────────────────────────────────────────┐
       │                                               │
┌──────▼──────────┐                    ┌─────────────▼──────┐
│  Resume Agent   │                    │    JD Agent         │
│  (Parsing)      │                    │  (Job Analysis)     │
└──────┬──────────┘                    └─────────────┬───────┘
       │                                               │
       │ Resume Data + Features                       │ Job Requirements
       │                                               │
       └───────────────┬───────────────────────────────┘
                       │
              ┌────────▼─────────┐
              │   Match Agent    │
              │  (ML + CSP)      │
              └────────┬──────────┘
                       │
              ┌────────▼─────────┐
              │  Ranking Agent   │
              │ (Best-First)     │
              └────────┬──────────┘
                       │
              ┌────────▼─────────┐
              │  Ranked Results  │
              └──────────────────┘
```

---

## File Structure & Relationships

### Backend Structure

```
backend/
├── app/
│   └── main.py                 # FastAPI application entry point
│
├── api/
│   └── routes.py               # All API endpoints (auth, upload, ranking)
│
├── agents/                     # Multi-Agent System
│   ├── base_agent.py          # Base class for all agents
│   ├── resume_agent.py        # Extracts data from resumes
│   ├── jd_agent.py            # Analyzes job descriptions
│   ├── match_agent.py         # Combines ML + CSP scores
│   └── ranking_agent.py       # Produces final ranked list
│
├── utils/                      # Utility modules
│   ├── resume_parser.py       # PDF/TXT parsing, name/skill extraction
│   └── feature_extractor.py   # TF-IDF, embeddings, structured features
│
├── ml/                         # Machine Learning
│   ├── models.py              # ML model trainer (RF, LR, MLP, etc.)
│   ├── predictor.py           # ML prediction service
│   └── explainability.py      # SHAP/LIME explanations
│
├── csp/                        # Constraint Satisfaction
│   └── solver.py              # CSP solver for eligibility checking
│
├── search/                     # Search Algorithms
│   └── best_first.py          # Best-First Search for ranking
│
├── rl/                         # Reinforcement Learning (optional)
│   └── q_learning.py          # Q-Learning for ranking improvement
│
├── database.py                 # SQLite database operations
│
└── requirements.txt            # Python dependencies
```

### Frontend Structure

```
frontend/
├── src/
│   ├── App.jsx                # Main app component, routing
│   ├── pages/
│   │   ├── Login.jsx          # User authentication
│   │   ├── Signup.jsx         # User registration
│   │   ├── UploadPage.jsx     # Resume upload + job description
│   │   ├── Dashboard.jsx      # Ranked candidates display
│   │   ├── CandidateProfile.jsx # Individual candidate details
│   │   └── ExplainabilityView.jsx # AI explanation view
│   └── utils/
│       └── api.js             # API client (axios wrapper)
│
└── package.json               # Node dependencies
```

---

## Data Flow

### Complete Processing Pipeline

```
1. USER UPLOADS RESUME
   └─> UploadPage.jsx
       └─> POST /upload-resume
           └─> routes.py::upload_resume()
               └─> ResumeAgent.process()
                   ├─> resume_parser.py::parse_resume()
                   │   ├─> Extract text from PDF/TXT
                   │   ├─> Extract name, skills, experience, education
                   │   └─> Return structured resume_data
                   │
                   └─> feature_extractor.py::extract_all_features()
                       ├─> TF-IDF features (1000 dims)
                       ├─> BERT embeddings (384 dims)
                       └─> Structured features (10 dims)
                           └─> Save to database

2. USER ENTERS JOB DESCRIPTION
   └─> UploadPage.jsx
       └─> POST /analyze-job-description
           └─> routes.py::analyze_job_description()
               └─> JDAgent.process()
                   └─> jd_agent.py::extract_requirements()
                       ├─> Extract required_skills
                       ├─> Extract optional_skills
                       ├─> Extract required_experience
                       └─> Return job_requirements
                           └─> Save to database

3. RANKING REQUEST
   └─> Dashboard.jsx
       └─> POST /rank-candidates
           └─> routes.py::rank_candidates()
               │
               ├─> For each candidate:
               │   └─> MatchAgent.process()
               │       ├─> MLPredictor.predict_fit_score()
               │       │   └─> Uses features to predict job-fit (0-1)
               │       │
               │       └─> CSPSolver.evaluate_candidate()
               │           ├─> Check required_skills match
               │           ├─> Check required_experience
               │           ├─> Check required_degree
               │           └─> Return eligibility_score (0-1)
               │
               └─> RankingAgent.process()
                   └─> BestFirstSearch.rank_candidates()
                       ├─> Calculate final_score = (ML*0.6) + (CSP*0.4)
                       ├─> Sort by final_score (descending)
                       └─> Return ranked list
```

---

## Core Components

### 1. Resume Parser (`backend/utils/resume_parser.py`)

**Purpose**: Extract structured information from resume files

**Key Methods**:
- `extract_text_from_pdf()`: Extract text from PDF files
- `extract_name()`: Extract candidate name (first few lines, name patterns)
- `extract_skills()`: Extract skills using keyword matching
- `extract_experience()`: Extract years, companies, positions
- `extract_education()`: Extract degrees, universities
- `parse_resume()`: Main method that orchestrates all extractions

**Output**: Dictionary with `name`, `skills`, `experience`, `education`, `certifications`, `projects`, `raw_text`, `cleaned_text`

### 2. Feature Extractor (`backend/utils/feature_extractor.py`)

**Purpose**: Generate feature vectors for ML models

**Key Methods**:
- `extract_tfidf_features()`: TF-IDF vectorization (1000 dimensions)
- `extract_embedding()`: BERT/SentenceTransformer embeddings (384 dimensions)
- `extract_structured_features()`: Structured features (10 dimensions)
  - Skills count
  - Experience years
  - Education level (PhD/Masters/Bachelors flags)
  - Certifications count
  - Projects count
  - Text length, word count
- `extract_all_features()`: Combines all features into `combined` vector

**Output**: Dictionary with `tfidf`, `embedding`, `structured`, `combined` arrays

### 3. ML Predictor (`backend/ml/predictor.py`)

**Purpose**: Predict job-fit scores using ML models

**Key Methods**:
- `predict_fit_score()`: Main prediction method
  - If model loaded: Uses trained model
  - Else: Uses `_default_score()` with job requirements check
- `_default_score()`: Conservative scoring when no model
  - Checks if candidate has required skills
  - If <50% required skills match → 0.0
  - If no skills at all → 0.0
  - Otherwise: Feature-based scoring (capped at 20-60%)

**Scoring Logic**:
- **With Required Skills**: Score = skill_match_ratio × 0.6 + feature_bonus × 0.4
- **With Optional Skills Only**: Score = optional_match_ratio × 0.4 + feature_bonus × 0.2 (capped at 60%)
- **No Skills**: Score = 0.0

### 4. CSP Solver (`backend/csp/solver.py`)

**Purpose**: Evaluate candidate eligibility using constraint satisfaction

**Key Methods**:
- `check_mandatory_skills()`: Verify all required skills present
- `check_required_experience()`: Verify experience requirement met
- `check_required_degree()`: Verify degree requirement met
- `check_optional_skills()`: Calculate optional skill match ratio
- `evaluate_candidate()`: Main evaluation method

**Scoring Logic**:
- **Hard Constraints Not Satisfied**: `eligibility_score = 0.0`
- **All Hard Constraints Satisfied**:
  - If no hard constraints exist (all optional):
    - No skills → `0.1`
    - Has skills → `0.3 + (optional_match × 0.4)` (max 0.7)
  - If hard constraints exist:
    - Base: `0.7`
    - Bonus: `min(optional_match × 0.3, 0.3)`
    - Total: `0.7 + bonus` (max 1.0)

### 5. Best-First Search (`backend/search/best_first.py`)

**Purpose**: Rank candidates using graph-based search

**Key Methods**:
- `rank_candidates()`: Main ranking method
  - If candidates have `final_score`: Sort directly by it
  - Else: Use Best-First Search with heuristic
- `heuristic()`: Calculate heuristic value
  - `h(n) = (ML × 0.6) + (CSP × 0.4)`
  - Same formula as Match-Agent for consistency

**Ranking Logic**:
1. Extract `final_score` from each candidate
2. Sort by `final_score` (descending)
3. Assign rank numbers (1, 2, 3, ...)
4. Preserve all metadata (name, filename, scores)

---

## AI Agents & Their Roles

### 1. Resume Agent (`backend/agents/resume_agent.py`)

**Role**: Extract and structure resume information

**Input**: File path, file type (PDF/TXT)

**Process**:
1. Parse resume using `ResumeParser`
2. Extract features using `FeatureExtractor`
3. Convert numpy arrays to lists for JSON

**Output**: `{resume_data, features, status}`

**Used By**: `/upload-resume` endpoint

---

### 2. JD Agent (`backend/agents/jd_agent.py`)

**Role**: Analyze job description and extract requirements

**Input**: Job description text

**Process**:
1. Extract required skills (mandatory keywords)
2. Extract optional skills (preferred keywords)
3. Extract experience requirements (years)
4. Extract degree requirements
5. Extract certification requirements

**Output**: `{job_requirements, status}`

**Used By**: `/analyze-job-description` endpoint

---

### 3. Match Agent (`backend/agents/match_agent.py`)

**Role**: Combine ML and CSP scores into final job-fit score

**Input**: `{candidate_data, job_requirements, features}`

**Process**:
1. Get ML score from `MLPredictor`
2. Get CSP score from `CSPSolver`
3. Calculate final score:
   - If hard constraints NOT satisfied: `final = ML × 0.1`
   - Else: `final = (ML × 0.6) + (CSP × 0.4)`

**Output**: `{ml_score, csp_score, final_score, csp_details, status}`

**Used By**: `/rank-candidates` endpoint

---

### 4. Ranking Agent (`backend/agents/ranking_agent.py`)

**Role**: Produce final ranked candidate list

**Input**: `{candidates, ml_scores, csp_scores, job_requirements}`

**Process**:
1. Use `BestFirstSearch.rank_candidates()`
2. Sort by `final_score` (if available)
3. Assign rank numbers

**Output**: `{ranked_candidates, total_candidates, status}`

**Used By**: `/rank-candidates` endpoint

---

## Scoring & Ranking Logic

### Final Score Calculation

```
final_score = (ML_score × 0.6) + (CSP_score × 0.4)
```

**Exception**: If hard constraints not satisfied:
```
final_score = ML_score × 0.1  (Heavy penalty)
```

### ML Score Calculation

**With Trained Model**:
- Uses model prediction (0-1)
- Applies skill match penalty if <50% required skills

**Without Model (Default)**:
- Checks required skills match
- If <50% match → 0.0
- If no skills → 0.0
- Otherwise: Feature-based scoring

### CSP Score Calculation

**Hard Constraints**:
- Required skills: All must be present
- Required experience: Must meet minimum
- Required degree: Must have degree
- Required certifications: All must be present

**If Any Hard Constraint Fails**: `CSP_score = 0.0`

**If All Hard Constraints Satisfied**:
- Base score: 0.7
- Optional skills bonus: `min(optional_match × 0.3, 0.3)`
- Total: `0.7 + bonus` (max 1.0)

### Ranking Algorithm

1. **Primary Method**: Direct sorting by `final_score`
   - Fastest and most accurate
   - Uses the exact score from Match-Agent

2. **Fallback Method**: Best-First Search
   - Builds graph of candidates
   - Uses heuristic: `(ML × 0.6) + (CSP × 0.4)`
   - Priority queue selects best candidates

---

## Database Schema

### Tables

**users**
- `id` (INTEGER PRIMARY KEY)
- `email` (TEXT UNIQUE)
- `name` (TEXT)
- `hashed_password` (TEXT)
- `created_at` (TIMESTAMP)

**resumes**
- `id` (INTEGER PRIMARY KEY)
- `user_id` (INTEGER, FK → users.id)
- `candidate_id` (TEXT UNIQUE)
- `resume_data` (TEXT JSON)
- `features` (TEXT JSON)
- `filename` (TEXT)
- `status` (TEXT)
- `created_at` (TIMESTAMP)

**job_descriptions**
- `id` (INTEGER PRIMARY KEY)
- `user_id` (INTEGER, FK → users.id)
- `job_id` (TEXT UNIQUE)
- `job_description` (TEXT)
- `job_requirements` (TEXT JSON)
- `created_at` (TIMESTAMP)

**rankings**
- `id` (INTEGER PRIMARY KEY)
- `user_id` (INTEGER, FK → users.id)
- `job_id` (TEXT)
- `ranking_data` (TEXT JSON)
- `created_at` (TIMESTAMP)

---

## API Endpoints

### Authentication
- `POST /register` - Create new user account
- `POST /login` - Authenticate user
- `POST /forgot-password` - Request password reset

### Resume Processing
- `POST /upload-resume` - Upload and parse resume
  - Uses: ResumeAgent
  - Returns: `{candidate_id, resume_data, features, status}`

### Job Description
- `POST /analyze-job-description` - Analyze job description
  - Uses: JDAgent
  - Returns: `{job_id, job_requirements, status}`

### Ranking
- `POST /rank-candidates` - Rank candidates for a job
  - Uses: MatchAgent, RankingAgent
  - Returns: `{ranked_candidates, total_candidates, status}`

### Candidate Details
- `GET /resume/{candidate_id}` - Get resume details
- `GET /resume/{candidate_id}/features` - Get feature vectors

### Explainability (Optional)
- `POST /shap-report` - Generate SHAP explanation
- `POST /lime-report` - Generate LIME explanation

---

## Frontend Structure

### Component Hierarchy

```
App.jsx
├── Router
    ├── /login → Login.jsx
    ├── /signup → Signup.jsx
    ├── /upload → UploadPage.jsx
    ├── /dashboard → Dashboard.jsx
    ├── /candidate/:id → CandidateProfile.jsx
    └── /explainability → ExplainabilityView.jsx
```

### Key Components

**UploadPage.jsx**:
- Handles multiple file uploads
- Job description input
- Calls `/analyze-job-description` then `/upload-resume` for each file
- Stores candidate_ids in sessionStorage
- Navigates to Dashboard

**Dashboard.jsx**:
- Displays ranked candidates
- Calls `/rank-candidates` with candidate_ids and job_requirements
- Shows: Rank, Name, Filename, Scores, Skills
- Search/filter functionality

**CandidateProfile.jsx**:
- Detailed candidate view
- Shows all extracted information
- Optional: LIME/SHAP explanations

---

## Key Design Decisions

### 1. Multi-Agent Architecture
- **Why**: Modular, testable, extensible
- **Benefit**: Each agent has single responsibility

### 2. Hybrid Scoring (ML + CSP)
- **Why**: ML provides predictive power, CSP ensures eligibility
- **Benefit**: Balances AI predictions with rule-based constraints

### 3. Direct Sorting vs Best-First Search
- **Why**: Direct sorting is faster and uses exact scores
- **Benefit**: Simpler, more accurate ranking

### 4. Strict Scoring for Non-Matches
- **Why**: Prevents false positives
- **Benefit**: Only truly matching candidates get high scores

### 5. Name Extraction from Resumes
- **Why**: Better UX than candidate IDs
- **Benefit**: Users see actual names in rankings

---

## Dependencies & Technologies

### Backend
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **SQLite**: Database
- **scikit-learn**: ML models
- **NLTK**: Text processing
- **PyPDF2**: PDF parsing
- **bcrypt**: Password hashing
- **pydantic**: Data validation

### Frontend
- **React**: UI framework
- **Vite**: Build tool
- **React Router**: Routing
- **Axios**: HTTP client

---

## File Relationships Summary

```
main.py
  └─> routes.py
      ├─> ResumeAgent → resume_parser.py + feature_extractor.py
      ├─> JDAgent → jd_agent.py
      ├─> MatchAgent → predictor.py + solver.py
      └─> RankingAgent → best_first.py
          └─> Uses final_score from MatchAgent
```

**Key Dependencies**:
- `routes.py` orchestrates all agents
- `MatchAgent` depends on `MLPredictor` and `CSPSolver`
- `ResumeAgent` depends on `ResumeParser` and `FeatureExtractor`
- All agents inherit from `BaseAgent`
- `database.py` provides data persistence
- Frontend calls backend via REST API

---

## Extension Points

1. **Add New ML Models**: Extend `MLModelTrainer` in `ml/models.py`
2. **Add New Constraints**: Extend `CSPSolver` in `csp/solver.py`
3. **Add New Agents**: Inherit from `BaseAgent` in `agents/base_agent.py`
4. **Add New Features**: Extend `FeatureExtractor` in `utils/feature_extractor.py`
5. **Add New Search Algorithms**: Implement in `search/` directory

---

## Testing

Test files in `backend/tests/`:
- `test_resume_parser.py`: Resume parsing tests
- `test_csp.py`: CSP solver tests
- `test_agents.py`: Agent functionality tests
- `test_search.py`: Search algorithm tests
- `test_integration.py`: End-to-end tests

---

This architecture provides a scalable, maintainable, and extensible foundation for the recruitment system.


