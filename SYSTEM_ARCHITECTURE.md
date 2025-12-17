# System Architecture & AI Components

## Complete AI Integration

This system implements a **Hybrid AI Recruitment Platform** that combines multiple AI paradigms:

### 1. **Multi-Agent System** 🤖

Four specialized agents work together:

#### **Resume-Agent**
- **Purpose**: Extract information from resumes
- **Process**:
  1. Parses PDF/TXT files
  2. Extracts: skills, experience, education, certifications, projects
  3. Generates features: TF-IDF, BERT embeddings, structured features
- **Output**: Structured resume data + feature vectors

#### **JD-Agent (Job Description Agent)**
- **Purpose**: Analyze job descriptions
- **Process**:
  1. Extracts required skills (mandatory)
  2. Extracts optional/preferred skills
  3. Identifies experience requirements
  4. Extracts degree and certification requirements
- **Output**: Structured job requirements

#### **Match-Agent**
- **Purpose**: Combine ML and CSP scores
- **Process**:
  1. Gets ML score from feature vectors (job fit prediction)
  2. Gets CSP score (eligibility based on constraints)
  3. Combines scores: `final_score = (ML * 0.6) + (CSP * 0.4)`
- **Output**: ML score, CSP score, final job-fit score

#### **Ranking-Agent**
- **Purpose**: Produce final ranked list
- **Process**:
  1. Uses Best-First Search algorithm
  2. Applies heuristic function with ML/CSP scores
  3. Produces ranked candidate list
- **Output**: Ranked candidates with scores

### 2. **Machine Learning Models** 🧠

#### **Supervised Learning**
- **Models Available**:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Multi-Layer Perceptron (MLP)
  - BERT-based Classifier

#### **Feature Engineering**
- **TF-IDF**: Text-based features (1000 dimensions)
- **BERT Embeddings**: Semantic understanding (384 dimensions)
- **Structured Features**: Skills count, experience, education level, etc. (10 dimensions)

#### **Default Scoring** (when model not trained)
- Uses feature analysis and normalization
- Provides baseline job-fit scores
- Works immediately without training data

### 3. **Constraint Satisfaction Problem (CSP)** ✅

#### **Hard Constraints** (Must be satisfied)
- **Mandatory Skills**: All required skills must be present
- **Required Experience**: Minimum years of experience
- **Required Degree**: Specific degree requirement
- **Required Certifications**: Mandatory certifications

#### **Soft Constraints** (Preferred)
- **Optional Skills**: Bonus points for additional skills
- **Preferred Experience**: Extra credit for more experience

#### **Scoring**
- **Eligibility Score**: 0.0 (failed hard constraints) to 1.0 (all satisfied)
- **Penalty System**: Reduces score for missing requirements

### 4. **Best-First Search Algorithm** 🔍

#### **Heuristic Function**
```
h(n) = ML_Fit_Score(n) * Skill_Match(n) * (1 - Eligibility_Penalty(n))
```

Where:
- **ML_Fit_Score**: Machine learning prediction (0-1)
- **Skill_Match**: Ratio of matched required/optional skills (0-1)
- **Eligibility_Penalty**: Inverse of CSP score (0-1)

#### **Search Process**
1. Build graph of candidates
2. Calculate heuristic for each candidate
3. Use priority queue (max-heap) to select best candidates
4. Rank by heuristic value (highest first)

### 5. **Reinforcement Learning** (Optional) 🎯

#### **Q-Learning Implementation**
- **State**: Current ranking + ML scores
- **Actions**: Promote, Demote, Reject, Keep
- **Reward**: Based on HR feedback
- **Policy**: Learns optimal ranking adjustments

#### **Integration**
- Can refine rankings based on HR selections
- Learns from feedback over time
- Improves ranking accuracy

### 6. **Explainable AI** 📊

#### **SHAP (SHapley Additive exPlanations)**
- **Global Explanations**: Feature importance across all candidates
- **Local Explanations**: Why specific candidate was ranked
- **Feature Contributions**: Shows which features matter most

#### **LIME (Local Interpretable Model-agnostic Explanations)**
- **Per-Candidate Explanations**: Why this candidate got this score
- **Feature Weights**: Positive/negative contributions
- **Human-Readable Reports**: Plain language explanations

## Complete Workflow

### Step 1: Upload & Parse
1. User uploads **multiple resumes** (PDF/TXT)
2. User enters **job description**
3. **Resume-Agent** processes each resume:
   - Extracts text
   - Parses structured data
   - Generates feature vectors (TF-IDF + BERT + Structured)

### Step 2: Job Analysis
4. **JD-Agent** analyzes job description:
   - Extracts required skills
   - Identifies experience requirements
   - Determines degree/certification needs

### Step 3: Candidate Evaluation
5. **Match-Agent** evaluates each candidate:
   - **ML Score**: Predicts job fit from features
   - **CSP Score**: Checks eligibility against constraints
   - **Final Score**: Combines both scores

### Step 4: Ranking
6. **Ranking-Agent** uses Best-First Search:
   - Builds candidate graph
   - Calculates heuristic for each candidate
   - Ranks by heuristic value (highest = best fit)

### Step 5: Display Results
7. Dashboard shows ranked candidates with:
   - Rank number
   - ML score, CSP score, Heuristic value
   - Skills, experience, education
   - Click to see detailed profile

### Step 6: Explainability (Optional)
8. User can view explanations:
   - SHAP: Feature importance
   - LIME: Per-candidate reasoning
   - Human-readable reports

## Error Handling

### Frontend
- ✅ File upload errors (invalid format, size limits)
- ✅ Network errors (connection issues)
- ✅ Validation errors (missing fields)
- ✅ Processing errors (with retry options)

### Backend
- ✅ Resume parsing errors (corrupted files)
- ✅ Feature extraction errors (fallback to structured features)
- ✅ ML prediction errors (fallback to default scoring)
- ✅ CSP evaluation errors (graceful degradation)
- ✅ Ranking errors (partial results with warnings)

## Key Features

✅ **Multiple Resume Upload**: Upload and process many resumes at once  
✅ **Real-time Processing**: Shows progress during upload and ranking  
✅ **Comprehensive Ranking**: Uses ML + CSP + Search algorithms  
✅ **Explainable Results**: SHAP/LIME explanations available  
✅ **Error Recovery**: Graceful handling of failures  
✅ **Professional UI**: Black & white theme, clean design  
✅ **Authentication**: Secure login/signup system  

## Technical Stack

**Backend**:
- FastAPI (REST API)
- scikit-learn (ML models)
- SentenceTransformers (BERT embeddings)
- NetworkX (Graph algorithms)
- SHAP & LIME (Explainability)

**Frontend**:
- React 18
- Axios (HTTP client)
- React Router (Navigation)
- Recharts (Visualizations)

## Performance

- **Upload**: ~2-5 seconds per resume
- **Ranking**: ~5-10 seconds for 10-20 candidates
- **Scalability**: Can handle 50+ resumes per job posting

## Next Steps for Production

1. Train ML models on real data
2. Add database (PostgreSQL/MongoDB)
3. Implement caching for faster responses
4. Add batch processing for large uploads
5. Deploy with Docker/Kubernetes

