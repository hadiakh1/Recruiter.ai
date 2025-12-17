"""
REST API Routes
All endpoints for the recruitment system
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional, Dict
import os
import sys
import numpy as np
import tempfile
import uuid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.resume_agent import ResumeAgent
from agents.jd_agent import JDAgent
from agents.match_agent import MatchAgent
from agents.ranking_agent import RankingAgent
from agents.base_agent import SharedMemory
from ml.predictor import MLPredictor
from ml.explainability import ExplainabilityModule
from rl.q_learning import RLAdapter
from database import (
    init_db, create_user, get_user_by_email, get_user_by_id,
    save_resume, get_resume, get_user_resumes,
    save_job_description, get_job_description, save_ranking
)
from pydantic import BaseModel
from datetime import datetime, timedelta
try:
    from jose import JWTError, jwt
except ImportError:
    # Fallback if python-jose not available
    import jwt
    JWTError = Exception

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    try:
        from passlib.context import CryptContext
        BCRYPT_AVAILABLE = False
        PASSLIB_AVAILABLE = True
    except ImportError:
        raise ImportError("bcrypt or passlib is required. Please install: pip install bcrypt or pip install passlib[bcrypt]")
import secrets
import hashlib

router = APIRouter()
security = HTTPBearer()

# Authentication setup - use bcrypt directly to avoid passlib initialization issues
if BCRYPT_AVAILABLE:
    # Use bcrypt directly
    def hash_password(password: str) -> str:
        """
        Hash password with bcrypt, handling passwords longer than 72 bytes.
        For passwords > 72 bytes, pre-hash with SHA256 first.
        """
        # Convert to bytes
        password_bytes = password.encode('utf-8') if isinstance(password, str) else password
        
        # If longer than 72 bytes, hash with SHA256 first
        if len(password_bytes) > 72:
            # SHA256 produces 32 bytes, hexdigest is 64 bytes (under 72 limit)
            sha256_hash = hashlib.sha256(password_bytes).hexdigest()
            password_to_hash = sha256_hash.encode('utf-8')
        else:
            password_to_hash = password_bytes
        
        # Ensure it's exactly 72 bytes or less
        if len(password_to_hash) > 72:
            password_to_hash = password_to_hash[:72]
        
        # Hash with bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password_to_hash, salt).decode('utf-8')
    
    def verify_password(password: str, hashed: str) -> bool:
        """
        Verify password against hash, handling passwords longer than 72 bytes.
        """
        # Convert to bytes
        password_bytes = password.encode('utf-8') if isinstance(password, str) else password
        
        # If longer than 72 bytes, hash with SHA256 first
        if len(password_bytes) > 72:
            sha256_hash = hashlib.sha256(password_bytes).hexdigest()
            password_to_verify = sha256_hash.encode('utf-8')
        else:
            password_to_verify = password_bytes
        
        # Ensure it's exactly 72 bytes or less
        if len(password_to_verify) > 72:
            password_to_verify = password_to_verify[:72]
        
        # Verify with bcrypt
        hashed_bytes = hashed.encode('utf-8') if isinstance(hashed, str) else hashed
        return bcrypt.checkpw(password_to_verify, hashed_bytes)
else:
    # Fallback to passlib
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def hash_password(password: str) -> str:
        """Hash password with passlib/bcrypt"""
        password_bytes = password.encode('utf-8') if isinstance(password, str) else password
        if len(password_bytes) > 72:
            sha256_hash = hashlib.sha256(password_bytes).hexdigest()
            password_to_hash = sha256_hash
        else:
            password_to_hash = password
        return pwd_context.hash(password_to_hash)
    
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password with passlib/bcrypt"""
        password_bytes = password.encode('utf-8') if isinstance(password, str) else password
        if len(password_bytes) > 72:
            sha256_hash = hashlib.sha256(password_bytes).hexdigest()
            password_to_verify = sha256_hash
        else:
            password_to_verify = password
        return pwd_context.verify(password_to_verify, hashed)
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize database
init_db()

# In-memory cache for active sessions (candidates_store for backward compatibility)
candidates_store = {}
job_requirements_store = {}

# Initialize shared memory and agents (with error handling)
shared_memory = SharedMemory()
try:
    resume_agent = ResumeAgent(shared_memory)
    jd_agent = JDAgent(shared_memory)
    match_agent = MatchAgent(shared_memory)
    ranking_agent = RankingAgent(shared_memory)
except Exception as e:
    print(f"Warning: Error initializing agents: {e}")
    # Create dummy agents that will fail gracefully
    resume_agent = None
    jd_agent = None
    match_agent = None
    ranking_agent = None

# Initialize ML predictor and explainability (with error handling)
try:
    ml_predictor = MLPredictor()
    explainability_module = ExplainabilityModule()
except Exception as e:
    print(f"Warning: Error initializing ML modules: {e}")
    ml_predictor = None
    explainability_module = None

# Initialize RL adapter (with error handling)
try:
    rl_adapter = RLAdapter()
except Exception as e:
    print(f"Warning: Error initializing RL adapter: {e}")
    rl_adapter = None

# Storage for candidates and job requirements (in production, use database)
candidates_store = {}
job_requirements_store = {}

# Pydantic models for request/response
class JobDescriptionRequest(BaseModel):
    job_description: str

class CandidateEvaluationRequest(BaseModel):
    candidate_id: str
    job_requirements: Dict

class RankingRequest(BaseModel):
    candidate_ids: List[str]
    job_requirements: Dict

class FeedbackRequest(BaseModel):
    ranking: List[Dict]
    ml_scores: List[float]
    selected_candidates: List[str]
    rejected_candidates: List[str]

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ForgotPasswordRequest(BaseModel):
    email: str

# Authentication helper functions
def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[Dict]:
    """Get current user from JWT token (optional for some endpoints)"""
    if not authorization:
        return None
    
    try:
        # Extract token from "Bearer <token>"
        if authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
        else:
            token = authorization
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        
        user = get_user_by_email(email)
        return user
    except (JWTError, Exception):
        return None

def verify_token(authorization: Optional[str] = Header(None)) -> Dict:
    """Verify JWT token and return user info (required)"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    try:
        # Extract token from "Bearer <token>"
        if authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
        else:
            token = authorization
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
        user = get_user_by_email(email)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")

# Authentication APIs
@router.post("/auth/signup")
async def signup(request: SignupRequest):
    """User signup with database"""
    try:
        email = request.email.lower().strip()
        password = request.password
        name = request.name.strip()
        
        # Validate input
        if not email or not password or not name:
            raise HTTPException(status_code=400, detail="All fields are required")
        
        if len(password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        
        # Check if user exists
        existing_user = get_user_by_email(email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Hash password and create user (handles passwords > 72 bytes)
        hashed_password = hash_password(password)
        user_id = create_user(email, name, hashed_password)
        
        if user_id is None:
            raise HTTPException(status_code=500, detail="Failed to create user account")
        
        # Create token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        expire = datetime.utcnow() + access_token_expires
        access_token = jwt.encode(
            {"sub": email, "exp": expire, "user_id": user_id},
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        
        return {
            "token": access_token,
            "user": {"email": email, "name": name, "id": user_id}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating account: {str(e)}")

@router.post("/auth/login")
async def login(request: LoginRequest):
    """User login with database"""
    try:
        email = request.email.lower().strip()
        password = request.password
        
        # Get user from database
        user = get_user_by_email(email)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Verify password (handles passwords > 72 bytes)
        if not verify_password(password, user['hashed_password']):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Create token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        expire = datetime.utcnow() + access_token_expires
        access_token = jwt.encode(
            {"sub": email, "exp": expire, "user_id": user['id']},
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        
        return {
            "token": access_token,
            "user": {"email": user['email'], "name": user['name'], "id": user['id']}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging in: {str(e)}")

@router.post("/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    """Forgot password (simplified - just returns success)"""
    try:
        email = request.email.lower().strip()
        user = get_user_by_email(email)
        # Don't reveal if email exists for security
        # In production, send email with reset link
        return {"message": "If email exists, reset link has been sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing forgot password request: {str(e)}")

# Resume APIs
@router.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...), user: Dict = Depends(verify_token)):
    """Upload and parse resume (linked to user account)"""
    try:
        if resume_agent is None:
            raise HTTPException(status_code=500, detail="Resume agent not initialized. Please check backend logs.")
        
        user_id = user['id']
        
        # Save uploaded file temporarily
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['pdf', 'txt', 'docx']:
            raise HTTPException(status_code=400, detail="Only PDF, TXT, and DOCX files are supported")
        
        file_type = 'pdf' if file_ext == 'pdf' else ('docx' if file_ext == 'docx' else 'txt')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Process with Resume Agent
        result = resume_agent.process({
            'file_path': tmp_path,
            'file_type': file_type
        })
        
        # Check if processing was successful
        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result.get('error', 'Failed to process resume'))
        
        # Generate unique candidate ID
        candidate_id = f"candidate_{user_id}_{uuid.uuid4().hex[:8]}"
        
        # Store in database
        resume_data = result.get('resume_data', {})
        features = result.get('features', {})
        filename = file.filename  # Store original filename
        save_resume(user_id, candidate_id, resume_data, features, filename)
        
        # Also store in memory cache for backward compatibility
        candidates_store[candidate_id] = {
            'resume_data': resume_data,
            'features': features,
            'filename': filename,
            'status': result.get('status', 'success'),
            'user_id': user_id
        }
        
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass  # Ignore cleanup errors
        
        return {
            'candidate_id': candidate_id,
            'resume_data': resume_data,
            'features': features,
            'status': result.get('status', 'success')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@router.post("/parse-resume")
async def parse_resume(file_path: str, file_type: str = "pdf"):
    """Parse resume from file path"""
    try:
        result = resume_agent.process({
            'file_path': file_path,
            'file_type': file_type
        })
        
        candidate_id = f"candidate_{len(candidates_store)}"
        candidates_store[candidate_id] = result
        
        return {
            'candidate_id': candidate_id,
            'resume_data': result.get('resume_data', {}),
            'status': result.get('status', 'success')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get-resume-features/{candidate_id}")
async def get_resume_features(candidate_id: str, user: Dict = Depends(verify_token)):
    """Get features for a parsed resume"""
    # Try database first
    resume = get_resume(candidate_id)
    if resume:
        if resume['user_id'] != user['id']:
            raise HTTPException(status_code=403, detail="Access denied")
        return {
            'candidate_id': candidate_id,
            'features': resume.get('features', {})
        }
    
    # Fallback to memory cache
    if candidate_id not in candidates_store:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    candidate = candidates_store[candidate_id]
    if candidate.get('user_id') != user['id']:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        'candidate_id': candidate_id,
        'features': candidate.get('features', {})
    }

# ML APIs
@router.post("/predict-fit")
async def predict_fit(candidate_id: str, job_requirements: Dict):
    """Predict job fit score for a candidate"""
    try:
        if candidate_id not in candidates_store:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        candidate = candidates_store[candidate_id]
        features = candidate.get('features', {})
        
        # Get combined features
        if 'combined' in features and len(features['combined']) > 0:
            feature_vector = np.array(features['combined'])
        elif 'structured' in features and len(features['structured']) > 0:
            feature_vector = np.array(features['structured'])
        else:
            raise HTTPException(status_code=400, detail="No features available")
        
        # Predict
        score = ml_predictor.predict_fit_score(feature_vector)
        
        return {
            'candidate_id': candidate_id,
            'fit_score': float(score),
            'status': 'success'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/candidate-embedding")
async def candidate_embedding(text: str):
    """Generate embedding for candidate text"""
    try:
        embedding = ml_predictor.generate_embedding(text)
        return {
            'embedding': embedding.tolist(),
            'dimension': len(embedding)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Search + CSP APIs
@router.post("/evaluate-candidate")
async def evaluate_candidate(request: CandidateEvaluationRequest):
    """Evaluate candidate using CSP"""
    try:
        if request.candidate_id not in candidates_store:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        candidate = candidates_store[request.candidate_id]
        resume_data = candidate.get('resume_data', {})
        
        # Use Match Agent for evaluation
        result = match_agent.process({
            'candidate_data': resume_data,
            'job_requirements': request.job_requirements,
            'features': candidate.get('features', {})
        })
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rank-candidates")
async def rank_candidates(request: RankingRequest, user: Dict = Depends(verify_token)):
    """Rank candidates using Best-First Search with ML, CSP, and Multi-Agent System"""
    try:
        if not request.candidate_ids or len(request.candidate_ids) == 0:
            raise HTTPException(status_code=400, detail="No candidate IDs provided")
        
        if not request.job_requirements:
            raise HTTPException(status_code=400, detail="Job requirements not provided")
        
        user_id = user['id']
        candidates = []
        ml_scores = []
        csp_scores = []
        errors = []
        
        if match_agent is None:
            raise HTTPException(status_code=500, detail="Match agent not initialized. Please check backend logs.")
        
        if ranking_agent is None:
            raise HTTPException(status_code=500, detail="Ranking agent not initialized. Please check backend logs.")
        
        # Process each candidate using Match-Agent (combines ML + CSP)
        for candidate_id in request.candidate_ids:
            try:
                # Try database first
                resume = get_resume(candidate_id)
                filename = None
                if resume:
                    if resume['user_id'] != user_id:
                        errors.append(f"Access denied for candidate {candidate_id}")
                        continue
                    resume_data = resume['resume_data']
                    features = resume.get('features', {})
                    filename = resume.get('filename')  # Get filename from database
                elif candidate_id in candidates_store:
                    candidate = candidates_store[candidate_id]
                    if candidate.get('user_id') != user_id:
                        errors.append(f"Access denied for candidate {candidate_id}")
                        continue
                    resume_data = candidate.get('resume_data', {})
                    features = candidate.get('features', {})
                    filename = candidate.get('filename')  # Get filename from memory cache
                else:
                    errors.append(f"Candidate {candidate_id} not found")
                    continue
                
                # Use Match-Agent to get both ML and CSP scores
                match_result = match_agent.process({
                    'candidate_data': resume_data,
                    'job_requirements': request.job_requirements,
                    'features': features
                })
                
                if match_result.get('status') == 'error':
                    errors.append(f"Error processing {candidate_id}: {match_result.get('error', 'Unknown error')}")
                    continue
                
                ml_score = match_result.get('ml_score', 0.0)
                csp_score = match_result.get('csp_score', 0.0)
                final_score = match_result.get('final_score', 0.0)  # Get the combined final score
                
                # Extract name from resume data
                candidate_name = resume_data.get('name', 'Unknown')
                
                candidates.append({
                    'candidate_id': candidate_id,
                    'resume_data': resume_data,
                    'candidate_data': resume_data,  # For compatibility
                    'name': candidate_name,  # Add extracted name
                    'filename': filename,  # Add original filename
                    'final_score': final_score  # Include final score for ranking
                })
                ml_scores.append(float(ml_score))
                csp_scores.append(float(csp_score))
                
            except Exception as e:
                errors.append(f"Error processing {candidate_id}: {str(e)}")
                continue
        
        if len(candidates) == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"No candidates could be processed. Errors: {'; '.join(errors)}"
            )
        
        # Use Ranking-Agent with Best-First Search to rank candidates
        result = ranking_agent.process({
            'candidates': candidates,
            'ml_scores': ml_scores,
            'csp_scores': csp_scores,
            'job_requirements': request.job_requirements
        })
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result.get('error', 'Ranking failed'))
        
        # Save ranking to database if job_id is provided
        ranked_candidates = result.get('ranked_candidates', [])
        if ranked_candidates and 'job_id' in request.job_requirements:
            try:
                save_ranking(user_id, request.job_requirements.get('job_id'), ranked_candidates)
            except:
                pass  # Don't fail if saving ranking fails
        
        # Add warnings if there were errors
        if errors:
            result['warnings'] = errors
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ranking candidates: {str(e)}")

# RL APIs
@router.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Provide HR feedback for RL training"""
    try:
        rl_adapter.train_on_feedback(
            ranking=request.ranking,
            ml_scores=request.ml_scores,
            hr_feedback={
                'selected_candidates': request.selected_candidates,
                'rejected_candidates': request.rejected_candidates
            }
        )
        
        return {
            'status': 'success',
            'message': 'Feedback received and model updated'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/improve-ranking")
async def improve_ranking(ranking: List[Dict], ml_scores: List[float]):
    """Improve ranking using RL"""
    try:
        improved_ranking = rl_adapter.improve_ranking(ranking, ml_scores)
        return {
            'original_ranking': ranking,
            'improved_ranking': improved_ranking,
            'status': 'success'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Explainability APIs
@router.post("/shap-report")
async def shap_report(candidate_id: str, X_train: Optional[List[List[float]]] = None):
    """Generate SHAP explanation report"""
    try:
        if candidate_id not in candidates_store:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        candidate = candidates_store[candidate_id]
        features = candidate.get('features', {})
        
        if 'combined' not in features:
            raise HTTPException(status_code=400, detail="Features not available")
        
        X = np.array([features['combined']])
        
        # Setup explainer if model available
        if ml_predictor.is_loaded and hasattr(ml_predictor.trainer, 'models'):
            model = ml_predictor.trainer.models.get(ml_predictor.model_name)
            if model:
                explainability_module.model = model
                if X_train:
                    explainability_module.setup_shap_explainer(
                        np.array(X_train), model, model_type='tree'
                    )
        
        explanation = explainability_module.generate_shap_explanation(X)
        
        return {
            'candidate_id': candidate_id,
            'explanation': explanation,
            'status': 'success'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lime-report")
async def lime_report(candidate_id: str, X_train: Optional[List[List[float]]] = None):
    """Generate LIME explanation report"""
    try:
        if candidate_id not in candidates_store:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        candidate = candidates_store[candidate_id]
        features = candidate.get('features', {})
        resume_data = candidate.get('resume_data', {})
        
        if 'combined' not in features:
            raise HTTPException(status_code=400, detail="Features not available")
        
        X = np.array([features['combined']])
        
        # Setup explainer if model available
        if ml_predictor.is_loaded and hasattr(ml_predictor.trainer, 'models'):
            model = ml_predictor.trainer.models.get(ml_predictor.model_name)
            if model:
                explainability_module.model = model
                if X_train:
                    explainability_module.setup_lime_explainer(
                        np.array(X_train), mode='tabular'
                    )
        
        explanation = explainability_module.generate_combined_explanation(
            X, {'candidate_id': candidate_id, **resume_data}
        )
        
        return {
            'candidate_id': candidate_id,
            'explanation': explanation,
            'status': 'success'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Job Description API
@router.post("/analyze-job-description")
async def analyze_job_description(request: JobDescriptionRequest, user: Dict = Depends(verify_token)):
    """Analyze job description and extract requirements (linked to user account)"""
    try:
        if jd_agent is None:
            raise HTTPException(status_code=500, detail="JD agent not initialized. Please check backend logs.")
        
        user_id = user['id']
        
        result = jd_agent.process({
            'job_description': request.job_description
        })
        
        job_requirements = result.get('job_requirements', {})
        
        # Generate unique job ID
        job_id = f"job_{user_id}_{uuid.uuid4().hex[:8]}"
        
        # Save to database
        save_job_description(user_id, job_id, request.job_description, job_requirements)
        
        # Also store in memory cache
        job_requirements_store[job_id] = job_requirements
        
        return {
            'job_id': job_id,
            'job_requirements': job_requirements,
            'status': result.get('status', 'success')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing job description: {str(e)}")





