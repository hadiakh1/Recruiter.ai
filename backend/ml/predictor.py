"""
ML Prediction API Module
Provides prediction services for job fit scores
"""
import numpy as np
from typing import Dict, List, Optional
from .models import MLModelTrainer

class MLPredictor:
    """ML Prediction service"""
    
    def __init__(self, model_name: str = 'random_forest'):
        self.trainer = MLModelTrainer()
        self.model_name = model_name
        self.is_loaded = False
    
    def load_model(self, model_path: str, scaler_path: Optional[str] = None):
        """Load a pre-trained model"""
        self.trainer.load_model(self.model_name, model_path, scaler_path)
        self.is_loaded = True
    
    def _default_score(self, features: np.ndarray, job_requirements: Optional[Dict] = None, candidate_data: Optional[Dict] = None) -> float:
        """Default scoring method when model is not loaded - based on feature analysis and job requirements"""
        if len(features) == 0:
            return 0.0
        
        feature_array = features.flatten()
        
        if len(feature_array) == 0:
            return 0.0
        
        # If job requirements are provided, use them for strict scoring
        if job_requirements and candidate_data:
            # Check if candidate has required skills
            required_skills = job_requirements.get('required_skills', [])
            optional_skills = job_requirements.get('optional_skills', [])
            candidate_skills = [s.lower().strip() for s in candidate_data.get('skills', [])]
            
            # Penalty if candidate has no skills at all
            if len(candidate_skills) == 0:
                return 0.0  # No skills = 0 ML score
            
            if required_skills:
                # Count how many required skills the candidate has
                matched_skills = 0
                for req_skill in required_skills:
                    req_skill_lower = req_skill.lower().strip()
                    for cand_skill in candidate_skills:
                        # Check for exact or substring match
                        if req_skill_lower == cand_skill or req_skill_lower in cand_skill or cand_skill in req_skill_lower:
                            matched_skills += 1
                            break
                
                # If candidate doesn't have required skills, give very low score
                skill_match_ratio = matched_skills / len(required_skills) if required_skills else 0.0
                
                # If less than 50% of required skills match, give very low score
                if skill_match_ratio < 0.5:
                    return 0.0  # No match = 0 score
                
                # Scale score based on skill match
                base_score = skill_match_ratio * 0.6  # Max 60% from skills
                
                # Add bonus from structured features (experience, education, etc.)
                structured_features = feature_array[:min(10, len(feature_array))]
                if len(structured_features) > 0:
                    max_val = np.max(np.abs(structured_features))
                    if max_val > 0:
                        normalized = np.clip(structured_features / (max_val + 1e-6), 0, 1)
                        feature_bonus = float(np.mean(normalized)) * 0.4  # Max 40% from features
                        return min(base_score + feature_bonus, 1.0)
                
                return base_score
            elif optional_skills:
                # Only optional skills - check if candidate has any of them
                matched_optional = 0
                for opt_skill in optional_skills:
                    opt_skill_lower = opt_skill.lower().strip()
                    for cand_skill in candidate_skills:
                        if opt_skill_lower == cand_skill or opt_skill_lower in cand_skill or cand_skill in opt_skill_lower:
                            matched_optional += 1
                            break
                
                # Score based on optional skill match (lower than required)
                if matched_optional == 0:
                    return 0.0  # No optional skills match = 0 score
                
                optional_match_ratio = matched_optional / len(optional_skills) if optional_skills else 0.0
                base_score = optional_match_ratio * 0.4  # Max 40% from optional skills
                
                # Add small bonus from features
                structured_features = feature_array[:min(10, len(feature_array))]
                if len(structured_features) > 0:
                    max_val = np.max(np.abs(structured_features))
                    if max_val > 0:
                        normalized = np.clip(structured_features / (max_val + 1e-6), 0, 1)
                        feature_bonus = float(np.mean(normalized)) * 0.2  # Max 20% from features
                        return min(base_score + feature_bonus, 0.6)  # Cap at 60% for optional-only
                
                return base_score
        
        # Fallback: very conservative scoring when no job requirements
        # Only give score if features have meaningful values
        max_val = np.max(np.abs(feature_array))
        if max_val < 0.01:  # Very small values = no match
            return 0.0
        
        # Conservative scoring - cap at 20% for unmatched cases
        normalized = np.clip(feature_array / (max_val + 1e-6), 0, 1)
        score = float(np.mean(normalized)) * 0.2  # Scale down significantly
        return min(max(score, 0.0), 0.2)  # Cap at 20% for unmatched cases
    
    def predict_fit_score(self, features: np.ndarray, job_requirements: Optional[Dict] = None, candidate_data: Optional[Dict] = None) -> float:
        """Predict job fit score (0-1)"""
        # Ensure features are in correct shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        if self.is_loaded:
            try:
                score = self.trainer.predict(self.model_name, features)
                predicted_score = float(score[0]) if isinstance(score, np.ndarray) else float(score)
                
                # Apply strict penalty if candidate doesn't meet requirements
                if job_requirements and candidate_data:
                    required_skills = job_requirements.get('required_skills', [])
                    candidate_skills = [s.lower().strip() for s in candidate_data.get('skills', [])]
                    
                    if required_skills:
                        matched_skills = sum(1 for req_skill in required_skills
                                            for cand_skill in candidate_skills
                                            if req_skill.lower().strip() == cand_skill or 
                                            req_skill.lower().strip() in cand_skill or 
                                            cand_skill in req_skill.lower().strip())
                        skill_match_ratio = matched_skills / len(required_skills)
                        
                        # If less than 50% match, heavily penalize
                        if skill_match_ratio < 0.5:
                            predicted_score = predicted_score * 0.1  # Reduce to 10% of original
                
                return min(max(predicted_score, 0.0), 1.0)
            except Exception as e:
                # Fallback to default if model prediction fails
                print(f"Model prediction failed: {e}, using default scoring")
                return self._default_score(features, job_requirements, candidate_data)
        else:
            # Use default scoring when model is not loaded
            return self._default_score(features, job_requirements, candidate_data)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (using SentenceTransformer)"""
        try:
            from sentence_transformers import SentenceTransformer
            
            if not hasattr(self, 'embedding_model'):
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            return self.embedding_model.encode(text, convert_to_numpy=True)
        except ImportError:
            # Return zero vector if sentence_transformers is not available
            return np.zeros(384)  # Default embedding size
    
    def batch_predict(self, features_list: List[np.ndarray]) -> List[float]:
        """Predict fit scores for multiple candidates"""
        scores = []
        for features in features_list:
            score = self.predict_fit_score(features)
            scores.append(score)
        return scores
