"""
Match Agent
Combines ML predictions and CSP results to compute final job-fit score
"""
from .base_agent import BaseAgent, SharedMemory
from typing import Dict, List
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.predictor import MLPredictor
from csp.solver import CSPSolver

class MatchAgent(BaseAgent):
    """Agent that combines ML and CSP to compute job-fit scores"""
    
    def __init__(self, shared_memory: SharedMemory = None, ml_model_path: str = None):
        super().__init__("match_agent", shared_memory)
        # Use the same default model name as in MLPredictor (rf_regressor)
        self.ml_predictor = MLPredictor()
        self.csp_solver = CSPSolver()
        
        # Load ML model if path provided (optional override)
        if ml_model_path:
            try:
                self.ml_predictor.load_model(ml_model_path)
            except Exception as e:
                # Model not available, will use default heuristic scoring
                print(f"Warning: could not load ML model in MatchAgent: {e}")
    
    def compute_job_fit_score(self, ml_score: float, csp_score: float,
                             ml_weight: float = 0.6, csp_weight: float = 0.4) -> float:
        """Combine ML and CSP scores into final job-fit score"""
        # Weighted combination
        final_score = (ml_score * ml_weight) + (csp_score * csp_weight)
        return min(max(final_score, 0.0), 1.0)  # Clamp to [0, 1]
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process candidate and compute job-fit score
        Input: {'candidate_data': Dict, 'job_requirements': Dict, 'features': Dict}
        Output: {'ml_score': float, 'csp_score': float, 'final_score': float, 'csp_details': Dict}
        """
        self.set_status("processing")
        
        try:
            candidate_data = input_data.get('candidate_data')
            job_requirements = input_data.get('job_requirements')
            features = input_data.get('features', {})
            
            # Get ML score - pass job_requirements and candidate_data for better scoring
            ml_score = 0.0
            if 'combined' in features and len(features['combined']) > 0:
                try:
                    feature_vector = np.array(features['combined'])
                    ml_score = self.ml_predictor.predict_fit_score(feature_vector, job_requirements, candidate_data)
                except:
                    # Fallback: use structured features only
                    if 'structured' in features and len(features['structured']) > 0:
                        feature_vector = np.array(features['structured'])
                        ml_score = self.ml_predictor.predict_fit_score(feature_vector, job_requirements, candidate_data)
            
            # Get CSP score
            csp_result = self.csp_solver.evaluate_candidate(candidate_data, job_requirements)
            csp_score = csp_result.get('eligibility_score', 0.0)
            hard_constraints_satisfied = csp_result.get('hard_constraints_satisfied', False)
            
            # If hard constraints are not satisfied, final score should be very low
            if not hard_constraints_satisfied:
                # Even if ML gives a score, if CSP hard constraints fail, heavily penalize
                final_score = ml_score * 0.1  # Reduce to 10% of ML score
            else:
                # Compute final score normally
                final_score = self.compute_job_fit_score(ml_score, csp_score)
            
            result = {
                'ml_score': float(ml_score),
                'csp_score': float(csp_score),
                'final_score': float(final_score),
                'csp_details': csp_result,
                'status': 'success'
            }
            
            # Publish results
            self.publish("match_result", result)
            
            self.results = result
            self.set_status("completed")
            
            return result
            
        except Exception as e:
            self.set_status("error")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def batch_process(self, candidates: List[Dict], job_requirements: Dict) -> List[Dict]:
        """Process multiple candidates"""
        results = []
        for candidate in candidates:
            input_data = {
                'candidate_data': candidate.get('resume_data', candidate),
                'job_requirements': job_requirements,
                'features': candidate.get('features', {})
            }
            result = self.process(input_data)
            results.append(result)
        return results





