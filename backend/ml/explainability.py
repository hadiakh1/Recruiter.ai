"""
Explainability Module
SHAP and LIME for model interpretability
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional imports for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    print("Warning: SHAP not available. SHAP explanations will be disabled.")

try:
    from lime import lime_tabular
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime_tabular = None
    LimeTextExplainer = None
    print("Warning: LIME not available. LIME explanations will be disabled.")

class ExplainabilityModule:
    """Module for generating SHAP and LIME explanations"""
    
    def __init__(self, model=None, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or []
        self.explainer_shap = None
        self.explainer_lime = None
    
    def setup_shap_explainer(self, X_train: np.ndarray, model, model_type: str = 'tree'):
        """Setup SHAP explainer"""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not available. Please install it or use Python 3.11/3.12.")
        if model_type == 'tree':
            self.explainer_shap = shap.TreeExplainer(model)
        elif model_type == 'linear':
            self.explainer_shap = shap.LinearExplainer(model, X_train)
        else:
            # Kernel explainer as fallback
            self.explainer_shap = shap.KernelExplainer(model.predict_proba, X_train[:100])
    
    def generate_shap_explanation(self, X: np.ndarray, 
                                  global_explanation: bool = True) -> Dict:
        """
        Generate SHAP explanations
        Returns both global and local explanations
        """
        if self.explainer_shap is None:
            raise ValueError("SHAP explainer not set up. Call setup_shap_explainer first.")
        
        shap_values = self.explainer_shap.shap_values(X)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        explanations = {
            'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            'base_value': float(self.explainer_shap.expected_value) if hasattr(self.explainer_shap, 'expected_value') else 0.0
        }
        
        if global_explanation:
            # Global feature importance
            if isinstance(shap_values, np.ndarray):
                feature_importance = np.abs(shap_values).mean(axis=0)
                explanations['global_importance'] = {
                    'features': self.feature_names[:len(feature_importance)] if self.feature_names else [f'feature_{i}' for i in range(len(feature_importance))],
                    'importance_scores': feature_importance.tolist()
                }
        
        # Local explanation for first instance
        if isinstance(shap_values, np.ndarray) and len(shap_values) > 0:
            explanations['local_explanation'] = {
                'features': self.feature_names[:len(shap_values[0])] if self.feature_names else [f'feature_{i}' for i in range(len(shap_values[0]))],
                'shap_values': shap_values[0].tolist(),
                'prediction': float(self.model.predict_proba(X[0:1])[0][1]) if hasattr(self.model, 'predict_proba') else 0.0
            }
        
        return explanations
    
    def setup_lime_explainer(self, X_train: np.ndarray, 
                            feature_names: Optional[List[str]] = None,
                            mode: str = 'tabular'):
        """Setup LIME explainer"""
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not available. Please install it or use Python 3.11/3.12.")
        if mode == 'tabular':
            self.explainer_lime = lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names or self.feature_names,
                mode='classification',
                discretize_continuous=True
            )
        else:
            self.explainer_lime = LimeTextExplainer(class_names=['Not Fit', 'Fit'])
    
    def generate_lime_explanation(self, X: np.ndarray, 
                                  instance_idx: int = 0,
                                  num_features: int = 10) -> Dict:
        """
        Generate LIME explanation for a specific instance
        """
        if self.explainer_lime is None:
            raise ValueError("LIME explainer not set up. Call setup_lime_explainer first.")
        
        instance = X[instance_idx:instance_idx+1]
        
        # Generate explanation
        explanation = self.explainer_lime.explain_instance(
            instance[0],
            self.model.predict_proba,
            num_features=num_features
        )
        
        # Extract explanation data
        exp_list = explanation.as_list()
        
        explanation_data = {
            'instance_idx': instance_idx,
            'prediction': float(self.model.predict_proba(instance)[0][1]) if hasattr(self.model, 'predict_proba') else 0.0,
            'explanation': [
                {
                    'feature': item[0],
                    'weight': float(item[1])
                }
                for item in exp_list
            ],
            'positive_features': [item[0] for item in exp_list if item[1] > 0],
            'negative_features': [item[0] for item in exp_list if item[1] < 0]
        }
        
        return explanation_data
    
    def generate_combined_explanation(self, X: np.ndarray, 
                                     candidate_data: Dict,
                                     instance_idx: int = 0) -> Dict:
        """Generate combined SHAP and LIME explanation with human-readable report"""
        explanations = {
            'candidate_id': candidate_data.get('candidate_id', 'unknown'),
            'shap': None,
            'lime': None,
            'human_readable': {}
        }
        
        try:
            # SHAP explanation
            shap_exp = self.generate_shap_explanation(X[instance_idx:instance_idx+1], global_explanation=False)
            explanations['shap'] = shap_exp.get('local_explanation', {})
        except Exception as e:
            explanations['shap_error'] = str(e)
        
        try:
            # LIME explanation
            lime_exp = self.generate_lime_explanation(X, instance_idx=instance_idx)
            explanations['lime'] = lime_exp
        except Exception as e:
            explanations['lime_error'] = str(e)
        
        # Generate human-readable report
        explanations['human_readable'] = self._generate_human_readable_report(
            candidate_data, explanations
        )
        
        return explanations
    
    def _generate_human_readable_report(self, candidate_data: Dict, 
                                       explanations: Dict) -> Dict:
        """Generate human-readable explanation report"""
        report = {
            'summary': '',
            'key_factors': [],
            'skill_match': {},
            'missing_skills': [],
            'recommendations': []
        }
        
        # Extract key information
        skills = candidate_data.get('skills', [])
        experience = candidate_data.get('experience', {}).get('years', 0)
        
        # Summary
        report['summary'] = f"Candidate has {len(skills)} skills and {experience} years of experience."
        
        # Key factors from LIME
        if explanations.get('lime') and explanations['lime'].get('explanation'):
            top_factors = sorted(
                explanations['lime']['explanation'],
                key=lambda x: abs(x['weight']),
                reverse=True
            )[:5]
            
            report['key_factors'] = [
                {
                    'factor': factor['feature'],
                    'impact': 'positive' if factor['weight'] > 0 else 'negative',
                    'magnitude': abs(factor['weight'])
                }
                for factor in top_factors
            ]
        
        # Skill match
        report['skill_match'] = {
            'total_skills': len(skills),
            'skills_list': skills[:10]  # Top 10
        }
        
        # Recommendations
        if experience < 2:
            report['recommendations'].append("Consider candidates with more experience")
        if len(skills) < 5:
            report['recommendations'].append("Candidate may benefit from additional skill development")
        
        return report





