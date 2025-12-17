"""
Resume Agent
Extracts information from resumes and produces features
"""
from .base_agent import BaseAgent, SharedMemory
from typing import Dict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.resume_parser import ResumeParser
from utils.feature_extractor import FeatureExtractor
import numpy as np
import numpy as np

class ResumeAgent(BaseAgent):
    """Agent responsible for resume parsing and feature extraction"""
    
    def __init__(self, shared_memory: SharedMemory = None):
        super().__init__("resume_agent", shared_memory)
        self.parser = ResumeParser()
        self.feature_extractor = FeatureExtractor()
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process resume file and extract features
        Input: {'file_path': str, 'file_type': 'pdf'|'txt'}
        Output: {'resume_data': Dict, 'features': Dict}
        """
        self.set_status("processing")
        
        try:
            file_path = input_data.get('file_path')
            file_type = input_data.get('file_type', 'pdf')
            
            if not file_path:
                raise ValueError("File path is required")
            
            # Parse resume
            resume_data = self.parser.parse_resume(file_path, file_type)
            
            if not resume_data:
                raise ValueError("Failed to parse resume data")
            
            # Extract features
            try:
                features = self.feature_extractor.extract_all_features(resume_data, use_embeddings=True)
            except Exception as e:
                print(f"Feature extraction error: {e}, using structured features only")
                # Fallback to structured features only
                features = {
                    'structured': self.feature_extractor.extract_structured_features(resume_data),
                    'combined': self.feature_extractor.extract_structured_features(resume_data)
                }
            
            # Convert numpy arrays to lists for JSON serialization
            result = {
                'resume_data': resume_data,
                'features': {
                    'tfidf': features.get('tfidf', np.array([])).tolist() if 'tfidf' in features else [],
                    'embedding': features.get('embedding', np.array([])).tolist() if 'embedding' in features else [],
                    'structured': features.get('structured', np.array([])).tolist() if 'structured' in features else [],
                    'combined': features.get('combined', np.array([])).tolist() if 'combined' in features else []
                },
                'status': 'success'
            }
            
            # Publish results
            self.publish("resume_data", resume_data)
            self.publish("features", features)
            
            self.results = result
            self.set_status("completed")
            
            return result
            
        except Exception as e:
            self.set_status("error")
            error_msg = str(e)
            print(f"Resume Agent error: {error_msg}")
            return {
                'status': 'error',
                'error': error_msg
            }





