"""
Feature Extraction Module
Generates feature vectors using TF-IDF and embeddings (BERT/SentenceTransformer)
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Optional
import joblib
import os

# Optional import for sentence transformers (not available on Python 3.13 yet)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    print("Warning: sentence_transformers not available. Embedding features will be disabled.")

class FeatureExtractor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize feature extractor with TF-IDF and SentenceTransformer"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(model_name)
        else:
            self.embedding_model = None
        self.is_fitted = False
        
    def fit_tfidf(self, texts: List[str]):
        """Fit TF-IDF vectorizer on training texts"""
        self.tfidf_vectorizer.fit(texts)
        self.is_fitted = True
    
    def extract_tfidf_features(self, text: str) -> np.ndarray:
        """Extract TF-IDF features from text"""
        if not self.is_fitted:
            # If not fitted, fit on the current text (single document mode)
            # This allows feature extraction without pre-training
            try:
                self.tfidf_vectorizer.fit([text])
                self.is_fitted = True
            except:
                # Fallback: return zero vector
                return np.zeros(1000)  # Default max_features size
        return self.tfidf_vectorizer.transform([text]).toarray()[0]
    
    def extract_embedding(self, text: str) -> np.ndarray:
        """Extract BERT/SentenceTransformer embedding"""
        if self.embedding_model is None:
            # Return zero vector if sentence_transformers is not available
            return np.zeros(384)  # Default embedding size
        return self.embedding_model.encode(text, convert_to_numpy=True)
    
    def extract_structured_features(self, resume_data: Dict) -> np.ndarray:
        """Extract structured features from parsed resume data"""
        features = []
        
        # Skills count
        features.append(len(resume_data.get('skills', [])))
        
        # Experience years
        exp_years = resume_data.get('experience', {}).get('years', 0)
        features.append(exp_years)
        
        # Number of companies
        features.append(len(resume_data.get('experience', {}).get('companies', [])))
        
        # Education level (encode as numeric)
        education = resume_data.get('education', [])
        has_phd = any('phd' in str(edu).lower() or 'doctorate' in str(edu).lower() for edu in education)
        has_masters = any('master' in str(edu).lower() or 'ms' in str(edu).lower() for edu in education)
        has_bachelors = any('bachelor' in str(edu).lower() or 'bs' in str(edu).lower() for edu in education)
        
        features.append(1 if has_phd else 0)
        features.append(1 if has_masters else 0)
        features.append(1 if has_bachelors else 0)
        
        # Certifications count
        features.append(len(resume_data.get('certifications', [])))
        
        # Projects count
        features.append(len(resume_data.get('projects', [])))
        
        # Text length features
        features.append(resume_data.get('text_length', 0) / 1000.0)  # Normalized
        features.append(resume_data.get('word_count', 0) / 100.0)  # Normalized
        
        return np.array(features)
    
    def extract_all_features(self, resume_data: Dict, use_embeddings: bool = True) -> Dict[str, np.ndarray]:
        """Extract all feature types"""
        features = {}
        
        cleaned_text = resume_data.get('cleaned_text', '')
        if not cleaned_text:
            cleaned_text = resume_data.get('raw_text', '')
        
        # TF-IDF features (will auto-fit if not fitted)
        try:
            if cleaned_text:
                features['tfidf'] = self.extract_tfidf_features(cleaned_text)
        except Exception as e:
            print(f"TF-IDF extraction failed: {e}, using zeros")
            features['tfidf'] = np.zeros(1000)
        
        # Embedding features
        try:
            if use_embeddings and cleaned_text:
                features['embedding'] = self.extract_embedding(cleaned_text)
        except Exception as e:
            print(f"Embedding extraction failed: {e}, using zeros")
            features['embedding'] = np.zeros(384)  # Default embedding size
        
        # Structured features (always available)
        features['structured'] = self.extract_structured_features(resume_data)
        
        # Combined feature vector - prioritize embedding + structured
        if 'embedding' in features:
            if 'tfidf' in features:
                features['combined'] = np.concatenate([
                    features['embedding'],
                    features['structured']
                ])
            else:
                features['combined'] = np.concatenate([
                    features['embedding'],
                    features['structured']
                ])
        elif 'tfidf' in features:
            features['combined'] = np.concatenate([
                features['tfidf'],
                features['structured']
            ])
        else:
            # Fallback: use structured features only
            features['combined'] = features['structured']
        
        return features
    
    def save(self, filepath: str):
        """Save the fitted vectorizer"""
        if self.is_fitted:
            joblib.dump(self.tfidf_vectorizer, filepath)
    
    def load(self, filepath: str):
        """Load a fitted vectorizer"""
        if os.path.exists(filepath):
            self.tfidf_vectorizer = joblib.load(filepath)
            self.is_fitted = True





