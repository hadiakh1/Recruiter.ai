"""
Machine Learning Models Module
Implements multiple classifiers: Logistic Regression, Random Forest, Decision Tree, MLP, BERT-based
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Optional imports for imbalanced learning
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    SMOTE = None
    RandomUnderSampler = None
    print("Warning: imbalanced-learn not available. Class imbalance handling will be disabled.")
import joblib
import os
from typing import Dict, List, Tuple, Optional

# Optional imports for PyTorch (not available on Python 3.13 yet)
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None
    AutoTokenizer = None
    AutoModel = None
    Dataset = None
    DataLoader = None
    print("Warning: PyTorch not available. BERT-based models will be disabled.")

if PYTORCH_AVAILABLE:
    class JobFitDataset(Dataset):
        """PyTorch Dataset for job fit classification"""
        def __init__(self, features, labels):
            self.features = torch.FloatTensor(features)
            self.labels = torch.LongTensor(labels)
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    class BERTClassifier(nn.Module):
        """BERT-based classifier for job fit prediction"""
        def __init__(self, model_name='bert-base-uncased', num_classes=2, hidden_dim=256):
            super(BERTClassifier, self).__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
            )
        
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            output = self.dropout(pooled_output)
            return self.classifier(output)
else:
    # Placeholder classes when PyTorch is not available
    class JobFitDataset:
        """Placeholder - PyTorch not available"""
        pass
    
    class BERTClassifier:
        """Placeholder - PyTorch not available"""
        pass

class MLModelTrainer:
    """Trainer for multiple ML models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.best_score = 0.0
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, 
                    handle_imbalance: bool = True,
                    random_state: int = 42) -> Tuple:
        """Prepare and split data, handle class imbalance"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        if handle_imbalance and IMBALANCED_LEARN_AVAILABLE:
            # Use SMOTE for oversampling
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train Logistic Regression model"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        scores = self._calculate_metrics(y_test, y_pred)
        
        self.models['logistic_regression'] = model
        self.scalers['logistic_regression'] = scaler
        scores['model_name'] = 'Logistic Regression'
        
        return scores
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           n_estimators: int = 100) -> Dict:
        """Train Random Forest model"""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        scores = self._calculate_metrics(y_test, y_pred)
        
        self.models['random_forest'] = model
        scores['model_name'] = 'Random Forest'
        scores['feature_importance'] = model.feature_importances_.tolist()
        
        return scores
    
    def train_decision_tree(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train Decision Tree model"""
        model = DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',
            max_depth=10
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        scores = self._calculate_metrics(y_test, y_pred)
        
        self.models['decision_tree'] = model
        scores['model_name'] = 'Decision Tree'
        
        return scores
    
    def train_mlp(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 hidden_layers: Tuple = (100, 50)) -> Dict:
        """Train Multi-Layer Perceptron model"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        scores = self._calculate_metrics(y_test, y_pred)
        
        self.models['mlp'] = model
        self.scalers['mlp'] = scaler
        scores['model_name'] = 'MLP'
        
        return scores
    
    def train_bert_classifier(self, texts: List[str], labels: List[int],
                             model_name: str = 'bert-base-uncased',
                             batch_size: int = 16,
                             epochs: int = 3,
                             learning_rate: float = 2e-5) -> Dict:
        """Train BERT-based classifier (simplified version)"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. BERT classifier requires PyTorch.")
        
        # Note: Full BERT training requires more setup. This is a simplified version.
        # For production, you'd need proper tokenization and training loop.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # This is a placeholder - full implementation would require proper training loop
        # For now, we'll use a simpler approach with embeddings
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(model_name)
            embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        except ImportError:
            raise ImportError("sentence_transformers is not available. Please install it or use Python 3.11/3.12.")
        
        # Use a simple classifier on top of embeddings
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train a simple classifier on embeddings
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        y_pred = rf_model.predict(X_test)
        scores = self._calculate_metrics(y_test, y_pred)
        
        self.models['bert_classifier'] = rf_model
        self.models['bert_tokenizer'] = tokenizer
        self.models['bert_embedder'] = embedding_model
        scores['model_name'] = 'BERT-based Classifier'
        
        return scores
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def cross_validate(self, model_name: str, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """Perform cross-validation"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)
        
        if scaler:
            X_scaled = scaler.transform(X)
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_weighted')
        else:
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)
        
        if scaler:
            X_scaled = scaler.transform(X)
            return model.predict_proba(X_scaled)[:, 1]  # Return probability of positive class
        else:
            return model.predict_proba(X)[:, 1]
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)
        
        joblib.dump(model, filepath)
        if scaler:
            scaler_path = filepath.replace('.pkl', '_scaler.pkl')
            joblib.dump(scaler, scaler_path)
    
    def load_model(self, model_name: str, model_path: str, scaler_path: Optional[str] = None):
        """Load trained model"""
        self.models[model_name] = joblib.load(model_path)
        if scaler_path and os.path.exists(scaler_path):
            self.scalers[model_name] = joblib.load(scaler_path)





