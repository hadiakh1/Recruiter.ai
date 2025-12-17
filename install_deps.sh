#!/bin/bash

# Simple dependency installer that handles Python 3.13 compatibility

cd "$(dirname "$0")/backend"

echo "Installing Python dependencies..."
echo "Python version: $(python3 --version)"

# Upgrade pip and build tools first
python3 -m pip install --user --upgrade pip setuptools wheel

# Install scikit-learn separately with compatible version for Python 3.13
echo "Installing scikit-learn (Python 3.13 compatible)..."
python3 -m pip install --user "scikit-learn>=1.5.0" || {
    echo "Warning: Could not install scikit-learn 1.5.0+, trying latest..."
    python3 -m pip install --user scikit-learn
}

# Install other dependencies
echo "Installing other dependencies..."
python3 -m pip install --user -r requirements.txt --no-deps || true

# Install remaining dependencies
python3 -m pip install --user \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    pydantic-core==2.14.1 \
    python-multipart==0.0.6 \
    torch \
    transformers \
    sentence-transformers \
    pandas \
    numpy \
    networkx \
    shap \
    lime \
    PyPDF2 \
    python-docx \
    nltk \
    imbalanced-learn \
    joblib \
    aiofiles \
    "python-jose[cryptography]" \
    "passlib[bcrypt]" \
    pytest \
    httpx

echo "✅ Dependencies installed!"


