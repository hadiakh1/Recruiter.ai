#!/bin/bash

# Simple backend startup script (no virtual environment)

cd "$(dirname "$0")/backend"

echo "Starting backend server..."
echo "Backend will run on: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""

# Install dependencies if needed
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

python3 -m pip install --user --upgrade pip setuptools wheel -q

# For Python 3.13+, some packages may not be available
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 13) else 1)" 2>/dev/null; then
    echo "⚠️  Python 3.13 detected - installing compatible packages..."
    python3 -m pip install --user -q -r requirements_core.txt 2>/dev/null || true
    python3 -m pip install --user -q "scikit-learn>=1.5.0" "numpy>=1.26.2" "pandas>=2.1.3" "joblib>=1.3.2" 2>/dev/null || true
    echo "⚠️  PyTorch not available for Python 3.13 - some ML features disabled"
else
    python3 -m pip install --user -q -r requirements.txt || python3 -m pip install -q -r requirements.txt
fi

# Download NLTK data with SSL bypass
python3 -c "
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import nltk
try:
    nltk.download('punkt_tab', quiet=True)
except:
    nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
" 2>/dev/null || true

# Start server
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

