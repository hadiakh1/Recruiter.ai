#!/bin/bash

# Simple startup script for Hybrid Recruitment System
# No virtual environment required

set -e  # Exit on error

echo "=========================================="
echo "  Hybrid Recruitment System Startup"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."
if ! command_exists python3; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+"
    exit 1
fi

if ! command_exists npm; then
    echo "❌ npm is not installed. Please install Node.js 16+"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Backend setup
echo -e "${BLUE}Setting up backend...${NC}"
cd backend

# Install Python dependencies (without virtual environment)
echo "Installing Python dependencies..."

# Check Python version and use appropriate requirements file
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

python3 -m pip install --user --upgrade pip setuptools wheel -q

# For Python 3.13+, some packages (like PyTorch) may not be available
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 13) else 1)" 2>/dev/null; then
    echo "⚠️  Python 3.13 detected - some ML packages may not be available"
    echo "Installing core dependencies first..."
    python3 -m pip install --user -q -r requirements_core.txt 2>/dev/null || true
    
    echo "Installing compatible ML packages..."
    # Install scikit-learn (has Python 3.13 support)
    python3 -m pip install --user -q "scikit-learn>=1.5.0" "numpy>=1.26.2" "pandas>=2.1.3" "joblib>=1.3.2" 2>/dev/null || true
    
    echo "⚠️  PyTorch and transformers are not available for Python 3.13 yet"
    echo "   ML features requiring these will be disabled"
    echo "   Consider using Python 3.11 or 3.12 for full functionality"
else
    echo "Installing all dependencies..."
    python3 -m pip install --user -q -r requirements.txt || python3 -m pip install -q -r requirements.txt
fi

# Download NLTK data (silent mode)
echo "Downloading NLTK data..."
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
" 2>/dev/null || echo "NLTK data download skipped (may already be installed)"

echo -e "${GREEN}✅ Backend setup complete${NC}"
echo ""

# Frontend setup
echo -e "${BLUE}Setting up frontend...${NC}"
cd ../frontend

# Install Node dependencies (only if node_modules doesn't exist or package.json changed)
if [ ! -d "node_modules" ] || [ "package.json" -nt "node_modules" ]; then
    echo "Installing Node dependencies..."
    npm install --silent
else
    echo "Node dependencies already installed"
fi

echo -e "${GREEN}✅ Frontend setup complete${NC}"
echo ""

# Start servers
echo "=========================================="
echo -e "${YELLOW}Starting servers...${NC}"
echo "=========================================="
echo ""
echo "Backend will run on: http://localhost:8000"
echo "Frontend will run on: http://localhost:3000"
echo "API docs: http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}"
echo ""

# Start backend in background
cd "$SCRIPT_DIR/backend"
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend in background
cd "$SCRIPT_DIR/frontend"
npm run dev > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "Servers stopped."
    exit 0
}

# Trap Ctrl+C
trap cleanup INT TERM

echo -e "${GREEN}✅ Both servers are running!${NC}"
echo ""
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Logs:"
echo "  Backend: tail -f /tmp/backend.log"
echo "  Frontend: tail -f /tmp/frontend.log"
echo ""

# Wait for processes
wait

