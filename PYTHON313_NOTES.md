# Python 3.13 Compatibility Notes

## Important: Python 3.13 Limitations

You're using **Python 3.13**, which is very new. Some packages don't have pre-built wheels yet:

### ✅ **Works:**
- FastAPI, Uvicorn
- Pydantic (v2.9.0+)
- scikit-learn (v1.5.0+)
- Most core dependencies

### ❌ **Not Available:**
- **PyTorch** - No Python 3.13 wheels yet
- **transformers** - Depends on PyTorch
- **sentence-transformers** - Depends on PyTorch
- Some older versions of pydantic-core

### 🔧 **Solutions:**

**Option 1: Use Python 3.11 or 3.12 (Recommended)**
```bash
# Install Python 3.11 or 3.12 using pyenv or Homebrew
brew install python@3.11
# Then use: python3.11 instead of python3
```

**Option 2: Run with Limited Features (Current Setup)**
The startup scripts will automatically:
- Install all compatible packages
- Skip PyTorch/transformers (ML features will be limited)
- Still allow you to use the API and frontend

**Option 3: Wait for Package Updates**
PyTorch and related packages should add Python 3.13 support in future releases.

## Current Status

The application will run but **ML features requiring PyTorch will be disabled**. You can still:
- Upload and parse resumes
- Use CSP (Constraint Satisfaction Problem) solver
- Use Best-First Search ranking
- Access the frontend and API

## Quick Start

Just run:
```bash
./start.sh
```

The script will automatically handle Python 3.13 compatibility.


