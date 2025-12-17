#!/usr/bin/env python
"""
Backend Server Startup Script
Starts the FastAPI server with proper error handling

"""
from app.main import app  # import the FastAPI app at top level

# Now uvicorn can find it:
# uvicorn start_server:app --reload

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    try:
        import fastapi
    except ImportError:
        missing.append("fastapi")
    
    try:
        import uvicorn
    except ImportError:
        missing.append("uvicorn")
    
    try:
        from passlib.context import CryptContext
    except ImportError:
        missing.append("passlib[bcrypt]")
    
    try:
        from jose import jwt
    except ImportError:
        missing.append("python-jose[cryptography]")
    
    if missing:
        print("ERROR: Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Start the server"""
    if not check_dependencies():
        sys.exit(1)
    
    try:
        import uvicorn
        from app.main import app
        
        print("=" * 50)
        print("Starting Hybrid Recruitment System Backend")
        print("=" * 50)
        print("Server will be available at: http://localhost:8000")
        print("API docs will be available at: http://localhost:8000/docs")
        print("=" * 50)
        print("\nPress Ctrl+C to stop the server\n")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=True  # Auto-reload on code changes
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\nERROR: Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

