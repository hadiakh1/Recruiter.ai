@echo off
echo Fixing Python dependencies...
echo.

echo Step 1: Uninstalling problematic packages...
python -m pip uninstall -y pydantic pydantic-core fastapi

echo.
echo Step 2: Installing pydantic-core first...
python -m pip install pydantic-core==2.14.1

echo.
echo Step 3: Installing pydantic...
python -m pip install pydantic==2.5.0

echo.
echo Step 4: Installing fastapi...
python -m pip install fastapi==0.104.1

echo.
echo Step 5: Installing all other requirements...
python -m pip install -r requirements.txt

echo.
echo Done! Try running the server again.
pause

