@echo off
echo ========================================================
echo   SIH26187 Prototype - Windows Environment Setup
echo ========================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10 or 3.11 from https://www.python.org/
    pause
    exit /b 1
)

:: 2. Setup Backend
echo [1/3] Setting up Python Virtual Environment in backend...
cd backend
if not exist "venv" (
    python -m venv venv
)
call venv\Scripts\activate.bat

echo [2/3] Installing Python backend dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

cd ..

:: 3. Setup Frontend
echo [3/3] Installing Frontend Node dependencies...
cd frontend
call npm install
cd ..

echo.
echo ========================================================
echo  Setup Completed Successfully!
echo  To run the project, execute 'run_project.bat'
echo ========================================================
pause
