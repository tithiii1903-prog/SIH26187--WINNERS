@echo off
echo ========================================================
echo   Starting SIH26187 Prototype (Backend + Frontend)
echo ========================================================
echo.

:: Start Backend in a separate window
echo Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Backend - FastAPI" cmd /k "cd backend && call venv\Scripts\activate.bat && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

:: Start Frontend in a separate window
echo Starting Frontend UI on http://localhost:5173 ...
start "Frontend - Vite" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are launching!
echo UI will be accessible at: http://localhost:5173
echo API Docs accessible at:    http://127.0.0.1:8000/docs
echo.
