@echo off
echo =========================================================
echo   APU Demand Forecasting System Startup
echo =========================================================
echo.
echo Starting FastAPI backend inside the virtual environment...
echo System logs will be printed below.
echo Press Ctrl+C to stop the server.
echo.

cd backend
..\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir app
