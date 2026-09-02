# SIH26187 Prototype - Setup & Execution Guide

This guide will help you easily set up and run this project on **Windows** (as well as macOS / Linux).

---

## 📋 Prerequisites

Before running the project, make sure you have installed:
1. **Python 3.10 or 3.11** (Check *"Add Python to PATH"* during Windows installation): [python.org](https://www.python.org/downloads/)
2. **Node.js (v18 or higher)**: [nodejs.org](https://nodejs.org/)

---

## ⚡ Quick Start on Windows (1-Click)

### Step 1: Automatic Setup
Double-click [`setup_windows.bat`](file:///setup_windows.bat) or run in Command Prompt:
```cmd
setup_windows.bat
```
*(This creates the Python `venv`, installs required dependencies from `requirements.txt`, and runs `npm install` for the frontend.)*

### Step 2: Start the Application
Double-click [`run_project.bat`](file:///run_project.bat) or run:
```cmd
run_project.bat
```
This will automatically open two terminal windows:
- **Backend API**: `http://127.0.0.1:8000` (Docs: `http://127.0.0.1:8000/docs`)
- **Frontend Dashboard**: `http://localhost:5173`

---

## 🛠️ Manual Setup & Run (Step-by-Step)

If you prefer manual execution:

### 1. Backend Setup
Open a terminal in the `backend` folder:
```cmd
cd backend
python -m venv venv

:: Activate virtual environment on Windows:
venv\Scripts\activate

:: On Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup
Open a second terminal in the `frontend` folder:
```cmd
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your web browser.
