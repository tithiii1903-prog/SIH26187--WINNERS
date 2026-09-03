# 🌐 IBVAP — Intelligent Border Video Analytics Platform (SIH26187)

[![Live Web Application](https://img.shields.io/badge/Live_App-Vercel-000000?style=for-the-badge&logo=vercel)](https://frontend-lemon-mu-24.vercel.app/)
[![Live Backend API](https://img.shields.io/badge/Backend_API-Railway-0B0D0E?style=for-the-badge&logo=railway)](https://sih26187-winners-production.up.railway.app/docs)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)](https://react.dev)

---

## 📌 Live Demo Links

* 🚀 **Live Production Application (Vercel)**:  
  👉 **[https://frontend-lemon-mu-24.vercel.app/](https://frontend-lemon-mu-24.vercel.app/)**

* ⚡ **Live Backend API & Swagger Docs (Railway)**:  
  👉 **[https://sih26187-winners-production.up.railway.app/docs](https://sih26187-winners-production.up.railway.app/docs)**

---

## 💡 About the Project

The **Intelligent Border Video Analytics Platform (IBVAP)** is a high-performance, real-time security surveillance and facial recognition command system designed for automated perimeter defense, border monitoring, and critical watchlist tracking.

### 🌟 Key Capabilities

1. **🎯 Subsystem 02 — High-Definition Facial Recognition Engine**
   - **RetinaFace High-Density Detection**: Detects faces directly on high-definition camera streams.
   - **ArcFace 512D Biometric Embeddings**: Generates 512-dimensional normalized feature vectors (InsightFace).
   - **Cosine Similarity Matching**: Real-time cross-referencing against enrolled watchlist profiles.
   - **Cloud Stream Fallback**: Automatic synthetic 30 FPS stream generator for cloud server deployments without physical USB webcams.

2. **🚗 Multi-Object Tracking & Vehicle Classification**
   - **YOLOv8 Real-Time AI Detector**: Deep learning detection for humans, cars, motorcycles, buses, and trucks.
   - **ByteTrack Multi-Object Tracking**: Assigns persistent track IDs and movement vectors across video frames.

3. **🛡️ Interactive Virtual Polygon Fence Intrusion Radar**
   - **Custom Fence Boundary Drawing**: Interactive HTML5 canvas polygon boundary editor.
   - **Real-Time Line-Crossing & Breach Alerts**: Instant visual/audible alarms when tracked targets enter restricted zones.

4. **👤 Biometric Watchlist Management**
   - **Single-Face Biometric Profile Enrollment**: Validates upload photos for exactly one face.
   - **Persistent SQLite Database (`command_center.db`)**: Stores watchlist records, embeddings (`.npy`), and historical security events.
   - **Zero Raw Vector Exposure**: Secure architecture that keeps raw embedding vectors hidden from public representations.

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend UI** | React 18, TypeScript, Vite, Custom Vanilla CSS Dark Mode, SVG/Canvas Drawing |
| **Backend API** | FastAPI, Uvicorn, Python 3.10 |
| **AI & Computer Vision** | PyTorch (CPU-optimized), InsightFace (ArcFace), Ultralytics YOLOv8, OpenCV, ONNX Runtime |
| **Database** | SQLite3 (`command_center.db`) |
| **Deployment** | Vercel (Frontend SPA) + Railway.app (Backend Docker Container) |

---

## 💻 Local Setup & Execution Guide

### Prerequisites
Before running locally, ensure you have installed:
* **Python 3.10 or 3.11**: [python.org/downloads](https://www.python.org/downloads/)
* **Node.js (v18 or higher)**: [nodejs.org](https://nodejs.org/)

---

### ⚡ 1-Click Quick Start on Windows

1. Run **Automatic Setup**:
   Double-click `setup_windows.bat` (or run in CMD: `setup_windows.bat`).
   *(Creates Python `venv`, installs `requirements.txt`, and runs `npm install` for frontend).*

2. Run **Application Launcher**:
   Double-click `run_project.bat` (or run in CMD: `run_project.bat`).
   *(Launches Backend on `http://127.0.0.1:8000` and Frontend on `http://localhost:5173`)*.

---

### 🛠️ Manual Step-by-Step Setup

#### 1. Backend Setup (Terminal 1)
```bash
cd hackthon_mittul/sih26187-prototype/backend

# Create & activate Python virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
# source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
* Backend running at: `http://127.0.0.1:8000`
* Interactive API Docs: `http://127.0.0.1:8000/docs`

#### 2. Frontend Setup (Terminal 2)
```bash
cd hackthon_mittul/sih26187-prototype/frontend

# Install dependencies
npm install

# Start Vite dev server
npm run dev
```
* Frontend running at: `http://localhost:5173`

---

## 📡 API Architecture Overview

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /api/health` | `GET` | Health check endpoint |
| `GET /api/feeds` | `GET` | List all surveillance camera/file feeds |
| `POST /api/feeds` | `POST` | Upload MP4 video feed |
| `POST /api/feeds/camera` | `POST` | Register device camera feed |
| `POST /api/feeds/{id}/start` | `POST` | Start live processing for primary feed |
| `GET /api/stream/{id}` | `GET` | MJPEG stream of AI-processed surveillance frames |
| `GET /api/watchlist` | `GET` | List biometric watchlist records |
| `POST /api/watchlist` | `POST` | Enroll new subject into facial watchlist |
| `POST /api/face-camera/start` | `POST` | Start HD Face Recognition Camera subsystem |
| `GET /api/face-camera/stream` | `GET` | MJPEG stream of HD Face Recognition camera with overlays |
| `GET /api/zones` | `GET` | Fetch active virtual fence polygon configuration |
| `POST /api/zones` | `POST` | Update virtual fence polygon boundaries |

---

## 📄 License & Attribution

Developed for **Smart India Hackathon (SIH26187)**. All rights reserved.
