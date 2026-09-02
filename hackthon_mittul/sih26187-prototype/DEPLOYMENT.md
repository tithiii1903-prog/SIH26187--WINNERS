# 🚀 Complete Online Deployment Guide1

This guide walks you through deploying the **Frontend to Vercel** and the **Backend + Database to Render / Railway / Hugging Face**.

---

## 📌 Summary of Architecture

- **Frontend**: [Vercel](https://vercel.com) (React + Vite + Tailwind/CSS SPA)
- **Backend**: [Render](https://render.com) or [Railway](https://railway.app) (FastAPI + PyTorch + InsightFace + YOLOv8 + OpenCV running in Docker)
- **Database & Storage**: Persistent Volume Disk attached to your backend container (persisting SQLite `command_center.db`, watchlist photos & embeddings).

---

## Step 1: Push Code to GitHub

If your project is not already on GitHub:
```bash
git add .
git commit -m "Configure deployment files for Vercel and Render/Docker" 
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

---

## Step 2: Deploy Backend & Database

### Option A: Render.com (Recommended)

1. Go to **[render.com](https://render.com)** and log in (sign up with GitHub).
2. Click **New +** → **Web Service**.
3. Select **Build and deploy from a Git repository** and connect your repository.
4. Fill in the configuration details:
   - **Name**: `sih-central-backend` (or your choice)
   - **Region**: Choose closest to you (e.g., Singapore, Frankfurt, Oregon)
   - **Branch**: `main`
   - **Root Directory**: `sih26187-prototype/backend`
   - **Runtime / Environment**: `Docker`
   - **Instance Type**: **Starter** (recommended for AI/PyTorch memory usage; or Free tier for light testing)
5. **(Important for Data Persistence)**:
   - Scroll to **Disks** → click **Add Disk**.
   - **Name**: `sih-data`
   - **Mount Path**: `/app/data`
   - **Size**: `1 GB` (or more)
6. Click **Create Web Service**.
7. Wait 3–5 minutes for the Docker image to build.
8. Once finished, copy your public backend URL:
   `https://sih-central-backend.onrender.com`

---

### Option B: Railway.app (Alternative)

1. Go to **[railway.app](https://railway.app)** and log in with GitHub.
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select your repository.
4. In the service settings:
   - Go to **Settings** → **Root Directory** → set to `/sih26187-prototype/backend`.
   - Railway will automatically detect the `Dockerfile`.
   - Under **Volumes**, add a Volume mounted to `/app/data`.
   - Under **Networking**, click **Generate Domain**.
5. Copy your domain: `https://...up.railway.app`.

---

## Step 3: Deploy Frontend to Vercel

1. Go to **[vercel.com](https://vercel.com)** and log in with GitHub.
2. Click **Add New...** → **Project**.
3. Import your GitHub repository.
4. Configure project settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: Click **Edit** and choose `sih26187-prototype/frontend`.
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. **Add Environment Variable**:
   - Expand the **Environment Variables** section.
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: Your backend URL from Step 2 (e.g. `https://sih-central-backend.onrender.com` — *do not add a trailing slash*).
6. Click **Deploy**.

---

## Step 4: Verification Checklist

1. Open your Vercel URL (e.g. `https://sih26187-prototype.vercel.app`).
2. Verify that:
   - [ ] Camera Feeds list loads without errors.
   - [ ] Uploading a new video or camera feed registers and begins AI processing.
   - [ ] Watchlist and Face Recognition register faces and save to the persistent database.
   - [ ] Live analytics stream correctly.

---

## 🛠️ Local Development

For testing locally at any time:
- Backend: `uvicorn app.main:app --reload --port 8000` (runs at `http://127.0.0.1:8000`)
- Frontend: `npm run dev` in `frontend/` (uses default fallback `http://127.0.0.1:8000`)
