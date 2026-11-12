# 🚀 CrowdPulse — Crowdfunding Success Predictor & Campaign Advisor

An end-to-end Applied Machine Learning decision-support system and interactive web application that predicts the success probability of Kickstarter campaigns before launch and provides actionable pre-launch recommendations.

![CrowdPulse Architecture](https://img.shields.io/badge/Architecture-FastAPI%20%2B%20Scikit--Learn-emerald?style=for-the-badge)
![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-HistGradientBoosting-orange?style=for-the-badge)
![Deployment](https://img.shields.io/badge/Deploy-Render%20%2F%20Docker-purple?style=for-the-badge)

---

## 📌 Executive Summary

Over **60% of crowdfunding campaigns on Kickstarter fail** to reach their funding goal. Because Kickstarter operates on an **all-or-nothing** payout mechanism, raising \$9,990 towards a \$10,000 goal yields \$0 in payouts, resulting in sunk marketing costs, lost prototype investments, and operational distress.

**CrowdPulse** acts as a pre-launch data advisor:
1. **Zero Data Leakage:** Operates strictly on pre-launch parameters (title, category, currency-normalized goal, duration, timing). Post-launch signals (`pledged`, `backers`) are rigorously excluded.
2. **Predictive Intelligence:** Outputs calibrated success probabilities using an optimized `HistGradientBoostingClassifier` within a Scikit-Learn `Pipeline`.
3. **2D Sweet-Spot Goal Sensitivity:** Dynamically sweeps funding targets to help creators discover their project's financial tipping point.
4. **Actionable Recommendations:** Heuristic and tree-derived advice tailored to category norms, title wording, and campaign length.

---

## 🏗️ Project Architecture

```
PROJECT/
├── backend/
│   └── main.py                     # FastAPI REST API & Static File Server
├── frontend/
│   └── index.html                  # Interactive Dashboard (TailwindCSS + Chart.js + Lucide)
├── model_artifacts/
│   ├── kickstarter_pipeline.joblib # Serialized Scikit-Learn Pipeline
│   └── model_metadata.json         # Performance metrics & category encodings
├── Notebook/
│   ├── AIML_Project.ipynb          # Full Research Notebook (EDA, CV, Diagnostics)
│   └── kickstarter_data.csv        # Historical Kickstarter Dataset
├── Dockerfile                      # Production Docker container definition
├── render.yaml                     # Free 1-Click Render Cloud deployment spec
├── requirements.txt                # Production Python dependencies
├── train.py                        # Automated pipeline retraining script
└── README.md                       # Comprehensive project documentation
```

---

## 📊 Model Evaluation & Benchmarks

The model was evaluated against **12,000 unseen holdout campaigns** using Stratified 80/20 splitting:

| Metric | Holdout Test Score | Notes |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.7177** | Calibrated class separation across all categories |
| **Accuracy** | **67.14%** | Strict pre-launch feature set |
| **F1 Score** | **0.5435** | Balanced harmonic mean for minority class |
| **Precision** | **61.58%** | Reliable positive predictions |

---

## 🚀 Running Locally

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/crowdpulse.git
cd crowdpulse

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Retrain Model Pipeline
```bash
python train.py
```
*Trains on the dataset and exports `model_artifacts/kickstarter_pipeline.joblib` in seconds.*

### 3. Launch Web App & Backend
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at:
* **Interactive Web App:** `http://127.0.0.1:8000/`
* **Interactive Swagger API Docs:** `http://127.0.0.1:8000/docs`

---

## 🌐 Deploying Online (100% Free)

### Method 1: Deploy to Render (Recommended - 3 Minutes)
1. Push this repository to **GitHub**.
2. Go to [Render.com](https://render.com) and sign in with GitHub.
3. Click **New +** > **Web Service**.
4. Select your GitHub repository.
5. Configure the settings:
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   * **Instance Type:** `Free`
6. Click **Create Web Service**. Render will build and deploy your app with a public `https://<your-app>.onrender.com` URL!

---

### Method 2: Deploy to Railway / Fly.io
This repository includes a production `Dockerfile`.
```bash
# Deploy with Railway CLI
railway up

# Or deploy with Fly.io
fly launch
```

---

### Method 3: Deploy to Hugging Face Spaces (Docker Space)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Select **Docker** as the SDK.
3. Push the files (`Dockerfile`, `requirements.txt`, `backend/`, `frontend/`, `model_artifacts/`) to your Space repository.
4. Hugging Face builds and hosts the app on a free GPU/CPU tier instantly.

---

## 🔌 REST API Reference

### 1. Health Check
`GET /api/health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "metadata": { ... }
}
```

### 2. Campaign Prediction
`POST /api/predict`
```json
{
  "title": "Tactical Deckbuilding Board Game",
  "main_category": "Games",
  "country": "US",
  "usd_goal": 12000.0,
  "duration_days": 30.0,
  "launch_month": 10,
  "launch_day_of_week": 1
}
```
**Response:**
```json
{
  "title": "Tactical Deckbuilding Board Game",
  "probability": 0.684,
  "probability_pct": "68.4%",
  "prediction": "Successful",
  "status": "Optimal",
  "risk_level": "Moderate Risk (Viable)",
  "actionable_advice": [
    "Optimal Duration: 25-35 days is the sweet spot for crowdfunding momentum.",
    "Campaign configuration looks well-optimized for launch!"
  ]
}
```

### 3. Sweet-Spot Sensitivity Sweep
`POST /api/simulate`
Generates a dynamic multi-point array mapping dollar goals to predicted probabilities for interactive graphing.

---

## ⚖️ Ethical Governance & Limitations
* **Unobserved Variables:** Prototype production quality, video aesthetics, social media follower count, and paid ad spend are outside the scope of raw campaign metadata.
* **Economic Drift:** Inflation and market shifts since data collection may require periodic model retraining (`train.py`).

# AIML_Sem1_Project
