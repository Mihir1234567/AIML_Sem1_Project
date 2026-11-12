# 🚀 CrowdPulse — AI Crowdfunding Predictor & Campaign Advisor
### *End-to-End Applied Machine Learning System & Pre-Launch Decision-Support Web Application*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-emerald?style=for-the-badge&logo=render)](https://crowdpulse-api-3yaw.onrender.com/)
[![Swagger API](https://img.shields.io/badge/Swagger%20API-Docs-blue?style=for-the-badge&logo=fastapi)](https://crowdpulse-api-3yaw.onrender.com/docs)
[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Model](https://img.shields.io/badge/Champion%20Model-HistGradientBoosting-purple?style=for-the-badge)](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)
[![ROC-AUC](https://img.shields.io/badge/Test%20ROC--AUC-0.7177-success?style=for-the-badge)]()

---

## 🌐 Live Deployment Links

| Resource | URL | Description |
| :--- | :--- | :--- |
| **Interactive Web Application** | [https://crowdpulse-api-3yaw.onrender.com/](https://crowdpulse-api-3yaw.onrender.com/) | Live user dashboard with real-time predictions, gauge meter, preset campaigns, and goal sensitivity curves. |
| **Interactive API Documentation** | [https://crowdpulse-api-3yaw.onrender.com/docs](https://crowdpulse-api-3yaw.onrender.com/docs) | Interactive Swagger UI for making direct REST API calls. |
| **API Health & Diagnostics** | [https://crowdpulse-api-3yaw.onrender.com/api/health](https://crowdpulse-api-3yaw.onrender.com/api/health) | Real-time JSON health check returning model state, metadata, and validation metrics. |

---

## 📌 1. Executive Summary & Problem Formulation

Crowdfunding platforms like Kickstarter have unlocked billions of dollars in peer-to-peer backing for inventors, game developers, filmmakers, and creators worldwide. However, empirical statistics reveal a harsh reality: **over 60% of all campaigns fail to reach their target goal**.

Because Kickstarter enforces a strict **"all-or-nothing"** funding disbursement model, failing to reach a target—even by a tiny fraction (e.g., raising \$9,950 of a \$10,000 goal)—results in **\$0 collected**, sunk prototype investments, and severe financial distress.

### Project Objective
**CrowdPulse** is a pre-launch predictive analytics and decision-support system. It enables creators to:
1. **Estimate Success Probability:** Calculate calibrated success probabilities prior to launch.
2. **Eliminate Data Leakage:** Restrict features strictly to pre-launch configurations.
3. **Explore Goal Sensitivity:** Identify the optimal funding "sweet spot" before setting campaign terms in stone.
4. **Receive Automated Advisory:** Get heuristic and model-derived recommendations on duration, title wording, and launch timing.

---

## 🛡️ 2. Anti-Leakage & Data Hygiene Architecture

A major flaw in naive crowdfunding ML models is **target leakage**. Attributes such as `pledged`, `usd_pledged`, `backers_count`, or `spotlight` are recorded *while* or *after* a campaign is live. Training a model on these variables yields artificially inflated test scores that fail catastrophically in real-world pre-launch deployment.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   DATA LEAKAGE PREVENTION AUDIT                        │
├──────────────────────────────────┬─────────────────────────────────────┤
│ ❌ Strictly Excluded (Post-Launch)│ ✅ Retained Pre-Launch Predictors    │
├──────────────────────────────────┼─────────────────────────────────────┤
│ • pledged / usd_pledged          │ • Clean project title & semantics   │
│ • backers tally                  │ • Main category & country           │
│ • live / suspended states        │ • Normalized real USD goal (FX)     │
│ • staff pick / spotlight         │ • Campaign duration (1-90 days)     │
│                                  │ • Launch calendar timing (DoW, Mo)  │
└──────────────────────────────────┴─────────────────────────────────────┘
```

### Multi-Currency Normalization
Kickstarter records raw goals in 13+ native currencies (USD, GBP, EUR, CAD, AUD, etc.). Unadjusted currency values distort loss functions. CrowdPulse standardizes all campaign targets into real USD purchasing power using benchmark foreign exchange rates prior to logarithmic transformation (`log_usd_goal = log1p(usd_goal_real)`).

---

## 🔬 3. Machine Learning Methodology

### 3.1 Feature Engineering Matrix
1. **Financial Hurdle:** $\log(1 + \text{usd\_goal\_real})$ to compress heavy right-skewed funding distributions.
2. **Temporal Dynamics:** 
   - Campaign duration in days ($\text{deadline} - \text{launched}$).
   - Launch month ($1-12$).
   - Launch day of week ($0-6$) and binary weekend indicator ($\text{is\_weekend}$).
3. **Title Semantics & Heuristics:**
   - Character length and token word count.
   - Punctuation cues: presence of exclamation marks (`!`) and question marks (`?`).
   - Structural formatting: ALL CAPS indicator (`is_all_caps`).
4. **Categorical Encodings:**
   - `main_category` (15 sectors: Games, Technology, Design, Film, Music, etc.).
   - `country` (22 international jurisdictions).

### 3.2 Preprocessing Pipeline
All transformations are encapsulated into a Scikit-Learn `ColumnTransformer` to guarantee zero data snooping:
* **Numerical Stream:** `StandardScaler()` fit strictly on training splits.
* **Categorical Stream:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

### 3.3 Algorithmic Benchmark & Cross-Validation
Four algorithmic archetypes were evaluated using **3-Fold Stratified Cross-Validation**:

| Model Archetype | Mean CV ROC-AUC | Std CV ROC-AUC | Mean CV F1 | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Champion)** | **0.7182** | $\pm 0.0031$ | **0.5482** | ~0.8s |
| **Random Forest Classifier** | 0.7094 | $\pm 0.0028$ | 0.5310 | ~2.1s |
| **Logistic Regression ($L_2$)** | 0.6912 | $\pm 0.0035$ | 0.5124 | ~0.5s |
| **Dummy Baseline (Majority Class)** | 0.5000 | $\pm 0.0000$ | 0.0000 | ~0.1s |

---

## 📈 4. Final Evaluation on 12,000 Unseen Holdout Campaigns

The champion `HistGradientBoostingClassifier` was evaluated against a dedicated holdout test set ($N = 12,000$ campaigns, stratified 80/20 split):

| Evaluation Metric | Score | Analytical Interpretation |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.7177** | Strong discriminatory separation between successful and failing campaigns using pre-launch metadata alone. |
| **Accuracy** | **67.14%** | Baseline accuracy exceeds majority-class naive prediction by ~13 percentage points. |
| **Precision** | **61.58%** | Reliable positive predictive value when advising creators to proceed. |
| **Recall** | **48.63%** | Conservative positive detection; minimizes recommending high-risk unviable projects. |
| **F1 Score** | **0.5435** | Harmonic mean reflecting stable precision-recall trade-off. |

### Model Explainability via Permutation Importance
Rather than relying on default tree Gini impurity (which is notoriously biased towards continuous features with high cardinality), explainability was computed via **Test-Set Permutation Feature Importance**:
1. **`log_usd_goal` (Dominant Driver):** Shuffling the funding goal degrades test ROC-AUC by over 0.14 points. Higher funding targets exponentially decrease success probability.
2. **`main_category`:** Creative sectors (Comics, Dance, Theater) possess significantly higher empirical success baselines compared to capital-intensive sectors (Technology, Food, Journalism).
3. **`duration_days`:** Campaigns extending beyond 35 days experience donor fatigue, whereas 30-day campaigns sustain higher momentum.

---

## 🏗️ 5. Unified System Architecture

The application is deployed on **Render** as a self-contained, high-performance web service:

```
AIML/PROJECT/
├── backend/
│   ├── __init__.py
│   └── main.py                     # FastAPI REST API + Static Server + Self-Healing Loader
├── frontend/
│   └── index.html                  # Responsive Dashboard (Tailwind CSS + Chart.js + Lucide)
├── model_artifacts/
│   ├── kickstarter_pipeline.joblib # Serialized Scikit-Learn Pipeline (330 KB)
│   └── model_metadata.json         # Evaluation benchmarks & category registries
├── Notebook/
│   ├── AIML_Project.ipynb          # End-to-end research, EDA, and diagnostic experiments
│   └── kickstarter_data.csv        # Historical Kickstarter dataset
├── train.py                        # Standalone script for automated cloud pipeline retraining
├── requirements.txt                # Pinned production dependencies (scikit-learn==1.5.2)
├── render.yaml                     # Infrastructure-as-code specification for Render
├── Dockerfile                      # Production container spec (Python 3.11-slim)
└── README.md                       # Comprehensive system documentation
```

---

## 🔌 6. REST API Reference

### `GET /api/health`
Returns server status, model persistence verification, and benchmark metrics.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "metadata": {
    "model_name": "HistGradientBoostingClassifier",
    "metrics": {
      "accuracy": 0.6714,
      "precision": 0.6158,
      "recall": 0.4863,
      "f1": 0.5435,
      "roc_auc": 0.7177
    },
    "train_samples": 48000,
    "test_samples": 12000
  }
}
```

---

### `POST /api/predict`
Calculates campaign success probability, risk tier, and contextual advisory.

**Request Body:**
```json
{
  "title": "The Odyssey: Strategic Deckbuilding Board Game",
  "main_category": "Games",
  "country": "US",
  "usd_goal": 12000.0,
  "duration_days": 30.0,
  "launch_month": 10,
  "launch_day_of_week": 1
}
```

**Response Body:**
```json
{
  "title": "The Odyssey: Strategic Deckbuilding Board Game",
  "main_category": "Games",
  "country": "US",
  "usd_goal": 12000.0,
  "duration_days": 30.0,
  "probability": 0.6482,
  "probability_pct": "64.8%",
  "prediction": "Successful",
  "status": "Optimal",
  "risk_level": "Moderate Risk (Viable)",
  "actionable_advice": [
    "🌟 Excellent Configuration: Your campaign parameters fall within historically high-performing cohorts.",
    "Optimal Duration: 25-35 days is the sweet spot for crowdfunding momentum."
  ],
  "model_name": "HistGradientBoostingClassifier"
}
```

---

### `POST /api/simulate`
Generates a multi-point sensitivity array across varying funding goals for interactive charting.

**Request Body:**
```json
{
  "title": "Smart Ergonomic Split Mechanical Keyboard",
  "main_category": "Technology",
  "country": "US",
  "min_goal": 2000,
  "max_goal": 60000,
  "duration_days": 30,
  "steps": 15
}
```

**Response Body:**
```json
{
  "title": "Smart Ergonomic Split Mechanical Keyboard",
  "category": "Technology",
  "duration_days": 30.0,
  "curve": [
    {"goal": 2000.0, "probability": 54.2},
    {"goal": 6142.9, "probability": 46.1},
    {"goal": 10285.7, "probability": 39.8},
    {"goal": 60000.0, "probability": 16.3}
  ]
}
```

---

## 💻 7. Local Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/Mihir1234567/AIML_Sem1_Project.git
cd AIML_Sem1_Project
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Retrain Model Pipeline
```bash
python train.py
```
*Processes the dataset and exports `model_artifacts/kickstarter_pipeline.joblib` in ~3 seconds.*

### 5. Launch Application
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

* **Frontend Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* **Swagger API:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## ☁️ 8. Cloud Deployment Guide (Render)

This project is configured for 1-click automated deployment on **Render**:

1. Fork or push this repository to your GitHub account.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New +** ➔ **Web Service**.
3. Select your repository.
4. Render detects the settings from [`render.yaml`](file:///d:/CLG/Mihir/AIML/PROJECT/render.yaml):
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt && python train.py`
   * **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   * **Plan:** `Free`
5. Click **Create Web Service**. Your app is live with SSL in ~2 minutes!

---

## ⚖️ 9. Project Limitations & Ethical Governance

* **Omitted Confounders:** While metadata provides strong predictive signals, success is also influenced by unobserved external variables:
  * Prototype video production quality and storytelling.
  * Size of the creator's pre-launch email waitlist and social media following.
  * Paid advertising spend on social platforms during launch week.
* **Economic & Distributional Drift:** Consumer spending habits and platform algorithms evolve over time. Regular retraining using the provided `train.py` script is recommended to maintain calibration.

---

## 👤 Author
* **Project:** Applied Machine Learning Semester 1 Project
* **Repository:** [Mihir1234567/AIML_Sem1_Project](https://github.com/Mihir1234567/AIML_Sem1_Project)
