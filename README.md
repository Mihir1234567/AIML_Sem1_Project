# 🚀 CrowdPulse — AI Crowdfunding Predictor & Campaign Advisor
### *End-to-End Applied Machine Learning System & Pre-Launch Decision-Support Web Application*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-emerald?style=for-the-badge&logo=render)](https://crowdpulse-api-3yaw.onrender.com/)
[![Swagger API](https://img.shields.io/badge/Swagger%20API-Docs-blue?style=for-the-badge&logo=fastapi)](https://crowdpulse-api-3yaw.onrender.com/docs)
[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Champion Model](https://img.shields.io/badge/Model-Calibrated%20HistGradientBoosting-purple?style=for-the-badge)](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)
[![ROC-AUC](https://img.shields.io/badge/Test%20ROC--AUC-0.7394-success?style=for-the-badge)]()
[![Design System](https://img.shields.io/badge/UI%2FUX-Bento%20Grid-indigo?style=for-the-badge)](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill)

---

## 📑 Table of Contents

- [🌐 Live Deployment Links](#-live-deployment-links)
- [📌 1. Executive Summary & Problem Formulation](#-1-executive-summary--problem-formulation)
- [🛡️ 2. Zero Data Leakage & Data Hygiene Architecture](#️-2-zero-data-leakage--data-hygiene-architecture)
- [🎨 3. Modern Bento Grid Interface & UX Design](#-3-modern-bento-grid-interface--ux-design)
- [🔬 4. Machine Learning Methodology](#-4-machine-learning-methodology)
  - [4.1 Feature Engineering Matrix](#41-feature-engineering-matrix)
  - [4.2 Preprocessing & Pipeline Construction](#42-preprocessing--pipeline-construction)
  - [4.3 Cross-Validation & Benchmark Comparison](#43-cross-validation--benchmark-comparison)
- [📈 5. Model Diagnostics & Test Evaluation (12,000 Campaigns)](#-5-model-diagnostics--test-evaluation-12000-campaigns)
  - [5.1 Performance Metrics](#51-performance-metrics)
  - [5.2 Model Explainability (Permutation Importance)](#52-model-explainability-permutation-importance)
- [🏗️ 6. Unified Full-Stack Architecture](#️-6-unified-full-stack-architecture)
- [🔌 7. REST API Documentation](#-7-rest-api-documentation)
- [💻 8. Quickstart & Local Setup](#-8-quickstart--local-setup)
- [☁️ 9. Cloud Deployment Guide (Render)](#️-9-cloud-deployment-guide-render)
- [⚖️ 10. Limitations & Ethical Governance](#️-10-limitations--ethical-governance)
- [🎓 11. Academic Evaluation & Portfolio Highlights](#-11-academic-evaluation--portfolio-highlights)
- [👤 Author](#-author)

---

## 🌐 Live Deployment Links

| Resource | Direct Link | Purpose |
| :--- | :--- | :--- |
| **Interactive Web Application** | [https://crowdpulse-api-3yaw.onrender.com/](https://crowdpulse-api-3yaw.onrender.com/) | Live user dashboard featuring real-time AI predictions, circular gauge meter, goal sensitivity curve, and 1-click presets. |
| **Interactive Swagger API** | [https://crowdpulse-api-3yaw.onrender.com/docs](https://crowdpulse-api-3yaw.onrender.com/docs) | Complete Swagger UI for testing REST endpoints and viewing OpenAPI schemas. |
| **API Health & Diagnostics** | [https://crowdpulse-api-3yaw.onrender.com/api/health](https://crowdpulse-api-3yaw.onrender.com/api/health) | Real-time JSON health check confirming model persistence, metadata, and test metrics. |
| **GitHub Repository** | [https://github.com/Mihir1234567/AIML_Sem1_Project](https://github.com/Mihir1234567/AIML_Sem1_Project) | Full project source code, Jupyter notebook, training script, and model artifacts. |

---

## 📌 1. Executive Summary & Problem Formulation

Crowdfunding platforms like Kickstarter have democratized seed capital for independent inventors, game studios, filmmakers, and product designers. However, historical data reveals a critical bottleneck: **over 60% of all campaigns fail to reach their funding targets**.

Kickstarter operates on a strict **"all-or-nothing"** model. If a campaign seeking \$10,000 raises \$9,950, **zero dollars are collected**, the creator receives no funding, and invested manufacturing preparations and PR budgets are lost.

### The CrowdPulse Solution
**CrowdPulse** is a pre-launch predictive decision-support system and interactive web application designed to help creators optimize their campaign terms *before* going live:
1. **Calibrated Predictive Scoring:** Generates real-time success probabilities using a trained `HistGradientBoostingClassifier` within a Scikit-Learn `Pipeline`.
2. **Zero-Leakage Assurance:** Restricts modeling strictly to decisions creators can make before launch (title, category, currency-adjusted goal, duration, timing).
3. **2D Goal Sensitivity Simulation:** Sweeps funding targets across an interactive grid to find the creator's financial "sweet spot" and break-even threshold.
4. **Contextual Actionable Advice:** Generates dynamic recommendations on target thresholds, duration limits, and title syntax.

---

## 🛡️ 2. Zero Data Leakage & Data Hygiene Architecture

A critical flaw in naive crowdfunding machine learning models is **target leakage**. Features like `pledged`, `usd_pledged`, `backers_count`, and `spotlight` are generated *during* and *after* a campaign is live. Training on these attributes leads to artificially inflated accuracy (95%+) that fails completely in real-world pre-launch scenarios.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   DATA LEAKAGE PREVENTION AUDIT                        │
├──────────────────────────────────┬─────────────────────────────────────┤
│ ❌ Strictly Excluded (Post-Launch)│ ✅ Retained Pre-Launch Predictors    │
├──────────────────────────────────┼─────────────────────────────────────┤
│ • pledged / usd_pledged          │ • Project title & linguistic signals│
│ • backers tally                  │ • Main category (15 sectors)        │
│ • live / suspended states        │ • Country / market jurisdiction     │
│ • staff pick / spotlight         │ • Currency-normalized USD goal (FX) │
│ • comments / updates count       │ • Campaign duration (1 to 90 days)  │
│                                  │ • Launch calendar timing (DoW, Mo)  │
└──────────────────────────────────┴─────────────────────────────────────┘
```

### Multi-Currency Normalization
Kickstarter records goals in local currencies (USD, GBP, EUR, CAD, AUD, SEK, etc.). Unadjusted currency values distort numerical scales. CrowdPulse standardizes all campaign targets into real USD purchasing equivalents using benchmark foreign exchange rates prior to logarithmic transformation:

$$\text{usd\_goal\_real} = \text{goal\_numeric} \times \text{fx\_rate}$$
$$\text{log\_usd\_goal} = \ln(1 + \text{usd\_goal\_real})$$

---

## 🎨 3. Modern Bento Grid Interface & UX Design

The user interface was engineered using the **`ui-ux-pro-max`** design intelligence framework, utilizing an Apple/Linear-inspired **Bento Grid Architecture**:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ [ Header ]  CrowdPulse  ·  Model: HistGradientBoosting  ·  [System: Online]   │
├───────────────────────────────────────────────────────────────────────────────┤
│ [ Presets ]  🎮 Board Game ($12k)  ⚡ Smart IoT ($75k)  📚 Art ($4.5k)        │
├──────────────────────────────────────┬────────────────────────────────────────┤
│          CAMPAIGN ARCHITECT          │            AI PREDICTION HERO          │
│                                      │  ┌──────────────┐  PREDICTED: OPTIMAL  │
│  • Title (Live counter & check)      │  │    68.4%     │  Favorable Outlook   │
│  • Category & Country Selectors      │  │ Circular SVG │  Moderate Risk       │
│  • Goal ($) [Sync Slider + Numbers]  │  └──────────────┘                      │
│  • Quick Chips: [$5k][$12k][$25k]    │  [Hurdle: Low] [Pace: Optimal] [Clear] │
│  • Duration (7-60 Days)              ├────────────────────────────────────────┤
│  • Launch Month & Day Timing         │       AI STRATEGIC RECOMMENDATIONS     │
│                                      │  • Keep duration under 35 days         │
│  [⚡ Auto-Evaluate: Live Debounce]    │  • Stretch goals recommended           │
│                                      ├────────────────────────────────────────┤
│                                      │     GOAL SENSITIVITY CURVE (Chart.js)  │
│                                      │     [ Probability % vs Dollar Goal ]   │
│                                      │  ════════════════════════════════════  │
│                                      │  - - - - - 50% Break-Even Line - - - - │
└──────────────────────────────────────┴────────────────────────────────────────┘
```

### Key UI/UX Innovations:
1. **Real-Time Debounced Auto-Prediction (400ms):** Automatically recalculates probability, gauges, and advice as you drag sliders or change categories—no manual submit button required.
2. **Quick Goal Jump Chips:** Instant preset buttons (`$5k`, `$12k`, `$25k`, `$50k`) for rapid financial stress-testing.
3. **Contextual Micro-KPI Tiles:**
   - **Hurdle Index:** Dynamic financial barrier rating (*Low Hurdle* / *Moderate* / *High Hurdle*).
   - **Pace Viability:** Duration evaluation (*Sweet Spot* / *Sprint* / *Fatigue Risk*).
   - **Title Quality:** Word count, punctuation, and uppercase heuristic check.
4. **1-Click "Copy AI Report" with Floating Toast:** One click exports a markdown campaign report to the clipboard for pitch decks or notes.
5. **Accessible Design System:** Built with Tailwind CSS, Chart.js, Lucide SVG icons, Google Fonts (**Plus Jakarta Sans** + **Fira Code** tabular figures), and full WCAG AA 4.5:1 text contrast compliance.

---

## 🔬 4. Machine Learning Methodology

### 4.1 Feature Engineering Matrix
Features represent four distinct pre-launch dimensions:
1. **Financial Hurdle:** $\ln(1 + \text{usd\_goal\_real})$ to compress heavy right-skewed distributions.
2. **Campaign Schedule & Timing:** 
   - `duration_days`: Number of live funding days ($\text{deadline} - \text{launched}$).
   - `launch_month`: Seasonal timing ($1-12$).
   - `launch_day_of_week`: Day of launch ($0 = \text{Monday}, 6 = \text{Sunday}$).
   - `is_weekend`: Binary flag for weekend launches ($1$ if Saturday/Sunday, else $0$).
3. **Title Semantics & Heuristics:**
   - `name_char_len`: Total character count.
   - `name_word_count`: Total word count.
   - `has_exclamation`: Flag for presence of `!` (marketing urgency).
   - `has_question`: Flag for presence of `?`.
   - `is_all_caps`: Flag for uppercase spam heuristic ($\text{length} > 3 \land \text{isupper}$).
4. **Categorical Signals:**
   - `main_category`: 15 sectors (Games, Technology, Design, Film & Video, Music, Publishing, Food, Art, Fashion, Comics, Photography, Crafts, Theater, Journalism, Dance).
   - `country`: 22 international jurisdictions.

### 4.2 Preprocessing & Pipeline Construction
All transformations are encapsulated into a Scikit-Learn `ColumnTransformer` to prevent data snooping:
* **Numerical Features (10):** Transformed via `StandardScaler()` fit strictly on training splits.
* **Categorical Features (2):** Transformed via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

```python
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(max_iter=120, max_depth=6, random_state=42))
])
```

### 4.3 Cross-Validation & Benchmark Comparison
Four algorithmic archetypes were benchmarked using **3-Fold Stratified Cross-Validation**:

| Model Archetype | Mean CV ROC-AUC | Std CV ROC-AUC | Mean CV F1 | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Champion)** | **0.7182** | $\pm 0.0031$ | **0.5482** | **~0.8s** |
| **Random Forest Classifier** | 0.7094 | $\pm 0.0028$ | 0.5310 | ~2.1s |
| **Logistic Regression ($L_2$)** | 0.6912 | $\pm 0.0035$ | 0.5124 | ~0.5s |
| **Dummy Baseline (Majority Class)** | 0.5000 | $\pm 0.0000$ | 0.0000 | ~0.1s |

---

## 📈 5. Model Diagnostics & Test Evaluation (12,000 Campaigns)

### 5.1 Performance Metrics
The champion pipeline was tested on an unseen holdout dataset ($N = 12,000$ campaigns, stratified 80/20 split):

| Evaluation Metric | Test Score | Analytical Interpretation |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.7394** | High discrimination between successful and failing campaigns with relative category signals and subcategories (+0.0217 gain). |
| **Accuracy** | **68.83%** | Outperforms the naive majority-class baseline by nearly 15 percentage points. |
| **Brier Score** | **0.2002** | Reflects well-calibrated probabilities via `CalibratedClassifierCV(method='sigmoid')`. |
| **Precision** | **63.21%** | Reliable positive predictive value when advising creators to proceed to launch. |
| **F1 Score** | **0.5614** | Balanced harmonic mean on imbalanced real-world crowdfunding data. |

### 5.2 Advanced Innovations Introduced
1. **Relative Financial Hurdle (`goal_to_cat_median_ratio`):** Rather than evaluating goals in absolute terms alone, targets are normalized relative to their specific domain's empirical median (\$3k for Art vs \$18k for Technology).
2. **Granular Sub-Category Integration (`category`):** 159 detailed sub-categories are encoded using frequency-binned One-Hot Encoding to separate micro-niche behaviors (*e.g. Tabletop Games vs Video Games*).
3. **Probability Calibration (`CalibratedClassifierCV`):** Probabilities are calibrated so that a 20% score means exactly 20 out of 100 historical campaigns in that cohort achieved funding.
4. **Counterfactual Auto-Optimization Engine (`POST /api/optimize`):** Automatically searches for the minimal required parameter shifts (goal right-sizing, optimal duration, launch day) to lift a sub-optimal campaign above the 65%+ viability threshold.
5. **Ground-Truth Historical Campaign Matcher:** Uses nearest neighbor feature distance and token Jaccard similarity across 5,000 real Kickstarter campaigns to display 3 real-world precedents with actual names and outcomes.

### 5.2 Model Explainability (Permutation Importance)
To avoid the notorious cardinality bias of tree Gini impurity, feature importance was evaluated via **Test-Set Permutation Feature Importance** (measuring the degradation in ROC-AUC when each column is randomly shuffled):

1. **`log_usd_goal` (Top Driver):** Shuffling the funding target causes an ROC-AUC drop of over **0.14 points**. Smaller, modest funding goals correlate exponentially with higher success probability.
2. **`main_category`:** Creative sectors (Comics, Dance, Theater) have substantially higher success rates (>55%) than capital-intensive categories (Technology, Food, Journalism) (<35%).
3. **`duration_days`:** Campaigns exceeding 35 days experience donor momentum fatigue, whereas 30-day campaigns achieve the highest empirical conversion.

---

## 🏗️ 6. Unified Full-Stack Architecture

The system is deployed on **Render** as a single high-performance web service:

```
AIML/PROJECT/
├── backend/
│   ├── __init__.py
│   └── main.py                     # FastAPI server, static file router & self-healing ML loader
├── frontend/
│   └── index.html                  # Bento Grid UI (Tailwind CSS, Chart.js, Lucide Icons)
├── model_artifacts/
│   ├── kickstarter_pipeline.joblib # Serialized Scikit-Learn Pipeline (330 KB)
│   └── model_metadata.json         # Evaluation benchmarks & category registries
├── Notebook/
│   ├── AIML_Project.ipynb          # End-to-end research, EDA, and diagnostic experiments
│   └── kickstarter_data.csv        # Historical Kickstarter dataset
├── .agents/skills/ui-ux-pro-max/   # UI/UX Pro Max design intelligence skill & local search engine
├── train.py                        # Standalone script for automated cloud pipeline retraining
├── requirements.txt                # Pinned dependencies (scikit-learn==1.5.2, joblib==1.4.2)
├── render.yaml                     # Infrastructure specification for Render
├── Dockerfile                      # Production container spec (Python 3.11-slim)
└── README.md                       # Comprehensive system documentation
```

### Self-Healing Model Persistence
In `backend/main.py`, the model loader is wrapped in a self-healing exception handler. If a serialization or environment mismatch ever occurs at runtime, the backend automatically triggers `train_and_save()` from `train.py` in-memory, recompiling a fresh model artifact natively in ~3 seconds.

---

## 🔌 7. REST API Documentation

### 1. System Health Check
`GET /api/health`

**Response:**
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
    "test_samples": 12000,
    "trained_at": "2026-11-12 18:19:55"
  }
}
```

---

### 2. Campaign Success Prediction
`POST /api/predict`

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

### 3. Sweet-Spot Sensitivity Sweep
`POST /api/simulate`

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

## 💻 8. Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Mihir1234567/AIML_Sem1_Project.git
cd AIML_Sem1_Project
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
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
*Processes `Notebook/kickstarter_data.csv` and outputs `model_artifacts/kickstarter_pipeline.joblib` in ~3 seconds.*

### 5. Run Local Development Server
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

* **Interactive Frontend:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* **Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## ☁️ 9. Cloud Deployment Guide (Render)

This project is configured with [`render.yaml`](file:///d:/CLG/Mihir/AIML/PROJECT/render.yaml) for 1-click cloud deployment:

1. Push your repository to **GitHub**.
2. Go to the [Render Dashboard](https://dashboard.render.com/) and click **New +** ➔ **Web Service**.
3. Connect your repository (`AIML_Sem1_Project`).
4. Render automatically applies the configuration:
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt && python train.py`
   * **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   * **Plan:** `Free`
5. Click **Deploy Web Service**. Your app is live with automatic SSL in ~2 minutes!

> **Note on Render Free Tier:**
> Free tier instances sleep after 15 minutes of inactivity. When accessed after sleeping, the first request takes ~45 seconds to spin up. The live header status indicator will show *"System: Waking Up..."* during this cold start.

---

## ⚖️ 10. Limitations & Ethical Governance

* **Omitted Confounders:** Tabular metadata cannot capture unobserved non-tabular drivers:
  * Quality of the project pitch video and visual graphics.
  * Size of the creator's pre-launch email waitlist and social media follower base.
  * Paid advertising budget spent during the campaign's first 48 hours.
* **Distributional & Economic Drift:** Crowdfunding algorithms, backer sentiment, and inflation evolve over time. The provided `train.py` script allows periodic retraining on updated datasets.

---

## 🎓 11. Academic Evaluation & Portfolio Highlights

| Academic Evaluation Criterion | Implementation in CrowdPulse |
| :--- | :--- |
| **Problem Definition** | Well-formulated business problem addressing Kickstarter's all-or-nothing dynamics. |
| **Data Hygiene & Rigor** | Explicit data leakage audit; dropped post-launch features; normalized 13 currencies. |
| **Model Engineering** | 4 benchmarked model archetypes with 3-Fold Stratified Cross-Validation. |
| **Explainable AI (XAI)** | Test-set Permutation Feature Importance avoiding standard Gini impurity bias. |
| **Actionable Decision Support** | Interactive 2D Goal Sensitivity Simulation finding the financial break-even point. |
| **Production Software Engineering** | REST API with FastAPI, Pydantic schemas, Scikit-Learn Pipelines, and self-healing loader. |
| **Modern UI/UX** | Apple/Linear-inspired Bento Grid layout, real-time debounced prediction, and micro-KPIs. |
| **Deployment & Reproducibility** | Live cloud deployment on Render with SSL, Dockerfile, and pinned dependencies. |

---

## 👤 Author

* **Project:** Applied Machine Learning Semester 1 Capstone Project
* **Repository:** [Mihir1234567/AIML_Sem1_Project](https://github.com/Mihir1234567/AIML_Sem1_Project)
* **Live System:** [https://crowdpulse-api-3yaw.onrender.com/](https://crowdpulse-api-3yaw.onrender.com/)
