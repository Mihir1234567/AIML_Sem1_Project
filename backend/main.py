import os
import json
import joblib

# Set execution environment for joblib/loky stability
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# App initialization
app = FastAPI(
    title="CrowdPulse - Crowdfunding Success Predictor API",
    description="Machine Learning API for predicting Kickstarter campaign success and providing data-driven recommendations.",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Artifact Locations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "model_artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "kickstarter_pipeline.joblib")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "model_metadata.json")

# In-memory cached model and metadata
model_pipeline = None
metadata = {}

def get_model():
    global model_pipeline, metadata
    if model_pipeline is None:
        try:
            if not os.path.exists(MODEL_PATH):
                raise RuntimeError(f"Trained model not found at {MODEL_PATH}.")
            model_pipeline = joblib.load(MODEL_PATH)
        except Exception as load_err:
            print(f"⚠️ Model load failed ({load_err}). Triggering self-healing fallback...")
            try:
                import sys
                if BASE_DIR not in sys.path:
                    sys.path.insert(0, BASE_DIR)
                from train import train_and_save
                train_and_save()
                model_pipeline = joblib.load(MODEL_PATH)
                print("✅ Model successfully retrained natively and loaded.")
            except Exception as train_err:
                raise RuntimeError(f"Model load and auto-retrain both failed. Load error: {load_err}. Retrain error: {train_err}")
        
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
    return model_pipeline

# ----------------- Pydantic Request / Response Schemas -----------------
class CampaignInput(BaseModel):
    title: str = Field(..., example="Eco-friendly Modular Mechanical Keyboard")
    main_category: str = Field(..., example="Technology")
    country: str = Field(default="US", example="US")
    usd_goal: float = Field(..., gt=0, example=15000.0)
    duration_days: float = Field(..., ge=1, le=90, example=30.0)
    launch_month: int = Field(default=10, ge=1, le=12, example=10)
    launch_day_of_week: int = Field(default=1, ge=0, le=6, example=1) # 0=Mon, 1=Tue, 6=Sun

class SimulationRequest(BaseModel):
    title: str = Field(..., example="Smart Travel Backpack")
    main_category: str = Field(..., example="Design")
    country: str = Field(default="US", example="US")
    min_goal: float = Field(default=2000.0, gt=0)
    max_goal: float = Field(default=50000.0, gt=0)
    duration_days: float = Field(default=30.0, ge=1, le=90)
    steps: int = Field(default=15, ge=5, le=30)

class PredictionResponse(BaseModel):
    title: str
    main_category: str
    country: str
    usd_goal: float
    duration_days: float
    probability: float
    probability_pct: str
    prediction: str
    status: str
    risk_level: str
    actionable_advice: List[str]
    model_name: str

# ----------------- Advisory Logic -----------------
def generate_recommendations(goal: float, duration: float, prob: float, category: str, title: str) -> List[str]:
    advice = []
    
    # 1. Primary Risk & Probability Conditioning (Model-Grounded)
    if prob < 0.35:
        advice.append(
            f"⚠️ Critical Hurdle Alert ({prob*100:.1f}% Estimated Viability): Over 75% of campaigns matching this profile fail to reach their goal under Kickstarter's all-or-nothing model. Do not launch without revising scope."
        )
        if goal > 10000:
            advice.append(
                f"Financial Restructuring: Consider lowering your initial goal to ${(goal * 0.5):,.0f} and utilizing stretch goals. Smaller initial targets trigger Kickstarter's trending algorithms faster."
            )
        else:
            advice.append(
                f"Audience Pre-Launch Deficit: For a ${goal:,.0f} goal in {category}, ensure you build a pre-launch backer waitlist covering at least 30-40% of the funds before day 1."
            )
    elif prob < 0.50:
        advice.append(
            f"⚡ Below Viability Threshold ({prob*100:.1f}% Probability): This campaign is hovering in the risk zone. Parameter tuning and pre-launch audience prep are recommended."
        )
        if goal > 15000:
            advice.append(
                f"Funding Right-Sizing: Campaigns in {category} have substantially higher success odds when the initial base target is under $12,000."
            )
    elif prob >= 0.70:
        advice.append(
            f"🌟 Strong Launch Profile ({prob*100:.1f}% Probability): Your campaign parameters fall into historically high-performing cohorts."
        )
    else:
        advice.append(
            f"Moderate Launch Profile ({prob*100:.1f}% Probability): Balanced viability. Focus on strong launch-week momentum to cross the 50% tipping point early."
        )

    # 2. Duration Insights (Context-Aware)
    if duration > 35:
        advice.append(
            f"Extended Duration Warning: 30-day campaigns historically outperform {int(duration)}-day campaigns by maintaining momentum and preventing donor fatigue."
        )
    elif duration < 14:
        advice.append(
            "Short Duration Risk: Less than 14 days leaves limited runway for Kickstarter's organic discovery algorithm to feature your campaign."
        )
    elif prob >= 0.50:
        advice.append(
            "Optimal Duration: 25–35 days maintains peak crowdfunding urgency and backer engagement."
        )

    # 3. Category Sector Dynamics
    hard_categories = ['technology', 'journalism', 'crafts', 'food']
    if category.lower() in hard_categories and prob < 0.55:
        advice.append(
            f"Category Friction: {category} projects face tight margins. Prioritize showing a working prototype, transparent BOM/manufacturing costs, and realistic shipping dates."
        )

    # 4. Title & Semantic Cues
    words = title.strip().split()
    if len(words) < 3:
        advice.append(
            "Title Polish: Descriptive titles (5–9 words) detailing clear backer benefits achieve higher click-through rates."
        )
    if title.isupper() and len(title) > 3:
        advice.append(
            "Formatting Fix: Avoid ALL CAPS titles. It triggers spam heuristics and reduces backer credibility."
        )
    if '!' in title:
        advice.append(
            "Punctuation Fix: Remove exclamation marks. Organic backer response favors clear benefit statements over promotional hype."
        )

    return advice

# ----------------- API Endpoints -----------------
@app.get("/api/health")
def health_check():
    try:
        get_model()
        return {"status": "healthy", "model_loaded": True, "metadata": metadata}
    except Exception as e:
        return {"status": "degraded", "model_loaded": False, "error": str(e)}

@app.get("/api/metadata")
def get_metadata():
    get_model()
    return metadata

@app.post("/api/predict", response_model=PredictionResponse)
def predict_campaign(campaign: CampaignInput):
    model = get_model()
    
    # Construct feature DataFrame
    row = pd.DataFrame([{
        'log_usd_goal': np.log1p(campaign.usd_goal),
        'duration_days': campaign.duration_days,
        'launch_month': campaign.launch_month,
        'launch_day_of_week': campaign.launch_day_of_week,
        'is_weekend': int(campaign.launch_day_of_week in [5, 6]),
        'name_char_len': len(campaign.title),
        'name_word_count': len(campaign.title.split()),
        'has_exclamation': int('!' in campaign.title),
        'has_question': int('?' in campaign.title),
        'is_all_caps': int(campaign.title.isupper() and len(campaign.title) > 3),
        'main_category': campaign.main_category,
        'country': campaign.country
    }])
    
    try:
        prob = float(model.predict_proba(row)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
    prediction = "Successful" if prob >= 0.50 else "Likely to Fail"
    
    if prob >= 0.75:
        risk_level = "Low Risk (High Confidence)"
    elif prob >= 0.50:
        risk_level = "Moderate Risk (Viable)"
    elif prob >= 0.30:
        risk_level = "High Risk (Sub-optimal)"
    else:
        risk_level = "Critical Risk (Severe Hurdles)"
        
    advice = generate_recommendations(
        goal=campaign.usd_goal,
        duration=campaign.duration_days,
        prob=prob,
        category=campaign.main_category,
        title=campaign.title
    )
    
    return PredictionResponse(
        title=campaign.title,
        main_category=campaign.main_category,
        country=campaign.country,
        usd_goal=campaign.usd_goal,
        duration_days=campaign.duration_days,
        probability=round(prob, 4),
        probability_pct=f"{prob * 100:.1f}%",
        prediction=prediction,
        status="Optimal" if prob >= 0.50 else "Needs Refinement",
        risk_level=risk_level,
        actionable_advice=advice,
        model_name=metadata.get("model_name", "HistGradientBoostingClassifier")
    )

@app.post("/api/simulate")
def simulate_goals(req: SimulationRequest):
    model = get_model()
    goals = np.linspace(req.min_goal, req.max_goal, req.steps)
    results = []
    
    for g in goals:
        row = pd.DataFrame([{
            'log_usd_goal': np.log1p(g),
            'duration_days': req.duration_days,
            'launch_month': 10,
            'launch_day_of_week': 1,
            'is_weekend': 0,
            'name_char_len': len(req.title),
            'name_word_count': len(req.title.split()),
            'has_exclamation': int('!' in req.title),
            'has_question': int('?' in req.title),
            'is_all_caps': int(req.title.isupper() and len(req.title) > 3),
            'main_category': req.main_category,
            'country': req.country
        }])
        prob = float(model.predict_proba(row)[0][1])
        results.append({
            "goal": round(float(g), 2),
            "probability": round(prob * 100, 1)
        })
        
    return {
        "title": req.title,
        "category": req.main_category,
        "duration_days": req.duration_days,
        "curve": results
    }

# Serve static frontend files if folder exists
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
