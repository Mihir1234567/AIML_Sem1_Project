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
    
    # Financial target insights
    if goal > 25000 and prob < 0.50:
        advice.append(f"High Funding Target Alert: Campaigns with goals > $25,000 have an average failure rate over 65%. Consider an initial modest target of ${(goal * 0.5):,.0f} with stretch goals.")
    elif goal > 50000:
        advice.append("Large budget requirement: Ensure your pre-launch email waitlist covers at least 30-40% of the funding goal before going live.")
        
    # Duration insights
    if duration > 35:
        advice.append(f"Extended Duration Warning: 30-day campaigns historically perform better than {int(duration)}-day campaigns by maintaining momentum and reducing donor fatigue.")
    elif duration < 14:
        advice.append("Short Duration: Less than 2 weeks leaves limited window for algorithmic discovery on Kickstarter's explore page.")
    else:
        advice.append("Optimal Duration: 25-35 days is the sweet spot for crowdfunding momentum.")
        
    # Title analysis
    words = title.split()
    if len(words) < 3:
        advice.append("Brief Title: Descriptive titles with 5-10 words detailing key benefits or product type see higher click-through rates.")
    if title.isupper():
        advice.append("Avoid ALL CAPS in title: It triggers spam heuristics and can reduce creator credibility.")
    if '!' in title:
        advice.append("Exclamation mark detected: Avoid over-hyped punctuation in titles.")
        
    # Category notes
    if category.lower() in ['technology', 'journalism', 'crafts'] and prob < 0.55:
        advice.append(f"Competitive Category: {category} has historically tight success margins. Emphasize a working prototype and transparent manufacturing timelines.")

    if not advice or (prob >= 0.70 and len(advice) <= 1):
        advice.insert(0, "🌟 Excellent Configuration: Your campaign parameters fall within historically high-performing cohorts.")
        
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
