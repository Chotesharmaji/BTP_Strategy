from pydantic import BaseModel
from typing import Dict, Tuple
from fastapi import FastAPI, HTTPException
import joblib
from pathlib import Path
import pandas as pd

# Import your Monte-Carlo simulation
from .mc_simulation import run_mc_simulation_uniform

app = FastAPI()

# Model path
MODEL_PATH = Path("models/rfr_model.joblib")
model = None


# -------------------------
# Load ML model at startup
# -------------------------
@app.on_event("startup")
def load_model():
    global model
    print("🔄 Loading model... (this may take some time)")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")


@app.get("/")
def root():
    return {"message": "Backend is running"}


@app.get("/ping")
def ping():
    return {"status": "ok", "model_loaded": model is not None}


# -------------------------
# Request body schema
# -------------------------
class SimulationRequest(BaseModel):
    left_sev: float
    right_sev: float
    brain_sev: float
    N: int
    bounds: Dict[str, Tuple[float, float]]


# -------------------------
# Main simulation endpoint
# -------------------------
@app.post("/simulate")
def simulate(req: SimulationRequest):
    try:
        print("📥 Received request:", req.dict())

        mc_samples, mc_predictions_df = run_mc_simulation_uniform(
            req.left_sev,
            req.right_sev,
            req.brain_sev,
            req.bounds,
            req.N,
            best_pipe=model
        )

        print("✅ Predictions shape:", mc_predictions_df.shape)

        # Convert samples → list of dicts (records)
        samples_records = mc_samples.to_dict(orient="records")

        # Convert predictions → list of lists (model output style)
        predictions_list = mc_predictions_df.values.tolist()

        # Final response (FULL compatibility for Streamlit scoring page)
        return {
            "samples_count": len(mc_samples),
            "samples": samples_records,        # <-- NEW
            "predictions": predictions_list
        }

    except Exception as e:
        print("❌ ERROR in simulate():", str(e))
        raise HTTPException(status_code=500, detail=str(e))
