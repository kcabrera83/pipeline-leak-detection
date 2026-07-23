"""FastAPI web server for pipeline leak detection."""

import pickle
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Pipeline Leak Detection",
    description="Pipeline leak classification, severity estimation, and batch analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models: dict[str, Any] = {}


@app.on_event("startup")
async def load_models():
    from pipeline_leak.models.leak_classifier import LeakClassifier
    from pipeline_leak.models.leak_size_estimator import LeakSizeEstimator
    try:
        models["classifier"] = LeakClassifier.load("outputs/models/leak_classifier.pkl")
        models["size_estimator"] = LeakSizeEstimator.load("outputs/models/leak_size_estimator.pkl")
        with open("outputs/models/preprocessor.pkl", "rb") as f:
            models["preprocessor"] = pickle.load(f)
    except Exception as e:
        print(f"  Error loading models: {e}")


class LeakPredictRequest(BaseModel):
    pipeline_type: str = "crude_oil"
    pipeline_length_km: float = 50.0
    pipeline_diameter_mm: float = 200.0
    pressure_upstream_mpa: float = 5.0
    pressure_downstream_mpa: float = 4.0
    flow_rate_m3h: float = 200.0
    temperature_c: float = 25.0
    ambient_temp_c: float = 20.0
    soil_moisture_pct: float = 40.0
    pipe_wall_thickness_mm: float = 12.0
    pressure_drop_mpa: float = 1.0
    flow_anomaly_m3h: float = 0.0
    acoustic_emission_db: float = 10.0
    temperature_diff_c: float = 1.0
    vibration_level_g: float = 0.3


class BatchReadings(BaseModel):
    readings: list[dict]


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "pipeline-leak-detection"}


@app.get("/api/models")
async def api_models():
    if "classifier" not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "status": "ok",
        "classifier": models["classifier"].best_name,
        "size_estimator": "trained" if models["size_estimator"].trained else "not_trained",
    }


@app.post("/api/predict")
async def api_predict(request: LeakPredictRequest):
    if not all(k in models for k in ("classifier", "size_estimator", "preprocessor")):
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        df = pd.DataFrame([request.model_dump()])
        X = models["preprocessor"].transform(df)
        has_leak = int(models["classifier"].predict(X)[0])
        proba = models["classifier"].predict_proba(X)[0]
        result = {
            "status": "ok",
            "has_leak": bool(has_leak),
            "leak_probability": round(float(max(proba)), 4),
            "leak_class": "LEAK DETECTED" if has_leak else "NO LEAK",
        }
        if has_leak:
            size_pred = float(models["size_estimator"].predict(X)[0])
            result["leak_severity"] = (
                "critical" if size_pred > 0.8 else
                "high" if size_pred > 0.5 else
                "medium" if size_pred > 0.3 else "low"
            )
            result["leak_size_score"] = round(size_pred, 4)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/batch")
async def api_batch(request: BatchReadings):
    if not all(k in models for k in ("classifier", "preprocessor")):
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        df = pd.DataFrame(request.readings)
        X = models["preprocessor"].transform(df)
        predictions = models["classifier"].predict(X)
        probas = models["classifier"].predict_proba(X)
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probas)):
            results.append({
                "index": i,
                "has_leak": bool(pred),
                "leak_probability": round(float(max(prob)), 4),
            })
        leak_count = sum(1 for r in results if r["has_leak"])
        return {
            "status": "ok",
            "total_readings": len(results),
            "leaks_detected": leak_count,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5005)
