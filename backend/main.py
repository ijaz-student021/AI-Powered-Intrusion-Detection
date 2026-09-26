from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os
import json
from . import model_loader
from .routes import predict, predict_batch

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SecureNet Corp. AI Intrusion Detection API")

# Wildcard origins + credentials is rejected by browsers anyway, and this API
# doesn't use cookies, so origins are restricted and credentials disabled.
_default_origins = "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:8000,http://localhost:8000"
allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
if os.path.exists(FIGURES_DIR):
    app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")

@app.on_event("startup")
async def startup_event():
    model_loader.load_artifacts()

@app.get("/api/health")
def health_check():
    return model_loader.get_health_status()

def _two_stage_metrics():
    path = os.path.join(BASE_DIR, "model", "twostage_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@app.get("/api/models")
def get_models():
    models = []
    m = _two_stage_metrics()
    if m and "twostage" in model_loader.MODELS:
        models.append({"id": "twostage", "name": m["name"], "test_accuracy": m["test_accuracy"],
                       "test_weighted_f1": m["test_weighted_f1"]})
    models += [
        {"id": "lightgbm", "name": "LightGBM", "test_accuracy": 0.71, "test_weighted_f1": 0.76},
        {"id": "mlp", "name": "MLP Neural Network", "test_accuracy": 0.67, "test_weighted_f1": 0.73},
    ]
    return models

@app.get("/api/metrics")
def get_metrics():
    """Detailed test-set metrics of the two-stage model (binary + per-class)."""
    return _two_stage_metrics() or {}

@app.get("/api/feature-importance")
def feature_importance(top: int = 10):
    """Top feature importances (split counts) of the attack detector, for the Visualizations page."""
    ts = model_loader.MODELS.get("twostage")
    if ts is not None:
        names, imp = ts.columns, ts.detector.feature_importances_
    else:
        pipe = model_loader.MODELS.get("lightgbm")
        if pipe is None or not model_loader.feature_columns:
            return []
        names, imp = model_loader.feature_columns, pipe.named_steps["classifier"].feature_importances_
    pairs = sorted(zip(names, imp), key=lambda x: -x[1])[:max(1, min(top, 30))]
    total = float(sum(imp)) or 1.0
    return [{"feature": f, "importance": round(float(v) / total, 4)} for f, v in pairs]

app.include_router(predict.router, prefix="/api")
app.include_router(predict_batch.router, prefix="/api")
