from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from . import model_loader
from .routes import predict, predict_batch

app = FastAPI(title="SecureNet Corp. AI Intrusion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.get("/api/models")
def get_models():
    return [
        {"id": "lightgbm", "name": "LightGBM", "test_accuracy": 0.71, "test_weighted_f1": 0.76},
        {"id": "mlp", "name": "MLP Neural Network", "test_accuracy": 0.67, "test_weighted_f1": 0.73}
    ]

app.include_router(predict.router, prefix="/api")
app.include_router(predict_batch.router, prefix="/api")
