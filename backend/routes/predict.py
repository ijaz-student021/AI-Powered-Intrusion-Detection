from fastapi import APIRouter, Depends, HTTPException, Request, Query
from typing import Optional
import logging
from .. import schemas
from .. import model_loader
from .. import preprocessing
from ..security import require_api_key
import pandas as pd
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)

def get_prediction_result(pipeline, processed_df):
    proba = pipeline.predict_proba(processed_df)[0]
    top_3_indices = np.argsort(proba)[::-1][:3]
    top_3 = []
    for idx in top_3_indices:
        cls_name = model_loader.class_names[idx]
        top_3.append({
            "class_name": cls_name,
            "probability": float(proba[idx])
        })
        
    pred_idx = top_3_indices[0]
    predicted_class = model_loader.class_names[pred_idx]
    confidence = float(proba[pred_idx])
    
    is_attack = (predicted_class != "Normal")
    low_confidence_category = predicted_class in ["Backdoor", "Analysis"]
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_3": top_3,
        "is_attack": is_attack,
        "low_confidence_category": low_confidence_category
    }

@router.post("/predict", dependencies=[Depends(require_api_key)])
async def predict(request: Request, model: str = Query("lightgbm")):
    if not model_loader.is_loaded:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")
        
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body.")
        
    try:
        df = pd.DataFrame([data])
        processed_df = preprocessing.preprocess_raw(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Preprocessing error")
        raise HTTPException(status_code=500, detail="Failed to preprocess the input record.")
        
    try:
        if model == "compare_all":
            results = {}
            pred_classes = set()
            for m_id, pipeline in model_loader.MODELS.items():
                res = get_prediction_result(pipeline, processed_df)
                results[m_id] = res
                pred_classes.add(res["predicted_class"])
                
            results["agreement"] = len(pred_classes) == 1
            return results
        else:
            if model not in model_loader.MODELS:
                raise HTTPException(status_code=400, detail="Invalid model selected.")
            
            pipeline = model_loader.MODELS[model]
            return get_prediction_result(pipeline, processed_df)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed.")
