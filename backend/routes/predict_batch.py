from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from .. import schemas
from .. import model_loader
from .. import preprocessing
import pandas as pd
import numpy as np
import io

router = APIRouter()

@router.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...), model: str = Query("lightgbm")):
    if not model_loader.is_loaded:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or malformed CSV file.")
        
    try:
        processed_df = preprocessing.preprocess_raw(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")
        
    try:
        total_rows = len(df)
        
        if model == "compare_all":
            all_results = []
            pipelines = model_loader.MODELS
            model_predictions = {}
            for m_id, pipeline in pipelines.items():
                probas = pipeline.predict_proba(processed_df)
                pred_indices = np.argmax(probas, axis=1)
                model_predictions[m_id] = {
                    "indices": pred_indices,
                    "probas": probas
                }
                
            for i in range(total_rows):
                row_res = {"row_index": i, "predictions": {}}
                pred_classes = set()
                
                for m_id, preds in model_predictions.items():
                    pred_idx = preds["indices"][i]
                    predicted_class = model_loader.class_names[pred_idx]
                    confidence = float(preds["probas"][i][pred_idx])
                    is_attack = (predicted_class != "Normal")
                    low_confidence = predicted_class in ["Backdoor", "Analysis"]
                    
                    row_res["predictions"][m_id] = {
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "is_attack": is_attack,
                        "low_confidence_category": low_confidence
                    }
                    pred_classes.add(predicted_class)
                
                row_res["agreement"] = len(pred_classes) == 1
                all_results.append(row_res)
                
            return {"results": all_results, "summary": {}}
            
        else:
            if model not in model_loader.MODELS:
                raise HTTPException(status_code=400, detail="Invalid model selected.")
                
            pipeline = model_loader.MODELS[model]
            probas = pipeline.predict_proba(processed_df)
            pred_indices = np.argmax(probas, axis=1)
            
            results = []
            summary_counts = {}
            
            for i in range(total_rows):
                pred_idx = pred_indices[i]
                predicted_class = model_loader.class_names[pred_idx]
                confidence = float(probas[i][pred_idx])
                
                is_attack = (predicted_class != "Normal")
                low_confidence = predicted_class in ["Backdoor", "Analysis"]
                
                results.append({
                    "row_index": i,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "is_attack": is_attack,
                    "low_confidence_category": low_confidence
                })
                
                summary_counts[predicted_class] = summary_counts.get(predicted_class, 0) + 1
                
            summary = {}
            for cls_name, count in summary_counts.items():
                summary[cls_name] = {
                    "count": count,
                    "percentage": round((count / total_rows) * 100, 2)
                }
                
            return {
                "results": results,
                "summary": summary
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
