import os
import json
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODELS = {}
label_encoder = None
feature_columns = None
class_names = None
preprocessing_meta = None
is_loaded = False
raw_feature_columns = []

def load_artifacts():
    global MODELS, label_encoder, feature_columns, class_names, preprocessing_meta, is_loaded, raw_feature_columns
    try:
        pipeline_lgb = joblib.load(os.path.join(MODEL_DIR, "pipeline_lgb.pkl"))
        pipeline_mlp = joblib.load(os.path.join(MODEL_DIR, "pipeline_mlp.pkl"))
        
        MODELS = {}
        # Two-stage detector is the recommended model, so it is listed first.
        two_stage_path = os.path.join(MODEL_DIR, "pipeline_twostage.pkl")
        if os.path.exists(two_stage_path):
            from . import two_stage  # noqa: F401  (needed to unpickle TwoStageIDS)
            MODELS["twostage"] = joblib.load(two_stage_path)
        MODELS["lightgbm"] = pipeline_lgb
        MODELS["mlp"] = pipeline_mlp
        
        label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        
        with open(os.path.join(MODEL_DIR, "feature_columns.json"), "r") as f:
            feature_columns = json.load(f)
            
        with open(os.path.join(MODEL_DIR, "class_names.json"), "r") as f:
            class_names = json.load(f)
            
        with open(os.path.join(MODEL_DIR, "preprocessing_meta.json"), "r") as f:
            preprocessing_meta = json.load(f)
            
        raw_cols_set = set()
        raw_feature_columns = []
        cat_cols = preprocessing_meta.get("categorical_columns", [])
        
        for col in feature_columns:
            is_dummy = False
            for cat in cat_cols:
                if col.startswith(cat + "_"):
                    if cat not in raw_cols_set:
                        raw_cols_set.add(cat)
                        raw_feature_columns.append(cat)
                    is_dummy = True
                    break
            if not is_dummy:
                if col not in raw_cols_set:
                    raw_cols_set.add(col)
                    raw_feature_columns.append(col)
                    
        is_loaded = True
        print("Model artifacts loaded successfully.")
    except Exception as e:
        print(f"Failed to load model artifacts: {e}")
        is_loaded = False

def get_health_status():
    return {
        "is_loaded": is_loaded,
        "models_loaded": list(MODELS.keys()) if MODELS else [],
        "feature_columns": raw_feature_columns
    }
