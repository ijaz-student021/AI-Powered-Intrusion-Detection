from pydantic import BaseModel
from typing import List, Dict, Any

class TopClass(BaseModel):
    class_name: str
    probability: float

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_3: List[TopClass]
    is_attack: bool
    low_confidence_category: bool

class BatchPredictionResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Dict[str, Any]]
