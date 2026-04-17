import bentoml
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pydantic import BaseModel

# Import the existing RiskPredictor
from inference.predict import RiskPredictor

# Define Pydantic models for BentoML IO
class RiskInput(BaseModel):
    features: Dict[str, float]

class RiskOutput(BaseModel):
    lgbm_prob: float
    gru_prob: float
    ensemble_prob: float
    anomaly_flag: bool
    risk_level: str
    human_explanation: str
    shap_top3: List[Dict[str, Any]]

@bentoml.service(
    name="bankriskservice",
    traffic={"timeout": 60}
)
class bankriskservice:
    def __init__(self):
        print("Initializing BankRiskService (BentoML)...")
        self.predictor = RiskPredictor()

    @bentoml.api
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        API Endpoint for real-time risk scoring.
        Accepts a dictionary of behavioral features and returns multi-model risk scores.
        """
        # If input is wrapped in 'features' key, extract it
        features = input_data.get("features", input_data)
        
        result = self.predictor.predict_from_features(features)
        
        # Ensure result is JSON serializable
        return result

    @bentoml.api
    def predict_customer(self, customer_id: str, week_number: int = 52) -> Dict[str, Any]:
        """
        API Endpoint for customer-based lookup.
        """
        result = self.predictor.predict_single(customer_id, week_number)
        return result

    @bentoml.api
    def health(self) -> Dict[str, bool]:
        return {"status": "healthy", "models_loaded": True}
