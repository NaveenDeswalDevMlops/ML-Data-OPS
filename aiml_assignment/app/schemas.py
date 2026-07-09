from pydantic import BaseModel

class PredictionRequest(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    Partner: int

class PredictionResponse(BaseModel):
    churn_prediction: int
    probability: float
    model_used: str
