import pickle
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI

# load the pipeline
with open("pipeline_v2.bin", "rb") as f:
    pipeline = pickle.load(f)

# request schema
class Lead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lead_source: Literal["paid_ads","organic_search","referral","email","direct","NA"]
    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)

# response schema (optional; nice for docs)
class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool

app = FastAPI(title="lead-scoring")

@app.post("/predict", response_model=PredictResponse)
def predict(lead: Lead) -> PredictResponse:
    prob = float(pipeline.predict_proba([lead.model_dump()])[0, 1])
    return PredictResponse(churn_probability=prob, churn=prob >= 0.5)