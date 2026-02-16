# src/api/schema.py

from pydantic import BaseModel, Field
from typing import Optional


class StrokeInput(BaseModel):
    """Input schema matching the Kaggle Stroke Prediction Dataset features."""

    gender: str = Field(..., example="Male", description="Male, Female, or Other")
    age: float = Field(..., example=67.0, ge=0, le=120)
    hypertension: int = Field(..., example=0, ge=0, le=1)
    heart_disease: int = Field(..., example=1, ge=0, le=1)
    ever_married: str = Field(..., example="Yes", description="Yes or No")
    work_type: str = Field(
        ..., example="Private",
        description="Private, Self-employed, Govt_job, children, or Never_worked",
    )
    Residence_type: str = Field(..., example="Urban", description="Urban or Rural")
    avg_glucose_level: float = Field(..., example=228.69, ge=0)
    bmi: Optional[float] = Field(None, example=36.6, ge=0, le=100)
    smoking_status: str = Field(
        ..., example="formerly smoked",
        description="formerly smoked, never smoked, smokes, or Unknown",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "age": 67.0,
                "hypertension": 0,
                "heart_disease": 1,
                "ever_married": "Yes",
                "work_type": "Private",
                "Residence_type": "Urban",
                "avg_glucose_level": 228.69,
                "bmi": 36.6,
                "smoking_status": "formerly smoked",
            }
        }


class StrokeOutput(BaseModel):
    """Output schema for the /predict endpoint."""

    prediction: int = Field(..., description="0 = no stroke, 1 = stroke")
    probability_stroke: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of stroke (class 1)"
    )
