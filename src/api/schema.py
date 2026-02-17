# src/api/schema.py

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class StrokeInput(BaseModel):
    """Input schema matching the new Stroke Prediction Dataset features."""

    Gender: str = Field(..., examples=["Male"], description="Male, Female, or Other")
    Age: float = Field(..., examples=[67.0], ge=0, le=120)
    SES: str = Field(..., examples=["Low"], description="Low, Medium, or High")
    Hypertension: int = Field(..., examples=[0], ge=0, le=1)
    Heart_Disease: int = Field(..., examples=[1], ge=0, le=1)
    Avg_Glucose: float = Field(..., examples=[228.69], ge=0)
    BMI: Optional[float] = Field(None, examples=[36.6], ge=0, le=100)
    Diabetes: int = Field(..., examples=[0], ge=0, le=1)
    Smoking_Status: str = Field(
        ..., examples=["Former"],
        description="Never, Former, Current, or Unknown",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Gender": "Male",
                "Age": 67.0,
                "SES": "Low",
                "Hypertension": 0,
                "Heart_Disease": 1,
                "Avg_Glucose": 228.69,
                "BMI": 36.6,
                "Diabetes": 0,
                "Smoking_Status": "Former",
            }
        }
    )


class StrokeOutput(BaseModel):
    """Output schema for the /predict endpoint."""

    prediction: int = Field(..., description="0 = no stroke, 1 = stroke")
    probability_stroke: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of stroke (class 1)"
    )
