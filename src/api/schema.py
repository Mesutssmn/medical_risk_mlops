# src/api/schema.py

<<<<<<< HEAD
from pydantic import BaseModel, Field, ConfigDict
=======
from pydantic import BaseModel, Field
>>>>>>> 459246e18e43c05e5cd1f76d2d37b0bcb0ea7965
from typing import Optional


class StrokeInput(BaseModel):
<<<<<<< HEAD
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
=======
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
>>>>>>> 459246e18e43c05e5cd1f76d2d37b0bcb0ea7965


class StrokeOutput(BaseModel):
    """Output schema for the /predict endpoint."""

    prediction: int = Field(..., description="0 = no stroke, 1 = stroke")
    probability_stroke: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of stroke (class 1)"
    )
