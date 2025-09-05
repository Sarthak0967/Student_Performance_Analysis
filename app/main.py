from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Load artifacts
model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
encoder = joblib.load(os.path.join(BASE_DIR, "encoder.joblib"))
feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.joblib"))

# Define FastAPI app
app = FastAPI(title="Student Performance Prediction API")

# Input schema (raw inputs before transformation)
class StudentData(BaseModel):
    Hours_Studied: float
    Attendance: float
    Sleep_Hours: float
    Previous_Scores: float
    Tutoring_Sessions: float
    Physical_Activity: float
    Parental_Involvement: str
    Access_to_Resources: str
    Motivation_Level: str
    Family_Income: str
    Teacher_Quality: str
    Peer_Influence: str
    Parental_Education_Level: str
    Distance_from_Home: str

categorical_mappings = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
    "Parental_Education_Level": ["High School", "College", "Post Graduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"]
}

# Define numerical features
numerical_features = [
    "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", 
    "Tutoring_Sessions", "Physical_Activity"
]

# Define the correct feature order used during training
feature_order = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources", 
    "Sleep_Hours", "Previous_Scores", "Motivation_Level", "Tutoring_Sessions", 
    "Family_Income", "Teacher_Quality", "Peer_Influence", "Physical_Activity", 
    "Parental_Education_Level", "Distance_from_Home", "Study_Efficiency", 
    "Improvement_Rate", "Tutoring_Effect"
]

@app.post("/predict")
def predict(data: StudentData):
    try:
        input_dict = data.dict()
        
        input_data = pd.DataFrame([input_dict])
        
        missing_features = [ feature for feature in numerical_features + list(categorical_mappings.keys()) if feature not in input_data]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {', '.join(missing_features)}")
        
        
        # Validate categorical features
        for feature, valid_values in categorical_mappings.items():
            value = input_dict[feature]   # ✅ extract single value
            if value not in valid_values:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value '{value}' for {feature}. Must be one of {valid_values}"
                )

        input_data["Study_Efficiency"] = input_data["Hours_Studied"] / (input_data["Attendance"] + 1)
        input_data["Improvement_Rate"] = input_data["Previous_Scores"] / (input_data["Hours_Studied"] + 1)
        input_data["Tutoring_Effect"] = input_data["Tutoring_Sessions"] / (input_data["Hours_Studied"] + 1)

        # Encode categorical features using predefined mappings
        for feature in categorical_mappings:
            input_data[feature] = categorical_mappings[feature].index(input_dict[feature])

        # Scale numerical features, including derived features
        numerical_features_with_derived = numerical_features + ["Study_Efficiency", "Improvement_Rate", "Tutoring_Effect"]
        input_data[numerical_features_with_derived] = scaler.transform(input_data[numerical_features_with_derived])

        # Ensure feature order matches training
        input_data = input_data[feature_order]

        # Make prediction
        prediction = model.predict(input_data)[0]
        
        print("DEBUG >>> New version of /predict is running")

        
        # Categorize result
        if prediction >= 85:
            status = "Excellent"
            recommendation = "Keep up the great work and maintain consistency."
        elif prediction >= 60:
            status = "Pass"
            recommendation = "Good performance, but focus on improving weak areas."
        else:
            status = "Fail"
            recommendation = "Needs improvement. Increase study hours and seek guidance."

        # ✅ Return more structured response
        return {
        "status": "success",
        "input": data.dict(),
        "prediction": float(prediction),
        "message": "Prediction generated successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"message": "Welcome to Student Performance Prediction API 🚀"}
