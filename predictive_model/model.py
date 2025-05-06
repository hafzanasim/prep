import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create dummy model on first load
def train_dummy_model():
    # Example dummy dataset
    X = [
        [3.0, 1],  # small tumor, early stage
        [5.0, 2],  # medium tumor, mid stage
        [7.0, 3],  # large tumor, late stage
    ]
    y = [0, 1, 1]  # 0 = low risk, 1 = high risk

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Map stage string (e.g., IIA, IIIB) to a numeric score
def stage_to_numeric(stage_str):
    mapping = {
        "I": 1, "IA": 1, "IB": 1,
        "II": 2, "IIA": 2, "IIB": 2,
        "III": 3, "IIIA": 3, "IIIB": 3,
        "IV": 4
    }
    return mapping.get(stage_str.upper(), 0)

# Make prediction based on extracted features
def predict_survival_risk(tumor_size_cm, stage):
    if tumor_size_cm is None or stage is None:
        return "Insufficient data"

    stage_score = stage_to_numeric(stage)
    model = train_dummy_model()

    input_features = np.array([[tumor_size_cm, stage_score]])
    prediction = model.predict(input_features)[0]
    risk = "High Risk" if prediction == 1 else "Low Risk"
    return risk
