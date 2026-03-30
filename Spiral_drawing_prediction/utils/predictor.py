import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image

# Load models
spiral_model = load_model("models/Final_spiral_ResNet101.h5")
wave_model = load_model("models/Final_wave_VGG19.h5")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------- SPIRAL ----------------
def interpret_severity(score):
    if score < 38.117172:
        return "Normal Pattern", "No motor impairment detected."
    elif score < 55:
        return "Mild", "Slight motor irregularities observed."
    elif score < 70:
        return "Moderate", "Noticeable motor impairment detected."
    elif score < 85:
        return "High", "Significant motor impairment observed."
    else:
        return "Severe", "Strong Parkinsonian motor patterns detected."


def predict_spiral(img_path):
    min_logit = -16.384981
    max_logit = 26.600843

    img = preprocess_image(img_path)
    logit = spiral_model.predict(img)[0][0]

    prob = sigmoid(logit)

    normalized = (logit - min_logit) / (max_logit - min_logit)
    normalized = np.clip(normalized, 0, 1)

    severity_score = normalized * 100
    severity_level, description = interpret_severity(severity_score)

    return {
        "type": "Spiral",
        "label": "Parkinson" if prob > 0.5 else "Healthy",
        "logit": float(logit),
        "probability": float(prob),
        "score": float(severity_score),
        "severity": severity_level,
        "description": description
    }


# ---------------- WAVE ----------------
def interpret_wave_severity(score):
    if score < 36.876736:
        return "Normal Pattern", "No motor impairment detected."
    elif score < 55:
        return "Mild", "Slight motor irregularities observed."
    elif score < 70:
        return "Moderate", "Noticeable motor impairment detected."
    elif score < 85:
        return "High", "Significant motor impairment observed."
    else:
        return "Severe", "Strong Parkinsonian motor patterns detected."


def predict_wave(img_path):
    min_logit = -45.584194
    max_logit = 78.02814

    img = preprocess_image(img_path)
    logit = wave_model.predict(img)[0][0]

    prob = sigmoid(logit)

    normalized = (logit - min_logit) / (max_logit - min_logit)
    normalized = np.clip(normalized, 0, 1)

    severity_score = normalized * 100
    severity_level, description = interpret_wave_severity(severity_score)

    return {
        "type": "Wave",
        "label": "Parkinson" if prob > 0.5 else "Healthy",
        "logit": float(logit),
        "probability": float(prob),
        "score": float(severity_score),
        "severity": severity_level,
        "description": description
    }