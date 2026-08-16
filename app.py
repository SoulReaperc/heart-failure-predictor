import os
import sys
import site
sys.path.append(site.getusersitepackages())

import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# -------------------------------------------------------------
# PyTorch Deep Learning Architectures
# -------------------------------------------------------------
class CardiovascularANN(nn.Module):
    def __init__(self, input_dim=15):
        super(CardiovascularANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.network(x)

class CardiovascularLSTM(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=32, num_layers=2):
        super(CardiovascularLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        x_seq = x.unsqueeze(2)
        _, (h_n, _) = self.lstm(x_seq)
        return self.fc(h_n[-1])

class CardiovascularGRU(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=32, num_layers=2):
        super(CardiovascularGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=1, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        x_seq = x.unsqueeze(2)
        _, h_n = self.gru(x_seq)
        return self.fc(h_n[-1])

# Load Preprocessing Artifacts
imputer_path = "imputer.pkl"
scaler_path = "scaler.pkl"
rf_model_path = "heart_model.pkl"

imputer = joblib.load(imputer_path) if os.path.exists(imputer_path) else None
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
rf_model = joblib.load(rf_model_path) if os.path.exists(rf_model_path) else None

# Load Deep Learning Models
ann_model = None
if os.path.exists("ann_model.pth"):
    ann_model = CardiovascularANN(input_dim=15)
    ann_model.load_state_dict(torch.load("ann_model.pth", map_location=torch.device('cpu')))
    ann_model.eval()

lstm_model = None
if os.path.exists("lstm_model.pth"):
    lstm_model = CardiovascularLSTM(input_dim=15)
    lstm_model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device('cpu')))
    lstm_model.eval()

gru_model = None
if os.path.exists("gru_model.pth"):
    gru_model = CardiovascularGRU(input_dim=15)
    gru_model.load_state_dict(torch.load("gru_model.pth", map_location=torch.device('cpu')))
    gru_model.eval()

# Load Comparative Metrics
metrics_path = "static/model_comparison_metrics.json"
metrics_data = {}
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

# 15 Framingham Feature Names
FEATURE_NAMES = [
    "male", "age", "education", "currentSmoker", "cigsPerDay", 
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", 
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

def analyze_risk_factors(data):
    """Identify patient-specific clinical risk triggers based on AHA guidelines."""
    warnings = []
    
    # Blood pressure
    sys_bp = data.get("sysBP", 0)
    dia_bp = data.get("diaBP", 0)
    if sys_bp >= 140 or dia_bp >= 90:
        warnings.append({
            "factor": f"Hypertension Stage 2 ({int(sys_bp)}/{int(dia_bp)} mmHg)",
            "severity": "high",
            "desc": "Blood pressure exceeds normal range (120/80 mmHg), placing excessive strain on arterial walls."
        })
    elif sys_bp >= 130 or dia_bp >= 80:
        warnings.append({
            "factor": f"Hypertension Stage 1 ({int(sys_bp)}/{int(dia_bp)} mmHg)",
            "severity": "moderate",
            "desc": "Elevated blood pressure indicates mild vascular resistance."
        })

    # Cholesterol
    chol = data.get("totChol", 0)
    if chol >= 240:
        warnings.append({
            "factor": f"High Total Cholesterol ({int(chol)} mg/dL)",
            "severity": "high",
            "desc": "Serum cholesterol is elevated (>240 mg/dL), increasing atherosclerotic plaque risk."
        })
    elif chol >= 200:
        warnings.append({
            "factor": f"Borderline High Cholesterol ({int(chol)} mg/dL)",
            "severity": "moderate",
            "desc": "Cholesterol is in the borderline zone (200-239 mg/dL)."
        })

    # Glucose & Diabetes
    glucose = data.get("glucose", 0)
    diabetes = data.get("diabetes", 0)
    if diabetes == 1 or glucose >= 126:
        warnings.append({
            "factor": f"Diabetic / Elevated Fasting Glucose ({int(glucose)} mg/dL)",
            "severity": "high",
            "desc": "High blood glucose damages micro- and macro-vasculature, compounding cardiac risk."
        })
    elif glucose >= 100:
        warnings.append({
            "factor": f"Impaired Fasting Glucose ({int(glucose)} mg/dL)",
            "severity": "moderate",
            "desc": "Pre-diabetic glucose range (100-125 mg/dL)."
        })

    # Smoking
    cigs = data.get("cigsPerDay", 0)
    smoker = data.get("currentSmoker", 0)
    if smoker == 1 and cigs >= 20:
        warnings.append({
            "factor": f"Heavy Smoker ({int(cigs)} cigs/day)",
            "severity": "high",
            "desc": "Heavy nicotine and carbon monoxide exposure significantly elevates coronary thrombosis risk."
        })
    elif smoker == 1 and cigs > 0:
        warnings.append({
            "factor": f"Active Smoker ({int(cigs)} cigs/day)",
            "severity": "moderate",
            "desc": "Smoking compromises arterial endothelial lining and elevates heart rate."
        })

    # BMI
    bmi = data.get("BMI", 0)
    if bmi >= 30:
        warnings.append({
            "factor": f"Obesity (BMI {bmi:.1f})",
            "severity": "moderate",
            "desc": "BMI >= 30 increases cardiac workload and metabolic syndrome risk."
        })

    # History of Stroke
    if data.get("prevalentStroke", 0) == 1:
        warnings.append({
            "factor": "Prior Cerebrovascular Event (Stroke)",
            "severity": "high",
            "desc": "Patient has a confirmed history of stroke, indicating systemic arterial disease."
        })

    return warnings

@app.route("/")
def home():
    return render_template("index.html", prediction=None, selected_model="ann", metrics=metrics_data)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_choice = request.form.get("model_choice", "ann")
        if imputer is None or scaler is None:
            return render_template("index.html", 
                                   prediction="❌ Preprocessing files (imputer/scaler) not found. Run model training first.", 
                                   color_class="error", 
                                   selected_model=model_choice,
                                   metrics=metrics_data)

        # Collect 15 features
        form_data = {}
        input_values = []
        
        for name in FEATURE_NAMES:
            raw_val = request.form.get(name)
            if raw_val is None or raw_val.strip() == "":
                val = np.nan
            else:
                val = float(raw_val)
            input_values.append(val)
            form_data[name] = val if not np.isnan(val) else ""

        # Preprocessing pipeline
        input_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # Inference based on selected architecture
        tensor_input = torch.tensor(input_scaled, dtype=torch.float32)

        if model_choice == "ann" and ann_model is not None:
            with torch.no_grad():
                logits = ann_model(tensor_input)
                prob = torch.sigmoid(logits).item()
                probability = round(prob * 100, 1)
                used_model_name = "PyTorch Artificial Neural Network (ANN)"
        elif model_choice == "lstm" and lstm_model is not None:
            with torch.no_grad():
                logits = lstm_model(tensor_input)
                prob = torch.sigmoid(logits).item()
                probability = round(prob * 100, 1)
                used_model_name = "PyTorch Long Short-Term Memory (LSTM)"
        elif model_choice == "gru" and gru_model is not None:
            with torch.no_grad():
                logits = gru_model(tensor_input)
                prob = torch.sigmoid(logits).item()
                probability = round(prob * 100, 1)
                used_model_name = "PyTorch Gated Recurrent Unit (GRU)"
        elif rf_model is not None:
            probs = rf_model.predict_proba(input_scaled)[0]
            probability = round(probs[1] * 100, 1)
            used_model_name = "Random Forest Ensemble (ML Baseline)"
        else:
            return render_template("index.html", 
                                   prediction="❌ Selected model file not available.", 
                                   color_class="error", 
                                   selected_model=model_choice,
                                   metrics=metrics_data)

        # Stratify risk levels
        if probability >= 50:
            result_title = "HIGH CARDIOVASCULAR RISK"
            color_class = "high-risk"
            risk_summary = "High 10-Year Probability of Coronary Heart Disease. Clinical consultation, lipid management, and blood pressure control strongly advised."
        elif probability >= 25:
            result_title = "MODERATE CARDIOVASCULAR RISK"
            color_class = "moderate-risk"
            risk_summary = "Elevated risk indicators detected. Lifestyle optimization and routine clinical monitoring recommended."
        else:
            result_title = "LOW CARDIOVASCULAR RISK"
            color_class = "low-risk"
            risk_summary = "Cardiovascular markers within healthy baseline limits."

        risk_factors = analyze_risk_factors(form_data)

        return render_template(
            "index.html",
            prediction=result_title,
            probability=probability,
            color_class=color_class,
            risk_summary=risk_summary,
            risk_factors=risk_factors,
            used_model=used_model_name,
            selected_model=model_choice,
            form_data=form_data,
            metrics=metrics_data
        )

    except Exception as e:
        return render_template("index.html", 
                               prediction=f"⚠️ Prediction Error: {str(e)}", 
                               color_class="error", 
                               selected_model=request.form.get("model_choice", "ann"),
                               metrics=metrics_data)

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics_data)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
