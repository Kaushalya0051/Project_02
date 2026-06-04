"""
Vehicle ML Prediction API v2.0
================================
Run locally:
    pip install flask flask-cors joblib scikit-learn numpy
    python api.py

Deploy FREE on Render.com:
    1. Create requirements.txt (see bottom of this file)
    2. Push api.py, vehicle_model.pkl, scaler.pkl, requirements.txt to GitHub
    3. Render.com -> New Web Service -> connect GitHub repo
    4. Build command : pip install -r requirements.txt
    5. Start command : python api.py
    6. Copy the URL -> paste in Flutter ml_service.dart _baseUrl

Test with curl:
    curl http://localhost:5000/health

    curl -X POST http://localhost:5000/predict
         -H "Content-Type: application/json"
         -d '{"rpm":2000,"coolant":90,"load":50,"throttle":30,
              "fuel":60,"speed":80,"battery":12.5,
              "oil":40,"vibration":0.2,"exhaust":350}'

requirements.txt content:
    flask
    flask-cors
    scikit-learn==1.3.2
    joblib
    numpy
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model - fail immediately with clear error if files missing
try:
    model  = joblib.load("vehicle_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("[API] Model and scaler loaded OK")
except FileNotFoundError as e:
    raise SystemExit(
        f"ERROR: {e}\n"
        "Run train_model.py first to generate the .pkl files."
    )

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
FAULT_NAMES = {
    0: "Safe to Ride",
    1: "Engine Fault Risk",
    2: "Brake Issue Risk",
    3: "Battery Issue Risk",
}

FAULT_ADVICE = {
    0: "All systems normal. Good to go!",
    1: "Engine warning: check temperature, oil pressure, and RPM. Visit mechanic.",
    2: "Brake warning: check brake fluid and brake pads. Reduce speed.",
    3: "Battery warning: check battery voltage and alternator. Avoid long trips.",
}

# Feature order must exactly match train_model.py FEATURES list
FEATURE_ORDER = [
    "engine_rpm",
    "coolant_temp_c",
    "engine_load_percent",
    "throttle_pos_percent",
    "fuel_level_percent",
    "vehicle_speed_kph",
    "battery_voltage_v",
    "oil_pressure_psi",
    "vibration_level",
    "exhaust_gas_temp_c",
]

# Flutter sends short keys - map to full feature names
KEY_MAP = {
    "rpm":       "engine_rpm",
    "coolant":   "coolant_temp_c",
    "load":      "engine_load_percent",
    "throttle":  "throttle_pos_percent",
    "fuel":      "fuel_level_percent",
    "speed":     "vehicle_speed_kph",
    "battery":   "battery_voltage_v",
    "oil":       "oil_pressure_psi",
    "vibration": "vibration_level",
    "exhaust":   "exhaust_gas_temp_c",
}

# Defaults for PIDs not available on all vehicles
DEFAULTS = {
    "engine_rpm":           1000.0,
    "coolant_temp_c":         90.0,
    "engine_load_percent":    40.0,
    "throttle_pos_percent":   20.0,
    "fuel_level_percent":     50.0,
    "vehicle_speed_kph":       0.0,
    "battery_voltage_v":      12.6,
    "oil_pressure_psi":       40.0,
    "vibration_level":         0.1,
    "exhaust_gas_temp_c":    300.0,
}

# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "model":   "RandomForest v2",
        "classes": list(FAULT_NAMES.values()),
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    body = request.json or {}

    # Map short Flutter keys to full feature names
    mapped = {}
    for short_key, feat_name in KEY_MAP.items():
        raw_val = body.get(short_key)
        if raw_val is None:
            raw_val = body.get(feat_name)
        if raw_val is not None:
            mapped[feat_name] = float(raw_val)
        else:
            mapped[feat_name] = DEFAULTS[feat_name]

    # Build feature vector in correct order
    features = np.array([[mapped[f] for f in FEATURE_ORDER]])

    try:
        scaled   = scaler.transform(features)
        pred_int = int(model.predict(scaled)[0])
        proba    = model.predict_proba(scaled)[0].tolist()
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    confidence = round(proba[pred_int] * 100, 1)

    all_probs = {
        FAULT_NAMES[i]: round(p * 100, 1)
        for i, p in enumerate(proba)
    }

    return jsonify({
        "fault_code":    pred_int,
        "fault_name":    FAULT_NAMES[pred_int],
        "advice":        FAULT_ADVICE[pred_int],
        "confidence":    confidence,
        "safe":          pred_int == 0,
        "probabilities": all_probs,
        "input_used":    mapped,
    })

# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[API] Running on port {port}")
    print(f"[API] Health check: GET  http://localhost:{port}/health")
    print(f"[API] Predict:      POST http://localhost:{port}/predict")
    app.run(host="0.0.0.0", port=port, debug=False)
