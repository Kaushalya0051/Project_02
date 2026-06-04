"""
Vehicle Fault Prediction - ML Training Script v2.0
===================================================
Dataset : synthetic_telemetry_data.csv (place in same folder)

Install dependencies:
    pip install pandas scikit-learn numpy joblib flask flask-cors

Run:
    python train_model.py

Output files:
    vehicle_model.pkl
    scaler.pkl
    scaler_params.json
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import joblib
import json
import sys

print("=" * 55)
print("  Vehicle Fault Prediction - ML Trainer")
print("=" * 55)

# -------------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------------
try:
    df = pd.read_csv("synthetic_telemetry_data.csv")
except FileNotFoundError:
    sys.exit("ERROR: synthetic_telemetry_data.csv not found.")

print(f"\nDataset loaded: {len(df)} rows, {len(df.columns)} columns")

# -------------------------------------------------------
# 2. BUILD FAULT LABEL
# Labels:
#   0 = Safe to Ride
#   1 = Engine Fault Risk
#   2 = Brake Issue Risk
#   3 = Battery Issue Risk
# -------------------------------------------------------
def make_label(row):
    if row["engine_failure_imminent"] == 1:
        return 1
    elif row["brake_issue_imminent"] == 1:
        return 2
    elif row["battery_issue_imminent"] == 1:
        return 3
    return 0

df["fault_label"] = df.apply(make_label, axis=1)

LABEL_NAMES = {
    0: "Safe",
    1: "Engine Risk",
    2: "Brake Risk",
    3: "Battery Risk"
}

print("\nRaw label counts:")
for k, v in df["fault_label"].value_counts().sort_index().items():
    print(f"  {LABEL_NAMES[k]:15s} : {v}")

# -------------------------------------------------------
# 3. SELECT FEATURES
# Only features your ESP32 OBD2 PIDs can provide
# -------------------------------------------------------
FEATURES = [
    "engine_rpm",             # PID 0x0C
    "coolant_temp_c",         # PID 0x05
    "engine_load_percent",    # PID 0x04
    "throttle_pos_percent",   # PID 0x11
    "fuel_level_percent",     # PID 0x2F
    "vehicle_speed_kph",      # PID 0x0D
    "battery_voltage_v",      # PID 0x42
    "oil_pressure_psi",       # PID 0x0B (if supported)
    "vibration_level",        # sensor / derived
    "exhaust_gas_temp_c",     # PID 0x5C (if supported)
]

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    sys.exit(f"ERROR: columns not found in CSV: {missing}")

# -------------------------------------------------------
# 4. BALANCE CLASSES
# Dataset is heavily imbalanced - upsample minority classes
# -------------------------------------------------------
TARGET_N = 350
print(f"\nBalancing classes to {TARGET_N} samples each...")

parts = []
safe_df = df[df["fault_label"] == 0].copy()
safe_sample = safe_df.sample(n=min(600, len(safe_df)), random_state=42)
safe_sample = safe_sample[FEATURES].copy()
safe_sample["__lbl__"] = 0
parts.append(safe_sample)

for lbl in [1, 2, 3]:
    subset = df[df["fault_label"] == lbl][FEATURES].copy()
    subset["__lbl__"] = lbl
    if len(subset) == 0:
        print(f"  WARNING: no samples for {LABEL_NAMES[lbl]} - skipping")
        continue
    upsampled = resample(subset, replace=True, n_samples=TARGET_N, random_state=42)
    parts.append(upsampled)
    print(f"  {LABEL_NAMES[lbl]:15s}: {len(subset)} -> {TARGET_N}")

df_bal = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
X = df_bal[FEATURES]
y = df_bal["__lbl__"].astype(int)
print(f"\nBalanced total: {len(df_bal)} rows")

# -------------------------------------------------------
# 5. SPLIT AND SCALE
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# -------------------------------------------------------
# 6. TRAIN
# -------------------------------------------------------
print("\nTraining RandomForest (200 trees)...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_s, y_train)

# -------------------------------------------------------
# 7. EVALUATE
# -------------------------------------------------------
y_pred = model.predict(X_test_s)

print("\n=== Classification Report ===")
print(classification_report(
    y_test, y_pred,
    target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
    digits=3,
))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="f1_macro")
print(f"5-fold CV F1 (macro): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# -------------------------------------------------------
# 8. FEATURE IMPORTANCE
# -------------------------------------------------------
print("\n=== Feature Importances ===")
pairs = zip(FEATURES, model.feature_importances_)
for feat, imp in sorted(pairs, key=lambda x: -x[1]):
    bar = "#" * int(imp * 50)
    print(f"  {feat:<30} {imp:.3f}  {bar}")

# -------------------------------------------------------
# 9. SAVE MODEL AND SCALER
# -------------------------------------------------------
joblib.dump(model,  "vehicle_model.pkl")
joblib.dump(scaler, "scaler.pkl")

params = {
    "features":     FEATURES,
    "label_names":  LABEL_NAMES,
    "scaler_mean":  scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "n_classes":    4,
}
with open("scaler_params.json", "w") as fh:
    json.dump(params, fh, indent=2)

print("\nSaved: vehicle_model.pkl")
print("Saved: scaler.pkl")
print("Saved: scaler_params.json")
print("\nDone! Next step: run  python api.py")
