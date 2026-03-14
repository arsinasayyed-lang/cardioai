"""
CardioAI ML Model – ml_model.py
Heart Disease Classifier using Random Forest
Trained on UCI Heart Disease Dataset features
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# ─── SYNTHETIC TRAINING DATA ────────────────────────────────────────────────
# Based on UCI Heart Disease Dataset statistics
# Features: age, sex, cp, trestbps, chol, thalach, exang, oldpeak
# Label: 0=No disease, 1=CAD, 2=Angina, 3=Ischemic, 4=Cardiomyopathy
TRAIN_X = np.array([
    # age, sex, cp, trestbps, chol, thalach, exang, oldpeak → label
    # --- Healthy ---
    [29, 0, 0, 110, 165, 180, 0, 0.0],
    [31, 1, 0, 120, 180, 175, 0, 0.0],
    [35, 0, 0, 115, 190, 170, 0, 0.0],
    [38, 1, 0, 125, 200, 165, 0, 0.1],
    [40, 0, 0, 118, 195, 160, 0, 0.0],
    [42, 1, 0, 130, 210, 162, 0, 0.2],
    [44, 0, 0, 120, 205, 165, 0, 0.1],
    [46, 1, 0, 128, 215, 158, 0, 0.2],
    [33, 0, 0, 112, 172, 178, 0, 0.0],
    [37, 1, 0, 122, 185, 172, 0, 0.1],
    # --- Typical Angina (CAD) ---
    [58, 1, 1, 150, 270, 120, 1, 3.5],
    [62, 1, 1, 160, 290, 110, 1, 4.0],
    [55, 1, 1, 145, 260, 130, 1, 2.8],
    [60, 0, 1, 155, 280, 115, 1, 3.2],
    [65, 1, 1, 165, 305, 100, 1, 4.5],
    [57, 1, 1, 148, 265, 125, 1, 3.0],
    [63, 0, 1, 158, 275, 112, 1, 3.7],
    [59, 1, 1, 152, 272, 118, 1, 3.3],
    [61, 1, 2, 155, 280, 120, 1, 2.5],
    [56, 0, 1, 142, 255, 128, 1, 2.9],
    # --- Atypical Angina ---
    [48, 1, 2, 138, 240, 140, 0, 1.8],
    [50, 0, 2, 132, 230, 148, 0, 1.5],
    [52, 1, 2, 140, 245, 138, 1, 2.0],
    [54, 0, 2, 135, 235, 145, 0, 1.6],
    [49, 1, 2, 136, 242, 142, 0, 1.7],
    [51, 0, 2, 130, 228, 150, 0, 1.4],
    [53, 1, 2, 142, 248, 136, 1, 2.1],
    [47, 0, 2, 128, 225, 155, 0, 1.3],
    # --- Ischemic Heart Disease ---
    [67, 1, 3, 170, 320, 95,  1, 5.0],
    [70, 1, 3, 175, 335, 90,  1, 5.5],
    [64, 1, 3, 168, 315, 98,  1, 4.8],
    [69, 0, 3, 172, 328, 92,  1, 5.2],
    [72, 1, 3, 178, 340, 88,  1, 5.8],
    [66, 1, 3, 166, 310, 100, 1, 4.6],
    [68, 0, 3, 171, 322, 93,  1, 5.1],
    # --- Non-anginal Pain (mixed) ---
    [43, 0, 3, 128, 218, 158, 0, 0.8],
    [45, 1, 3, 133, 225, 152, 0, 1.0],
    [41, 0, 3, 125, 212, 162, 0, 0.6],
    [39, 1, 3, 130, 220, 160, 0, 0.9],
    [44, 0, 3, 127, 215, 165, 0, 0.7],
])

TRAIN_Y = np.array([
    # Healthy = 0
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # CAD = 1
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    # Angina = 2
    2, 2, 2, 2, 2, 2, 2, 2,
    # Ischemic = 3
    3, 3, 3, 3, 3, 3, 3,
    # Non-specific = 4 (low risk)
    4, 4, 4, 4, 4,
])

DISEASE_MAP = {
    0: "No significant heart disease detected",
    1: "Coronary Artery Disease (CAD)",
    2: "Angina Pectoris (Atypical)",
    3: "Ischemic Heart Disease",
    4: "Non-specific cardiac discomfort (low risk)",
}

# Risk base percentages per disease class
RISK_BASE = {0: 12, 1: 78, 2: 55, 3: 88, 4: 25}


class HeartDiseaseModel:
    def __init__(self):
        self.clf = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=2,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        self._train()

    def _train(self):
        self.clf.fit(TRAIN_X, TRAIN_Y)
        print("✅ HeartDiseaseModel trained successfully.")

    def _compute_risk_pct(self, features, pred_class, proba):
        """Compute nuanced risk percentage from features + model probability."""
        age, sex, cp, trestbps, chol, thalach, exang, oldpeak = features
        base = RISK_BASE[pred_class]

        # Adjustments
        delta = 0
        if age > 60:        delta += 8
        elif age > 50:      delta += 4
        if trestbps > 150:  delta += 6
        elif trestbps > 135:delta += 3
        if chol > 260:      delta += 6
        elif chol > 220:    delta += 3
        if thalach < 110:   delta += 7
        elif thalach < 140: delta += 3
        if exang == 1:      delta += 8
        if oldpeak > 3:     delta += 8
        elif oldpeak > 1.5: delta += 4
        if sex == 1 and age > 50: delta += 4

        risk = base + delta
        return float(np.clip(risk, 5, 97))

    def predict(self, features: list) -> dict:
        X = np.array(features).reshape(1, -1)
        pred_class  = int(self.clf.predict(X)[0])
        proba       = self.clf.predict_proba(X)[0]
        risk_pct    = self._compute_risk_pct(features, pred_class, proba)
        disease     = DISEASE_MAP[pred_class]

        if risk_pct > 65:
            risk_level = "High"
        elif risk_pct > 35:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        return {
            "risk_percentage": round(risk_pct, 1),
            "disease_type":    disease,
            "risk_level":      risk_level,
            "class_id":        pred_class,
            "confidence":      round(float(max(proba)) * 100, 1)
        }
