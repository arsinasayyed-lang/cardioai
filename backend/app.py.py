from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def predict_heart_disease(data):
    age = float(data['age'])
    sex = float(data['sex'])
    cp = float(data['cp'])
    trestbps = float(data['trestbps'])
    chol = float(data['chol'])
    thalach = float(data['thalach'])
    exang = float(data['exang'])
    oldpeak = float(data['oldpeak'])
    score = 0
    if age > 65:
        score += 25
    elif age > 55:
        score += 18
    elif age > 45:
        score += 10
    else:
        score += 3
    if sex == 1:
        score += 8
    if cp == 1:
        score += 20
    elif cp == 2:
        score += 13
    elif cp == 3:
        score += 7
    if trestbps > 160:
        score += 15
    elif trestbps > 140:
        score += 10
    elif trestbps > 120:
        score += 5
    if chol > 280:
        score += 14
    elif chol > 240:
        score += 9
    elif chol > 200:
        score += 4
    if thalach < 100:
        score += 18
    elif thalach < 130:
        score += 10
    elif thalach < 150:
        score += 5
    if exang == 1:
        score += 16
    if oldpeak > 3:
        score += 14
    elif oldpeak > 2:
        score += 9
    elif oldpeak > 1:
        score += 5
    risk_pct = min(round((score / 130) * 100), 96)
    if risk_pct >= 70:
        if cp == 1:
            disease = "Coronary Artery Disease (CAD)"
        elif oldpeak > 3:
            disease = "Ischemic Heart Disease"
        else:
            disease = "Coronary Artery Disease (CAD)"
    elif risk_pct >= 40:
        if cp == 2:
            disease = "Angina Pectoris (Atypical)"
        else:
            disease = "Mild Cardiac Risk - Further Testing Advised"
    else:
        disease = "No Significant Heart Disease Detected"
    if risk_pct >= 65:
        risk_level = "High"
    elif risk_pct >= 35:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    return {
        "risk_percentage": risk_pct,
        "disease_type": disease,
        "risk_level": risk_level
    }

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "CardioAI v1.0"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        required = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing: {missing}"}), 400
        result = predict_heart_disease(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("CardioAI Server running at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
```
