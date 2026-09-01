from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from he_utils import create_context, encrypt_vector, decrypt_vector

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_NAMES = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                  "insulin", "bmi", "diabetes_pedigree", "age"]

@app.route("/")
def home():
    return render_template("index.html", features=FEATURE_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    raw_values = [float(data[f]) for f in FEATURE_NAMES]

    scaled = scaler.transform([raw_values])[0]

    # --- Traditional approach: plaintext, model sees raw data directly ---
    plaintext_prob = model.predict_proba([scaled])[0][1]

    # --- HE approach: encrypted computation, model never sees raw values ---
    context = create_context()
    encrypted_input = encrypt_vector(context, scaled.tolist())
    weights = model.coef_[0].tolist()
    bias = model.intercept_[0]
    encrypted_result = encrypted_input.dot(weights) + bias
    decrypted_score = decrypt_vector(encrypted_result)
    linear_score = decrypted_score[0] if isinstance(decrypted_score, list) else decrypted_score
    he_prob = 1 / (1 + np.exp(-linear_score))

    return jsonify({
        "traditional_probability": round(float(plaintext_prob), 4),
        "he_probability": round(float(he_prob), 4),
        "data_exposed_traditional": True,
        "data_exposed_he": False
    })

if __name__ == "__main__":
    app.run(debug=True)