import joblib
import numpy as np
from he_utils import create_context, encrypt_vector, decrypt_vector

def run_encrypted_inference(raw_patient_data, model_path, scaler_path):
    """
    Generic encrypted inference — works for ANY trained logistic regression model,
    not just diabetes. Pass in the model/scaler file paths for whichever model
    you're running.

    raw_patient_data: list of raw feature values, in the same order the model was trained on
    model_path: path to the .pkl model file (e.g. "models/diabetes_model.pkl")
    scaler_path: path to the .pkl scaler file (e.g. "models/diabetes_scaler.pkl")
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    scaled_data = scaler.transform([raw_patient_data])[0]

    context = create_context()

    encrypted_input = encrypt_vector(context, scaled_data.tolist())

    weights = model.coef_[0].tolist()
    bias = model.intercept_[0]

    encrypted_result = encrypted_input.dot(weights) + bias

    decrypted_score = decrypt_vector(encrypted_result)

    linear_score = decrypted_score[0] if isinstance(decrypted_score, list) else decrypted_score
    probability = 1 / (1 + np.exp(-linear_score))

    return probability

if __name__ == "__main__":
    sample_patient = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    prob = run_encrypted_inference(sample_patient, "models/diabetes_model.pkl", "models/diabetes_scaler.pkl")
    print(f"Diabetes risk probability (via encrypted computation): {prob:.4f}")