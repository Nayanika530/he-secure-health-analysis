import joblib
import numpy as np
from he_utils import create_context, encrypt_vector, decrypt_vector

def run_encrypted_inference(raw_patient_data):
    """
    raw_patient_data: list of 8 raw feature values, same order as training:
    [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    """
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

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
    prob = run_encrypted_inference(sample_patient)
    print(f"Diabetes risk probability (via encrypted computation): {prob:.4f}")