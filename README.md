# Privital

A privacy-preserving health risk analysis platform that performs machine learning inference on encrypted data using homomorphic encryption. Patient input values remain encrypted throughout computation; the server never has access to plaintext health data during inference.

**Live demo:** [https://privital.onrender.com](https://privital.onrender.com)

## Overview

Standard machine learning inference requires plaintext access to input data. This creates a structural privacy problem in healthcare and other sensitive domains: any party running inference — an AI vendor, a cloud provider, an internal analytics team — necessarily sees the raw data it is analyzing.

Privital demonstrates an alternative. Using the CKKS homomorphic encryption scheme (via Microsoft SEAL, through the TenSEAL Python bindings), input data is encrypted client-side before any computation occurs. The server performs the model's arithmetic — a weighted sum of encrypted feature values — directly on ciphertext. The result is decrypted only after computation completes. At no point does the computing party have access to the underlying values.

This is implemented and demonstrated across three independent risk-assessment models, to show the approach generalizes rather than being specific to one dataset or use case.

## What it does

- **Diabetes risk assessment** — logistic regression trained on the Pima Indians Diabetes dataset
- **Cardiovascular risk assessment** — logistic regression trained on the Framingham Heart Study dataset
- **PHQ-9 depression screening** — the standard 9-item clinical instrument, implemented as an encrypted weighted sum rather than a trained model, since PHQ-9 scoring is a fixed, validated formula rather than something learned from data

Each analysis type returns a result computed via homomorphic encryption alongside the equivalent plaintext computation, for direct comparison.

Registered users can save analysis results to a personal history and view a dashboard showing the volume of encrypted computations performed, with a running count confirming zero raw health values are stored on the server.

## How the encryption works

1. Input values are encrypted client-side into a CKKS ciphertext vector.
2. The trained model's weights and bias are applied to the encrypted vector via homomorphic dot product and addition. This computation occurs entirely on ciphertext.
3. The resulting encrypted linear score is decrypted.
4. A sigmoid function is applied to the decrypted score to produce a final probability.

**A note on the last step, since it matters for evaluating this project honestly:** CKKS supports addition and multiplication on ciphertext, but not arbitrary nonlinear functions such as sigmoid. Implementing sigmoid homomorphically requires polynomial approximation, which introduces additional noise and implementation complexity beyond the scope of this project. The design decision made here is to compute the linear component (weights · features + bias) homomorphically, decrypt only that single scalar value, and apply sigmoid in plaintext afterward. This means the sensitive input data — the actual patient measurements — is never decrypted or exposed at any point in the pipeline. Only a single derived score is decrypted, briefly, before the final probability is produced. This is a documented design tradeoff, not an oversight, and is consistent with how linear/logistic models are commonly handled in applied homomorphic encryption work.

## Architecture

```
User input
    |
Client-side encryption (CKKS)
    |
Encrypted feature vector
    |
Homomorphic dot product with model weights
    |
Encrypted linear score
    |
Decryption (single scalar)
    |
Sigmoid (plaintext)
    |
Risk probability
```

## Tech stack

| Layer | Technology |
|---|---|
| Encryption | TenSEAL (Microsoft SEAL bindings), CKKS scheme |
| ML models | scikit-learn (logistic regression) |
| Backend | Flask |
| Authentication | Flask-Login, Werkzeug password hashing |
| Data storage | SQLAlchemy, SQLite |
| Frontend | HTML, vanilla JavaScript, Chart.js |
| Deployment | Render |

## Datasets

- Diabetes: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Cardiovascular: Framingham Heart Study dataset

Both are established, publicly available clinical research datasets, used here for demonstration purposes only.

## Limitations

This is a portfolio and research demonstration, not a production healthcare system. Specifically:

- The application does not implement role-based access control, consent management, or granular data-sharing permissions.
- Audit logging is limited to per-user analysis history; it is not tamper-evident or cryptographically chained.
- The deployment uses SQLite on Render's free tier, which has ephemeral storage — stored accounts and history may be reset on service restart.
- The models are trained on limited public research datasets and are not validated for clinical use. Outputs are risk indicators for demonstration purposes, not medical advice or diagnosis.
- Key management is simplified for demonstration; a production system would require a formal client-side key generation and storage architecture.

## Local setup

```bash
git clone https://github.com/Nayanika530/he-secure-health-analysis.git
cd he-secure-health-analysis
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python app.py
```

Application runs at `http://127.0.0.1:5000`.

## Project background

Built as a portfolio project for a B.Tech Cybersecurity specialization, focused on demonstrating applied privacy-preserving computation. Part of a broader project sequence exploring cryptography and security engineering.
