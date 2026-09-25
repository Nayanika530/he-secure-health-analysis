from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import db, User, HealthRecord
from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import numpy as np
from encrypted_inference import run_encrypted_inference
from phq9_screener import run_encrypted_phq9, PHQ9_QUESTIONS

app = Flask(__name__)

app.config['SECRET_KEY'] = 'change-this-later-to-something-random'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///privital.db'
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

DIABETES_FEATURES = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                      "insulin", "bmi", "diabetes_pedigree", "age"]

CARDIO_FEATURES = ["male", "age", "education", "currentSmoker", "cigsPerDay",
                    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
                    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]

diabetes_model = joblib.load("models/diabetes_model.pkl")
diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")
cardio_model = joblib.load("models/cardio_model.pkl")
cardio_scaler = joblib.load("models/cardio_scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html",
                            diabetes_features=DIABETES_FEATURES,
                            cardio_features=CARDIO_FEATURES,
                            phq9_questions=PHQ9_QUESTIONS)

@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    data = request.get_json()
    raw_values = [float(data[f]) for f in DIABETES_FEATURES]
    scaled = diabetes_scaler.transform([raw_values])[0]
    plaintext_prob = diabetes_model.predict_proba([scaled])[0][1]
    he_prob = run_encrypted_inference(raw_values, "models/diabetes_model.pkl", "models/diabetes_scaler.pkl")
    if current_user.is_authenticated:
        record = HealthRecord(user_id=current_user.id, record_type="diabetes",
                               result_summary=f"HE: {round(float(he_prob),4)}")
        db.session.add(record)
        db.session.commit()
    return jsonify({
        "type": "probability",
        "traditional_probability": round(float(plaintext_prob), 4),
        "he_probability": round(float(he_prob), 4)
    })

@app.route("/predict/cardio", methods=["POST"])
def predict_cardio():
    data = request.get_json()
    raw_values = [float(data[f]) for f in CARDIO_FEATURES]
    scaled = cardio_scaler.transform([raw_values])[0]
    plaintext_prob = cardio_model.predict_proba([scaled])[0][1]
    he_prob = run_encrypted_inference(raw_values, "models/cardio_model.pkl", "models/cardio_scaler.pkl")
    return jsonify({
        "type": "probability",
        "traditional_probability": round(float(plaintext_prob), 4),
        "he_probability": round(float(he_prob), 4)
    })

@app.route("/predict/phq9", methods=["POST"])
def predict_phq9():
    data = request.get_json()
    answers = [int(data[f"q{i}"]) for i in range(9)]
    score, band = run_encrypted_phq9(answers)
    if current_user.is_authenticated:
        record = HealthRecord(user_id=current_user.id, record_type="phq9",
                               result_summary=f"{score}/27 - {band}")
        db.session.add(record)
        db.session.commit()
    return jsonify({
        "type": "phq9",
        "score": score,
        "band": band
    })

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        if User.query.filter_by(email=email).first():
            return "Email already registered", 400
        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for("home"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("home"))
        return "Invalid credentials", 401
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/history")
@login_required
def history():
    records = HealthRecord.query.filter_by(user_id=current_user.id).order_by(HealthRecord.timestamp.desc()).all()
    return render_template("history.html", records=records)

@app.route("/privacy-dashboard")
@login_required
def privacy_dashboard():
    total_records = HealthRecord.query.filter_by(user_id=current_user.id).count()
    diabetes_count = HealthRecord.query.filter_by(user_id=current_user.id, record_type="diabetes").count()
    cardio_count = HealthRecord.query.filter_by(user_id=current_user.id, record_type="cardio").count()
    phq9_count = HealthRecord.query.filter_by(user_id=current_user.id, record_type="phq9").count()

    return render_template("privacy_dashboard.html",
                            total_records=total_records,
                            diabetes_count=diabetes_count,
                            cardio_count=cardio_count,
                            phq9_count=phq9_count)

if __name__ == "__main__":
    app.run(debug=True)