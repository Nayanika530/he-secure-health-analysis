import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

columns = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
           "insulin", "bmi", "diabetes_pedigree", "age", "outcome"]

df = pd.read_csv("data/diabetes.csv", names=columns)

X = df.drop("outcome", axis=1)
y = df["outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Plaintext model accuracy: {accuracy:.4f}")

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Saved model.pkl and scaler.pkl")
print("Weights:", model.coef_)
print("Bias:", model.intercept_)