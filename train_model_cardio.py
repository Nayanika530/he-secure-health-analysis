import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# This dataset already has column headers, unlike the diabetes one
df = pd.read_csv("data/cardio.csv")

# Drop rows with missing values — this dataset has some NA entries
df = df.dropna()

X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Cardiovascular model accuracy: {accuracy:.4f}")

joblib.dump(model, "models/cardio_model.pkl")
joblib.dump(scaler, "models/cardio_scaler.pkl")

print("Saved cardio_model.pkl and cardio_scaler.pkl")
print("Feature order used:", list(X.columns))