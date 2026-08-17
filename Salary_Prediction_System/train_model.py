# ============================================================
# train_model.py
# Salary Prediction System
# Train Machine Learning Model
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# Load Dataset
# ============================================================

print("Loading dataset...")

df = pd.read_csv("Salary Data.csv")

# ============================================================
# Remove Missing Values
# ============================================================

df = df.dropna()

print("Dataset Shape:", df.shape)

# ============================================================
# Features & Target
# ============================================================

X = df.drop("Salary", axis=1)

y = df["Salary"]

# ============================================================
# Identify Categorical & Numeric Columns
# ============================================================

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

numeric_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumeric Columns:")
print(numeric_columns)

# ============================================================
# Preprocessing
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),

        (
            "num",
            "passthrough",
            numeric_columns
        )

    ]

)

# ============================================================
# Build Pipeline
# ============================================================

model = Pipeline([

    ("preprocessor", preprocessor),

    ("regressor", LinearRegression())

])

# ============================================================
# Split Dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42

)

# ============================================================
# Train Model
# ============================================================

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Training Completed!")

# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# Evaluation
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\n============== MODEL PERFORMANCE ==============")

print(f"MAE : {mae:.2f}")

print(f"MSE : {mse:.2f}")

print(f"R² Score : {r2:.4f}")

print("===============================================\n")

# ============================================================
# Save Model
# ============================================================

joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")

# ============================================================
# Save Feature Names
# ============================================================

joblib.dump(list(X.columns), "features.pkl")

print("Feature list saved as features.pkl")

print("\nProject Training Completed Successfully!")