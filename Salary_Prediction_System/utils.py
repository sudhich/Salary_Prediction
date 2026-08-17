
import joblib
import pandas as pd
from tkinter import messagebox

MODEL_FILE="model.pkl"

def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        messagebox.showerror("Model Missing","Please run train_model.py first.")
        return None

def predict_salary(model, age, gender, education, job_title, experience):
    row=pd.DataFrame([{
        "Age": float(age),
        "Gender": gender,
        "Education Level": education,
        "Job Title": job_title,
        "Years of Experience": float(experience)
    }])
    pred=model.predict(row)[0]
    return round(float(pred),2)

def validate_inputs(age, experience):
    try:
        age=float(age)
        experience=float(experience)
    except ValueError:
        return False,"Age and Experience must be numeric."
    if age<=0 or experience<0:
        return False,"Enter valid positive values."
    return True,""

def format_currency(value):
    return f"₹ {value:,.2f}"