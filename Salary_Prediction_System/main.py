import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

CSV_FILE = "Salary Data.csv"

model = None
r2 = None

def train_model():
    global model, r2
    try:
        df = pd.read_csv(CSV_FILE).dropna()
        X = df.drop(columns=["Salary"])
        y = df["Salary"]

        cat = X.select_dtypes(include="object").columns.tolist()
        #print(cat)
        num = [c for c in X.columns if c not in cat]
        #print(num)

        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num)
        ])

        model = Pipeline([
            ("prep", pre),
            ("reg", LinearRegression())
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)

        status.config(text=f"Model trained (R² = {r2:.3f})", fg="green")
        messagebox.showinfo("Success", "Model trained successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def predict():
    if model is None:
        messagebox.showwarning("Warning", "Train the model first.")
        return
    try:
        row = pd.DataFrame([{
            "Age": float(age.get()),
            "Gender": gender.get(),
            "Education Level": education.get(),
            "Job Title": job.get(),
            "Years of Experience": float(exp.get())
        }])
        sal = model.predict(row)[0]
        result.config(text=f"Predicted Salary: {sal:,.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

df = pd.read_csv(CSV_FILE).dropna()

root = tk.Tk()
root.title("Salary Prediction System")
root.geometry("500x500")

status = tk.Label(root, text="Model not trained", fg="red")
status.pack(pady=5)

tk.Button(root, text="Train Model", command=train_model,bg="yellow", fg="black").pack()

def add_field(lbl):
    tk.Label(root, text=lbl).pack()
    return tk.Entry(root)

age = add_field("Age")
age.pack()

tk.Label(root, text="Gender").pack()
gender = ttk.Combobox(root, values=sorted(df["Gender"].dropna().unique().tolist()))
gender.pack()

tk.Label(root, text="Education Level").pack()
education = ttk.Combobox(root, values=sorted(df["Education Level"].dropna().unique().tolist()))
education.pack()

tk.Label(root, text="Job Title").pack()
job = ttk.Combobox(root, values=sorted(df["Job Title"].dropna().unique().tolist()))
job.pack()

exp = add_field("Years of Experience")
exp.pack()

tk.Button(root, text="Predict Salary", command=predict,bg="green", fg="white").pack(pady=10)

result = tk.Label(root, text="", font=("Arial",16,"bold"))
result.pack()


# Clear all input fields
def clear_fields():
    age.delete(0, tk.END)
    gender.set("")
    education.set("")
    job.set("")
    exp.delete(0, tk.END)
    result.config(text="")


# Exit the application
def exit_app():
    if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
        root.destroy()

# Frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

tk.Button(button_frame, text="Clear", command=clear_fields,
          width=12, bg="orange", fg="white").grid(row=0, column=0, padx=5)

tk.Button(button_frame, text="Exit", command=exit_app,
          width=12, bg="red", fg="white").grid(row=0, column=1, padx=5)

root.mainloop()