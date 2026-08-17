Employee Salary Prediction System

Overview

A beginner-friendly Machine Learning project that predicts an employee'ssalary using Linear Regression and a Tkinter GUI.

The project demonstrates the complete ML workflow: - Load a CSVdataset - Data preprocessing - Train a Linear Regression model -Evaluate model performance - Save the trained model - Predict salarythrough a desktop GUI

Project Structure

Salary_Prediction_System/
│
├── Salary Data.csv
├── main.py
├── train_model.py
├── utils.py
├── requirements.txt
├── README.md
├── model.pkl
├── features.pkl
└── assets/

Dataset

File: Salary Data.csv

Columns: - Age - Gender - Education Level - Job Title - Years ofExperience - Salary (Target)

Requirements

Install dependencies:

pip install -r requirements.txt

Train the Model

python train_model.py

This creates: - model.pkl - features.pkl

Run the GUI

python main.py

Application Workflow

Click Train Model

Enter:

Age

Gender

Education Level

Job Title

Years of Experience

Click Predict Salary

View the predicted salary.

Machine Learning

Algorithm: Linear Regression

Train/Test Split: 80/20

Categorical Encoding: OneHotEncoder

Evaluation Metrics:

R² Score

MAE

MSE

Features

Train Model button

Predict Salary button

Input validation

Save trained model

Desktop GUI

Beginner-friendly code

Future Improvements

Random Forest Regression

XGBoost

Model comparison

Prediction history

Export predictions to CSV

Data visualization dashboard

Author

Created as an educational Machine Learning project for Day 20: MLPractical Project.

step=1: pip install -r requirements.txt
step=2: python train_model.py
step=3: python main.py
step=4:
Option 1 (Quick Fix)

Click the Train Model button.

If it changes to:

✅ Model Trained

then enter:

Age: 30
Gender: Male
Education: Bachelor's
Job Title: Software Engineer
Experience: 5

and click Predict Salary.