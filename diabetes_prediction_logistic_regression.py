# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 21:08:09 2025

@author: varik
"""

# Multiple logstic regression 2nd example

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Expanded dataset
data = pd.DataFrame({
    'Age': [50, 30, 40, 35, 60, 45, 55, 25, 65, 38, 70, 29, 48, 52],
    'BMI': [30, 25, 28, 24, 32, 29, 27, 22, 34, 26, 33, 23, 28, 31],
    'BloodPressure': [85, 70, 75, 80, 90, 78, 82, 65, 92, 77, 88, 68, 79, 85],
    'Glucose': [180, 120, 150, 130, 200, 170, 160, 110, 210, 140, 190, 115, 155, 175],
    'Diabetes': [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1]
})

# Defining the predictors (independent variables) and target variable
X = data[['Age', 'BMI', 'BloodPressure', 'Glucose']]
y = data['Diabetes']

# Splitting the dataset into training and testing sets
# 70% of the data is used for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating an instance of the LogisticRegression model
model = LogisticRegression(max_iter=1000)

# Training the model with the training data
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Evaluating the model's accuracy
print('Accuracy:', accuracy_score(y_test, y_pred))

# Printing the confusion matrix to see the true positives, true negatives, false positives, and false negatives
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Printing the classification report which includes precision, recall, and F1-score for each class
print('Classification Report:\n', classification_report(y_test, y_pred))

# Displaying the coefficients of the logistic regression model
# These coefficients indicate the impact of each predictor on the log odds of the outcome
print('Coefficients:', model.coef_)

# Displaying the intercept of the logistic regression model
print('Intercept:', model.intercept_)

# Function to predict diabetes based on user input
def predict_diabetes(age, bmi, blood_pressure, glucose):
    # Creating a NumPy array with the input data
    input_data = np.array([[age, bmi, blood_pressure, glucose]])
   
    # Using the trained model to make a prediction
    prediction = model.predict(input_data)
   
    # Using the trained model to calculate the probability of having diabetes
    probability = model.predict_proba(input_data)
   
    # Returning the prediction and the probability of having diabetes
    return prediction[0], probability[0][1]

# Prompting the user to enter values for age, BMI, blood pressure, and glucose
age = float(input("Enter Age: "))
bmi = float(input("Enter BMI: "))
blood_pressure = float(input("Enter Blood Pressure: "))
glucose = float(input("Enter Glucose: "))

# Making a prediction based on the user's input
prediction, probability = predict_diabetes(age, bmi, blood_pressure, glucose)

# Displaying the prediction and the probability of having diabetes
print(f"Prediction (1 = Diabetes, 0 = No Diabetes): {prediction}")
print(f"Probability of having Diabetes: {probability:.2f}")