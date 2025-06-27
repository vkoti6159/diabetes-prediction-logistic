# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 21:09:25 2025

@author: varik
"""
#Logistic regression
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Create a small synthetic dataset
data = pd.DataFrame({
    'Runs Scored': [250, 300, 150, 200, 280, 350, 230, 270, 190, 300],
    'Wickets Lost': [4, 2, 6, 5, 3, 1, 7, 4, 6, 2],
    'Overs Bowled': [50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
    'Win': [1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
})
# Define the feature variables (independent variables) and target variable (dependent variable)
X = data[['Runs Scored', 'Wickets Lost', 'Overs Bowled']]
y = data['Win']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
conf_matrix = confusion_matrix(y_test, y_pred)

#Simplified Results
#Confusion Matrix: Shows how many predictions matched the actual outcomes.
#Accuracy Score: How often the model was correct.
#Classification Report: Includes precision, recall, and F1-score.

accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation results
print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy Score:")
print(accuracy)
print("\nClassification Report:")
print(class_report)
# Display the model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_[0]}")
print(f"Coefficient for Runs Scored: {model.coef_[0][0]}")
print(f"Coefficient for Wickets Lost: {model.coef_[0][1]}")
print(f"Coefficient for Overs Bowled: {model.coef_[0][2]}")

# Interactive input for new prediction
print("\nEnter match details to predict the outcome:")

# Take user inputs
runs_scored = float(input("Runs Scored: "))
wickets_lost = float(input("Wickets Lost: "))
overs_bowled = float(input("Overs Bowled: "))

# Create DataFrame from user inputs
input_data = pd.DataFrame({
    'Runs Scored': [runs_scored],
    'Wickets Lost': [wickets_lost],
    'Overs Bowled': [overs_bowled]
})

# Predict the outcome for the new input
predicted_win = model.predict(input_data)

# Display the prediction result
print("\nPredicted Outcome for Input Parameters:")
print("India Wins" if predicted_win[0] == 1 else "India Loses")
