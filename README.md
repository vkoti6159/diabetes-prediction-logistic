# diabetes-prediction-logistic
Multiple Logistic Regression Model to Predict Diabetes
# Diabetes Prediction using Multiple Logistic Regression

This project uses a multiple logistic regression model to predict the likelihood of a person having diabetes. The model takes into account four key health features:

- Age
- BMI (Body Mass Index)
- Blood Pressure
- Glucose Level

### 📊 Dataset
A small synthetic dataset with realistic values was used for training and testing. The data contains both positive and negative diabetes cases.

### 🧠 Model
The model was trained using `scikit-learn`'s LogisticRegression and evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

### ⚙️ Libraries Used
- `numpy`: for numerical operations
- `pandas`: for data manipulation
- `scikit-learn`: for building and evaluating the logistic regression model

### 🧪 Features
- Predicts whether a person is diabetic based on health inputs.
- Displays prediction result and probability score.
- Includes a custom function to take user input and predict diabetes.
- Optional: [Streamlit UI version included for interactive prediction]

### 🚀 How to Run
```bash
pip install -r requirements.txt
python diabetes_prediction_logistic_regression.py

