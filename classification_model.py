# train_classification_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Ensure the 'models' directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Load the dataset
# Adjust the path to your dataset accordingly
df = pd.read_csv('data/cleaned_churn_data.csv')

# Select the relevant features
features = ['gender', 'SeniorCitizen', 'InternetService', 'Contract', 'PaymentMethod']
X = df[features]
y = df['Churn']

# Handle missing values if any
X = X.fillna('No')

# Encode categorical variables
label_encoders = {}
for col in features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model and label encoders
joblib.dump(model, 'models/classification_model.pkl')
joblib.dump(label_encoders, 'models/classification_encoders.pkl')

# Optional: Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model trained successfully with accuracy: {accuracy:.2f}")

