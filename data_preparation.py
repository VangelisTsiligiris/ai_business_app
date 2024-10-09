# data_preparation.py

import pandas as pd

# Load Customer Churn Dataset
churn_data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Load Customer Segmentation Dataset
segmentation_data = pd.read_csv('data/Mall_Customers.csv')

# Explore the datasets
print("Churn Data Sample:")
print(churn_data.head())

print("\nSegmentation Data Sample:")
print(segmentation_data.head())

# Customer Churn Data Cleaning
churn_data.dropna(inplace=True)

# Convert TotalCharges to numeric
churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'], errors='coerce')
churn_data['TotalCharges'].fillna(churn_data['TotalCharges'].mean(), inplace=True)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod']

for col in categorical_cols:
    churn_data[col] = le.fit_transform(churn_data[col])

# Encode target variable
churn_data['Churn'] = churn_data['Churn'].map({'Yes': 1, 'No': 0})
# Save the cleaned data
churn_data.to_csv('data/cleaned_churn_data.csv', index=False)

# data_preparation.py (continued)

# Customer Segmentation Data Cleaning
# No missing values in this dataset
# Select relevant features
segmentation_data = segmentation_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Save the cleaned data
segmentation_data.to_csv('data/cleaned_segmentation_data.csv', index=False)

