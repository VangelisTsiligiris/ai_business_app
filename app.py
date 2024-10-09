# app.py

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the regression model
regression_model = joblib.load('models/regression_model.pkl')



# Load the trained classification model
classification_model = joblib.load('models/classification_model.pkl')

# Load or recreate the label encoders used during training
# Define the categorical columns
categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaperlessBilling', 'PaymentMethod']

# Create a dictionary to store label encoders for each categorical column
label_encoders = {}

# Known categories for each categorical variable (based on the training data)
categories = {
    'gender': ['Female', 'Male'],
    'SeniorCitizen': [0, 1],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['No phone service', 'Yes', 'No'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No internet service', 'Yes', 'No'],
    'OnlineBackup': ['No internet service', 'Yes', 'No'],
    'DeviceProtection': ['No internet service', 'Yes', 'No'],
    'TechSupport': ['No internet service', 'Yes', 'No'],
    'StreamingTV': ['No internet service', 'Yes', 'No'],
    'StreamingMovies': ['No internet service', 'Yes', 'No'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
}

# Create and fit label encoders
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(categories[col])
    label_encoders[col] = le

# Route definitions
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    if request.method == 'POST':
        # Get form data
        data = [
            float(request.form['MedInc']),
            float(request.form['HouseAge']),
            float(request.form['AveRooms']),
            float(request.form['AveBedrms']),
            float(request.form['Population']),
            float(request.form['AveOccup']),
            float(request.form['Latitude']),
            float(request.form['Longitude']),
        ]

        # Convert to numpy array and reshape
        data_array = np.array([data])

        # Make prediction
        prediction = regression_model.predict(data_array)
        predicted_price = prediction[0] * 100000  # Convert to actual price

        return render_template('regression.html', prediction=predicted_price)
    else:
        return render_template('regression.html')


@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        # Get user input from the form
        input_data = request.form.to_dict()
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Data preprocessing steps
        # Convert numerical columns to numeric types
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numerical_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Handle missing values if any numerical field is empty
        input_df[numerical_cols] = input_df[numerical_cols].fillna(0)

        # Encode categorical variables using the label encoders
        for col in categorical_cols:
            if col in input_df.columns:
                # Handle cases where the input value might not be in the fitted label encoder
                if input_df[col][0] not in label_encoders[col].classes_:
                    # Add the new class to the existing classes
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, input_df[col][0])
                input_df[col] = label_encoders[col].transform(input_df[col])
            else:
                # If the column is missing in input (should not happen), set to a default value
                input_df[col] = 0

        # Ensure all features are present in the correct order
        model_features = classification_model.feature_names_in_
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        # Make prediction
        prediction = classification_model.predict(input_df)
        result = 'Yes' if prediction[0] == 1 else 'No'

        return render_template('classification.html', prediction=result)
    else:
        return render_template('classification.html')

@app.route('/clustering')
def clustering():
    # Load clustered data
    data = pd.read_csv('data/clustered_data.csv')
    
    # Rename columns to avoid special characters
    data.rename(columns={
        'Annual Income (k$)': 'AnnualIncome',
        'Spending Score (1-100)': 'SpendingScore'
    }, inplace=True)
    
    # Replace NaN/None values
    data = data.replace({np.nan: None})
    data = data.astype(object).where(pd.notnull(data), None)
    
    # Convert data to JSON serializable format
    data_json = data.to_dict(orient='records')
    
    # Debugging: Print data_json to verify it's correct
    print("Data being passed to template:", data_json[:5])
    
    return render_template('clustering.html', data=data_json)

if __name__ == '__main__':
    app.run(debug=True)
