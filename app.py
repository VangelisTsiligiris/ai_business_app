# app.py

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the regression model
regression_model = joblib.load('models/regression_model.pkl')

# Load the trained classification model
classification_model = joblib.load('models/classification_model.pkl')

# Load the label encoders
label_encoders = joblib.load('models/classification_encoders.pkl')

# Define the categorical columns
categorical_cols = ['gender', 'SeniorCitizen', 'InternetService', 'Contract', 'PaymentMethod']

def generate_explanation(input_data, prediction):
    reasons = []

    if input_data['Contract'] == 'Month-to-month':
        reasons.append('Customers with month-to-month contracts are more likely to churn.')
    if input_data['PaymentMethod'] == 'Electronic check':
        reasons.append('Customers who pay with electronic checks are more likely to churn.')
    if input_data['InternetService'] == 'Fiber optic':
        reasons.append('Fiber optic users have higher churn rates due to potential service issues.')
    if input_data['SeniorCitizen'] == '1':
        reasons.append('Senior citizens are more likely to churn.')
    if input_data['gender'] == 'Female':
        reasons.append('Female customers have slightly higher churn rates.')

    if prediction == 'Yes':
        explanation = 'The model predicts that the customer will churn because: ' + ' '.join(reasons)
    else:
        explanation = 'The model predicts that the customer will not churn.'

    return explanation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    if request.method == 'POST':
        try:
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
        except Exception as e:
            error_message = f"An error occurred during regression prediction: {e}"
            return render_template('regression.html', error=error_message)
    else:
        return render_template('regression.html')

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        # Get user input from the form
        input_data = request.form.to_dict()

        # Validate inputs (optional but recommended)
        valid_genders = ['Female', 'Male']
        valid_senior_citizens = ['0', '1']
        valid_internet_services = ['DSL', 'Fiber optic', 'No']
        valid_contracts = ['Month-to-month', 'One year', 'Two year']
        valid_payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']

        errors = []
        if input_data['gender'] not in valid_genders:
            errors.append('Invalid gender selected.')
        if input_data['SeniorCitizen'] not in valid_senior_citizens:
            errors.append('Invalid Senior Citizen status.')
        if input_data['InternetService'] not in valid_internet_services:
            errors.append('Invalid Internet Service selected.')
        if input_data['Contract'] not in valid_contracts:
            errors.append('Invalid Contract type selected.')
        if input_data['PaymentMethod'] not in valid_payment_methods:
            errors.append('Invalid Payment Method selected.')

        if errors:
            return render_template('classification.html', errors=errors)

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical variables using the label encoders
        for col in categorical_cols:
            if col in input_df.columns:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except ValueError:
                    # Handle unseen categories
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, input_df[col][0])
                    input_df[col] = label_encoders[col].transform(input_df[col])
            else:
                # If the column is missing in input (should not happen), set to a default value
                input_df[col] = 0

        # Ensure all features are present in the correct order
        model_features = classification_model.feature_names_in_
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        # Make prediction
        try:
            prediction = classification_model.predict(input_df)[0]
            result = 'Yes' if prediction == 1 else 'No'

            # Generate explanation
            explanation = generate_explanation(input_data, result)

            return render_template('classification.html', prediction=result, explanation=explanation)
        except Exception as e:
            error_message = f"An error occurred during classification prediction: {e}"
            return render_template('classification.html', errors=[error_message])
    else:
        return render_template('classification.html')

@app.route('/clustering')
def clustering():
    try:
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
    except Exception as e:
        error_message = f"An error occurred while loading clustering data: {e}"
        return render_template('clustering.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
