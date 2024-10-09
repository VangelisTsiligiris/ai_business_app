# clustering_model.py

import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# Load the data
data = pd.read_csv('data/cleaned_segmentation_data.csv')

# Features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize KMeans with n_init set explicitly
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
kmeans.fit(X_scaled)

# Assign cluster labels
data['Cluster'] = kmeans.labels_

# Save the model and scaler
joblib.dump(kmeans, 'models/clustering_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save the clustered data
data.to_csv('data/clustered_data.csv', index=False)
