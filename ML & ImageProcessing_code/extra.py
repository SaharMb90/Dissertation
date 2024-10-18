import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from random import sample

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
data = pd.read_csv('data.csv')

# Standardize 'Light Intensity'
data['Light Intensity'] = data['Light Intensity'].str.capitalize()
light_intensity_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
data['Light Intensity'] = data['Light Intensity'].map(light_intensity_mapping)

# Simulate growth rates with variability for the first and second months
data['Growth rate 1'] = np.random.uniform(120, 130, len(data))  # Simulated growth for first month
data['Growth rate 2'] = np.random.uniform(150, 160, len(data))  # Simulated growth for second month

# Simulate third month growth rate with variability
y = 0.5 * data['Growth rate 1'] + 0.5 * data['Growth rate 2']  # New target with variability

# Only focus on key features for now
X = data[['Temperature', 'TDS',  'Light Intensity', 'pH Level', 'Growth rate 1', 'Growth rate 2']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Bagging model with Decision Tree as the base estimator
bagging_model = BaggingRegressor(
    estimator=DecisionTreeRegressor(random_state=42), 
    n_estimators=50, 
    random_state=42
)
bagging_model.fit(X_train, y_train)

# Finding the best conditions based on Bagging (Decision Tree) predictions
print("\nFinding the best conditions based on Bagging (Decision Tree) predictions...")

# Define reduced ranges of values for key features to make the process faster
temp_range = np.linspace(18, 28, 60)  
tds_range = np.linspace(100, 400, 50)  
ph_range = np.linspace(5.5, 9.5, 50)       
light_intensity_range = [0, 1, 2]  # Light Intensity: 0 (Low), 1 (Medium), 2 (High)

# Randomly sample a subset of combinations to reduce computational time
param_space = [(temp, tds, ph, light) for temp in temp_range 
                                       for tds in tds_range 
                                       for ph in ph_range 
                                       for light in light_intensity_range]
sampled_param_space = sample(param_space, 100)  # Randomly sample 100 combinations from the parameter space

# Initialize variables to track the best conditions
best_conditions = None
best_prediction = -np.inf

# Loop through the sampled combinations to find the best conditions
for temp, tds, ph, light in sampled_param_space:
    # Create a sample with these conditions and match the column names
    sample = pd.DataFrame([[temp, tds, light, ph, 128.58, 157.57]], 
                          columns=['Temperature', 'TDS', 'Light Intensity', 'pH Level', 'Growth rate 1', 'Growth rate 2'])
    
    # Scale the sample
    sample_scaled = scaler.transform(sample)
    
    # Predict the growth rate using the Bagging model
    predicted_growth = bagging_model.predict(sample_scaled)[0]
    
    # Update the best conditions if the predicted growth rate is higher
    if predicted_growth > best_prediction:
        best_prediction = predicted_growth
        best_conditions = {'Temperature': temp, 'TDS': tds, 'Light Intensity': light, 'pH Level': ph}

# Print the best conditions found
print("\nBest Conditions for Maximum Predicted Growth Rate (Bagging with Decision Tree):")
for key, value in best_conditions.items():
    if isinstance(value, float):  # Apply .2f format only for floats
        print(f"{key}: {value:.2f}")
    else:  # For string (Light Intensity)
        print(f"{key}: {value}")
print(f"Predicted Growth Rate: {best_prediction:.2f}%")
