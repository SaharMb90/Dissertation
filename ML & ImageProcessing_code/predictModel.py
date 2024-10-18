import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

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
X = data[['Temperature', 'TDS', 'EC', 'Light Intensity', 'pH Level', 'Growth rate 1', 'Growth rate 2']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define ensemble models using Bagging and Stacking

# 1. Bagging with Decision Tree Regressor
bagging_tree = BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42), n_estimators=50, random_state=42)

# 2. Stacking with Random Forest and Decision Tree Regressors
stacking_model = StackingRegressor(
    estimators=[('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('dt', DecisionTreeRegressor(random_state=42))],
    final_estimator=LinearRegression()
)
# Define hyperparameter grid for Random Forest (for comparison)
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Hyperparameter tuning for Random Forest
print("\nTuning Random Forest...")
rf = RandomForestRegressor(random_state=42)
rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=5, n_jobs=-1, scoring='r2')
rf_grid_search.fit(X_train, y_train)
best_rf = rf_grid_search.best_estimator_
print(f"Best Hyperparameters for Random Forest: {rf_grid_search.best_params_}")

# Train the models, make predictions, and evaluate performance
models = {
    "Best Random Forest": best_rf,
    "Best Gradient Boosting": GradientBoostingRegressor(learning_rate=0.2, max_depth=3, min_samples_leaf=2, min_samples_split=10, n_estimators=50, random_state=42),
    "Bagging (Decision Tree)": bagging_tree,
    "Stacking (Random Forest + Decision Tree)": stacking_model,
    "Linear Regression": LinearRegression(),
    "Support Vector Regressor": SVR()
}
# Train each model, make predictions, and evaluate performance
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error and R² Score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - Mean Squared Error (MSE): {mse:.2f}, R² Score: {r2:.2f}")
