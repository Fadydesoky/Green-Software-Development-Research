# 1. Enhanced Model Performance Analysis
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and prepare data
df = pd.read_csv('allDataMean.csv', encoding='latin1')
features = ['Active Power (W', 'CPU Power Consumption (%)', 'Voltage (V)', 'Current (A)']
X = df[features]
y = df['Active Energy (KWh)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Calculate MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Calculate range of actual values
actual_range = y_test.max() - y_test.min()

print("Enhanced Model Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f} KWh")
print(f"Root Mean Square Error (RMSE): {rmse:.2f} KWh")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R² Score: {r2:.4f}")
print(f"\
Context:")
print(f"Range of actual values: {actual_range:.2f} KWh")
print(f"MAE as percentage of range: {(mae/actual_range)*100:.2f}%")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\
Feature Importance:")
print(feature_importance)