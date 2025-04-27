# Import necessary libraries for predictive modeling
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
# Note: Column names from previous output:
# ['WeekDay', 'Voltage (V)', 'Current (A)', 'Active Power (W', 'Frequency (Hz)', 'Active Energy (KWh)', 'Power Factor', 'ESP32 Temperature (°C)', 'CPU Consumption (%)', 'CPU Power Consumption (%)', 'CPU Temperature (%)', 'GPU Consumption (%)', 'GPU Power Consumption (%)', 'GPU Temperature (%)', 'RAM Consumption (%)', 'RAM Power Consumption (%)']

df = pd.read_csv('allDataMean.csv', encoding='latin1')

# Drop rows that might have missing values in key columns
cols_needed = ['Active Energy (KWh)', 'Active Power (W', 'CPU Power Consumption (%)', 'Voltage (V)', 'Current (A)']
df = df.dropna(subset=cols_needed)

# Use a feature set. For simplicity, predict 'Active Energy (KWh)' from a few selected features
features = ['Active Power (W', 'CPU Power Consumption (%)', 'Voltage (V)', 'Current (A)']
X = df[features]
y = df['Active Energy (KWh)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print('Predictive Model Performance:')
print('Mean Absolute Error: ' + str(mae))
print('Mean Squared Error: ' + str(mse))
print('R2 Score: ' + str(r2))

# Summary of correlation analysis
energy_correlations = df.corr()['Active Energy (KWh)'].sort_values(ascending=False)
print('\
Detailed Correlation Analysis with Active Energy (KWh):')
print(energy_correlations.head(10))

print('\
Completed predictive modeling and correlation analysis.')