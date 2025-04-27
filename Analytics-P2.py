import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read the data
df = pd.read_csv('allDataMean.csv', encoding='latin1')

# Calculate correlations between energy consumption and other metrics
energy_correlations = df.corr()['Active Energy (KWh)'].sort_values(ascending=False)

# Create a bar plot of correlations
plt.figure(figsize=(12, 6))
energy_correlations.plot(kind='bar')
plt.title('Correlations with Active Energy Consumption')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Print key correlations
print("\
Top correlations with Active Energy:")
print(energy_correlations.head())

# Basic statistics for energy consumption
print("\
Energy consumption statistics:")
print(df['Active Energy (KWh)'].describe())

# Calculate efficiency metrics
df['Power_Efficiency'] = df['Active Power (W)'] / df['CPU Power Consumption (%)']
print("\
Power efficiency statistics:")
print(df['Power_Efficiency'].describe())