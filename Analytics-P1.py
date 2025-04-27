# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Try reading the CSV file using a different encoding 
try:
    df = pd.read_csv('allDataMean.csv', encoding='latin1')
except Exception as e:
    print('Error reading CSV:', e)
else:
    print('CSV loaded successfully with latin1 encoding')
    
    # Display basic information about the dataset
    print("\
Dataset Info:")
    print(df.info())
    print("\
First few rows of the data:")
    print(df.head())

    # Quick summary description of numerical data if present
    print("\
Summary Statistics:")
    print(df.describe())

    # Plot some visualizations if possible. 
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        # Plot a pairplot for numeric columns, using only first 100 rows to reduce load
        sns.pairplot(df[numeric_cols].iloc[:100])
        plt.show()
        print('Pairplot for numeric columns plotted.')
    else:
        print('No numeric columns found for pairplot.')

    # Display the correlation heatmap if numeric columns exist
    if numeric_cols:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.show()
        print('Correlation heatmap plotted.')
    else:
        print('No numeric columns found for correlation heatmap.')

    print('Data visualization done.')