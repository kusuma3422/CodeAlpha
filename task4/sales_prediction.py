# sales_prediction.py

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load the dataset
df = pd.read_csv("Advertising.csv")  
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Understand the data
print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# Step 4: Handle missing values
df.dropna(inplace=True)

# Step 5: Encode categorical variables if any
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Step 6: Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 7: Feature selection
# Assume 'Sales' is the target variable
X = df.drop('Sales', axis=1)
y = df['Sales']

# Step 8: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Feature Scaling
scaler = StandardScaler()
X
