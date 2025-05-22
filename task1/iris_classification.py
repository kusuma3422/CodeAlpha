# iris_classification.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Show basic info
print("First 5 rows of the dataset:")
print(df.head())
print("\nSpecies distribution:")
print(df['species'].value_counts())

# Optional: visualize pairplot
sns.pairplot(df, hue='species')
plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.show()

# Features and labels
X = df.drop('species', axis=1)
y = df['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Predict a new sample
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example flower
sample_scaled = scaler.transform(sample)
predicted_class = model.predict(sample_scaled)
print(f"\nPredicted class for sample {sample.tolist()[0]}: {predicted_class[0]}")
