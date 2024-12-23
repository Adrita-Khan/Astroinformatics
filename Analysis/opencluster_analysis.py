# -*- coding: utf-8 -*-
"""opencluster_analysis

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/195M_LZUQVWC2KmdESRIAwtMQP45_Q9vD
"""

pip install pandas matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the raw GitHub URL
raw_url = 'https://raw.githubusercontent.com/Adrita-Khan/Astroinformatics/main/Datasets/Open%20Cluster/opencluster.tsv'

# Read the TSV file into a pandas DataFrame
try:
    df = pd.read_csv(raw_url, sep='\t')
    print("Dataset successfully loaded!")
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Display information about the dataset
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the raw GitHub URL
raw_url = 'https://raw.githubusercontent.com/Adrita-Khan/AstroSignal/main/Datasets/Open%20Cluster/opencluster.tsv'

# Read the TSV file into a pandas DataFrame with the correct separator
try:
    df = pd.read_csv(raw_url, sep=';')  # Changed sep from '\t' to ';'
    print("Dataset successfully loaded with correct delimiter!")
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Display information about the dataset
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Convert relevant columns to numeric (if not already)
numeric_columns = ['ID', 'Mode', 'Mean', 'beta', 'gamma', 'logM-14', 'logM-25',
                   'logM-50', 'logM-75', 'logM-84', 'logT-14', 'logT-25',
                   'logT-50', 'logT-75', 'logT-84', 'AV-14', 'AV-25',
                   'AV-50', 'AV-75', 'AV-84', 'D5', 'Dnorm5']

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Verify conversion
print("\nData types after conversion:")
print(df.dtypes)

# Handle missing values if any (since initial check shows none, this is precautionary)
# For example, you can fill missing values or drop them
# df = df.dropna()
# or
# df = df.fillna(method='ffill')  # Forward fill as an example

print("\nDetailed Summary Statistics:")
print(df.describe(include='all'))

import math

# Determine number of numerical columns
num_cols = len(numeric_columns)
cols_per_row = 4
rows = math.ceil(num_cols / cols_per_row)

plt.figure(figsize=(20, rows * 4))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(rows, cols_per_row, i)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

sns.pairplot(df[numeric_columns].sample(1000))  # Sampling for performance
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()

plt.figure(figsize=(20, 15))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(math.ceil(num_cols / cols_per_row), cols_per_row, i)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Select features for clustering
features = ['logM-14', 'logM-25', 'logM-50', 'logM-75', 'logM-84',
           'logT-14', 'logT-25', 'logT-50', 'logT-75', 'logT-84',
           'AV-14', 'AV-25', 'AV-50', 'AV-75', 'AV-84', 'D5', 'Dnorm5']

X = df[features]

# Handle missing values if any
X = X.dropna()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# From the Elbow plot, choose an optimal k (e.g., k=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
df['PC1'] = principal_components[:, 0]
df['PC2'] = principal_components[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1', alpha=0.6)
plt.title('Clusters Visualization using PCA')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define target and features
target = 'logM-14'
features = ['Mode', 'Mean', 'beta', 'gamma', 'logM-25', 'logM-50',
           'logM-75', 'logM-84', 'logT-14', 'logT-25', 'logT-50',
           'logT-75', 'logT-84', 'AV-14', 'AV-25', 'AV-50',
           'AV-75', 'AV-84', 'D5', 'Dnorm5']

X = df[features]
y = df[target]

# Handle missing values
X = X.dropna()
y = y[X.index]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nLinear Regression Model Evaluation:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Define target and features
target = 'Mode'
features = ['Mean', 'beta', 'gamma', 'logM-14', 'logM-25', 'logM-50',
           'logM-75', 'logM-84', 'logT-14', 'logT-25', 'logT-50',
           'logT-75', 'logT-84', 'AV-14', 'AV-25', 'AV-50',
           'AV-75', 'AV-84', 'D5', 'Dnorm5']

X = df[features]
y = df[target]

# Handle missing values
X = X.dropna()
y = y[X.index]

# Encode target if it's categorical (assuming 'Mode' is numerical, else use LabelEncoder)
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances[:10], y=feature_importances.index[:10])
plt.title('Top 10 Feature Importances')
plt.show()