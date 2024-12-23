# -*- coding: utf-8 -*-
"""SDSS_Plotting_and_Visualization

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wT4IjNFvssFoZSjv0kRq1VFV8d8RdvRT
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime

# Load data from GitHub directly
url = "https://raw.githubusercontent.com/Adrita-Khan/Astroinformatics/main/Basic%20Python%20Operations/Datasets/Skyserver_SQL12_4_2024%208_48_44%20AM.csv"



dataframe = pd.read_csv(url)

# Strip any leading/trailing spaces in the column names
dataframe.columns = dataframe.columns.str.strip()

# Check if the columns are loaded correctly
print(dataframe.columns)

# Inspect the first few rows to check the data
print(dataframe.head())

# Check for missing values
missing_values = dataframe.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Summary statistics for numerical columns
summary_stats = dataframe.describe()
print("Summary statistics:")
print(summary_stats)

# Check data types of the columns
data_types = dataframe.dtypes
print("Data types of each column:")
print(data_types)

# Select only the numeric columns
numeric_df = dataframe.select_dtypes(include=[np.number])

# Calculate the correlation matrix on the numeric columns
correlation_matrix = numeric_df.corr()

# Plot the correlation matrix using a heatmap
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Plot the distribution of magnitude columns
magnitude_columns = ['u', 'g', 'r', 'i', 'z']

plt.figure(figsize=(12, 8))
for i, col in enumerate(magnitude_columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(dataframe[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Plot the distribution of the 'class' column
plt.figure(figsize=(8, 6))
sns.countplot(data=dataframe, x='class')
plt.title('Distribution of Object Classes')
plt.xticks(rotation=45)
plt.show()

# Plot boxplots for magnitude columns to detect outliers
plt.figure(figsize=(12, 8))
for i, col in enumerate(magnitude_columns, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=dataframe, x=col)
    plt.title(f'Outlier Detection for {col}')
plt.tight_layout()
plt.show()

# Scatter plot to investigate redshift vs magnitudes
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dataframe, x='redshift', y='u', color='blue', label='u')
sns.scatterplot(data=dataframe, x='redshift', y='g', color='green', label='g')
sns.scatterplot(data=dataframe, x='redshift', y='r', color='red', label='r')
sns.scatterplot(data=dataframe, x='redshift', y='i', color='purple', label='i')
sns.scatterplot(data=dataframe, x='redshift', y='z', color='orange', label='z')
plt.legend()
plt.title('Redshift vs Magnitude')
plt.xlabel('Redshift')
plt.ylabel('Magnitude')
plt.show()

# Scatter plot for spatial distribution (ra vs dec)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dataframe, x='ra', y='dec', color='darkblue', s=30)
plt.title('Spatial Distribution of Objects (ra vs dec)')
plt.xlabel('Right Ascension (ra)')
plt.ylabel('Declination (dec)')
plt.show()

# Plot time evolution of observations based on 'plate' and 'mjd'
plt.figure(figsize=(12, 8))
sns.scatterplot(data=dataframe, x='mjd', y='plate', hue='redshift', palette='viridis', size='redshift', sizes=(20, 200))
plt.title('Plate vs MJD over Time with Redshift')
plt.xlabel('MJD (Modified Julian Date)')
plt.ylabel('Plate')
plt.show()

# Check for extreme outliers in redshift
outliers_redshift = dataframe[dataframe['redshift'] > 1]  # Redshift values should usually be small (cosmologically speaking)
print(outliers_redshift[['objid', 'ra', 'dec', 'redshift']])

from sklearn.decomposition import PCA

# Apply PCA to reduce dimensions
pca = PCA(n_components=2)
principal_components = pca.fit_transform(dataframe[['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']])

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

# Plot the PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue=dataframe['class'], palette='Set1')
plt.title('PCA of Astronomical Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Pairplot to explore the relationship of redshift with other magnitudes
sns.pairplot(dataframe[['redshift', 'u', 'g', 'r', 'i', 'z']])
plt.show()

# Plot redshift over time (MJD)
plt.figure(figsize=(12, 6))
sns.scatterplot(data=dataframe, x='mjd', y='redshift', hue='class', palette='Set2', size='redshift', sizes=(20, 200), alpha=0.6)
plt.title('Redshift vs MJD (Time Evolution)')
plt.xlabel('Modified Julian Date (MJD)')
plt.ylabel('Redshift')
plt.show()

# Plot the distribution of redshift values
plt.figure(figsize=(8, 6))
sns.histplot(dataframe['redshift'], kde=True, bins=30, color='purple')
plt.title('Distribution of Redshift')
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of magnitudes for each class
magnitude_columns = ['u', 'g', 'r', 'i', 'z']
plt.figure(figsize=(12, 8))
for i, col in enumerate(magnitude_columns, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=dataframe, x='class', y=col)
    plt.title(f'{col} Magnitude by Class')
plt.tight_layout()
plt.show()

# Scatter plot of ra vs dec for each class
plt.figure(figsize=(12, 8))
sns.scatterplot(data=dataframe, x='ra', y='dec', hue='class', palette='Set1', alpha=0.7)
plt.title('Spatial Distribution of Objects by Class (ra vs dec)')
plt.xlabel('Right Ascension (ra)')
plt.ylabel('Declination (dec)')
plt.legend(title='Class')
plt.show()

# Boxplot of redshift by class
plt.figure(figsize=(8, 6))
sns.boxplot(data=dataframe, x='class', y='redshift', palette='Set1')
plt.title('Redshift Distribution by Class')
plt.xlabel('Class')
plt.ylabel('Redshift')
plt.show()

# Calculate mean magnitude by class for each band
mean_magnitudes = dataframe.groupby('class')[['u', 'g', 'r', 'i', 'z']].mean()

# Plot the mean magnitudes
mean_magnitudes.plot(kind='bar', figsize=(10, 6), colormap='Set2')
plt.title('Mean Magnitudes by Class')
plt.xlabel('Class')
plt.ylabel('Mean Magnitude')
plt.xticks(rotation=45)
plt.show()

# Filter high redshift objects
high_redshift = dataframe[dataframe['redshift'] > 1.0]

# Display some characteristics of high redshift objects
print(high_redshift[['objid', 'ra', 'dec', 'class', 'redshift']].head())

from sklearn.decomposition import PCA

# Select the numeric features for PCA
pca_features = dataframe[['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']]

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(pca_features)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

# Visualize PCA results with class labels
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue=dataframe['class'], palette='Set1', s=60, alpha=0.7)
plt.title('PCA of Astronomical Data (ra, dec, magnitudes, redshift)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Class')
plt.show()

# Scatterplot matrix (pairplot) for ra, dec, and other features
sns.pairplot(dataframe[['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']])
plt.show()

# Boxplot of magnitudes by field
plt.figure(figsize=(12, 8))
for i, col in enumerate(magnitude_columns, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=dataframe, x='field', y=col)
    plt.title(f'{col} Magnitude by Field')
plt.tight_layout()
plt.show()

# Check for unrealistic values in magnitude columns
unrealistic_values = dataframe[(dataframe['u'] < 0) | (dataframe['g'] < 0) | (dataframe['r'] < 0) |
                               (dataframe['i'] < 0) | (dataframe['z'] < 0) | (dataframe['redshift'] < 0)]
print(unrealistic_values[['objid', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']])

from sklearn.cluster import KMeans

# Select features for clustering
clustering_features = dataframe[['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
dataframe['cluster'] = kmeans.fit_predict(clustering_features)

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dataframe, x='ra', y='dec', hue='cluster', palette='Set2', s=60, alpha=0.7)
plt.title('Clustering Analysis of Astronomical Data')
plt.xlabel('Right Ascension (ra)')
plt.ylabel('Declination (dec)')
plt.show()

# Create new feature: magnitude difference between different bands
dataframe['g_r_diff'] = dataframe['g'] - dataframe['r']
dataframe['r_i_diff'] = dataframe['r'] - dataframe['i']

# Visualize the distribution of these new features
plt.figure(figsize=(12, 8))
sns.histplot(dataframe['g_r_diff'], kde=True, color='orange', label='g - r')
sns.histplot(dataframe['r_i_diff'], kde=True, color='purple', label='r - i')
plt.legend()
plt.title('Distribution of Magnitude Differences')
plt.show()

# Plot redshift over time (MJD) for different classes
plt.figure(figsize=(12, 6))
sns.lineplot(data=dataframe, x='mjd', y='redshift', hue='class', marker='o', dashes=False)
plt.title('Redshift Evolution Over Time (MJD) by Class')
plt.xlabel('Modified Julian Date (MJD)')
plt.ylabel('Redshift')
plt.legend(title='Class')
plt.show()

# Calculate pairwise correlation of magnitude columns
magnitude_columns = ['u', 'g', 'r', 'i', 'z']
correlation_magnitudes = dataframe[magnitude_columns].corr()

# Plot a heatmap of the pairwise correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_magnitudes, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Magnitude Columns')
plt.show()

from sklearn.cluster import DBSCAN

# Apply DBSCAN clustering (density-based)
dbscan = DBSCAN(eps=0.3, min_samples=10)
dataframe['dbscan_cluster'] = dbscan.fit_predict(clustering_features)

# Plot the DBSCAN clustering results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dataframe, x='ra', y='dec', hue='dbscan_cluster', palette='Set2', s=60, alpha=0.7)
plt.title('DBSCAN Clustering of Astronomical Data')
plt.xlabel('Right Ascension (ra)')
plt.ylabel('Declination (dec)')
plt.show()

from sklearn.cluster import AgglomerativeClustering

# Apply Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)
dataframe['agglo_cluster'] = agglo.fit_predict(clustering_features)

# Plot the Agglomerative Clustering results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dataframe, x='ra', y='dec', hue='agglo_cluster', palette='Set1', s=60, alpha=0.7)
plt.title('Agglomerative Clustering of Astronomical Data')
plt.xlabel('Right Ascension (ra)')
plt.ylabel('Declination (dec)')
plt.show()

from sklearn.ensemble import RandomForestClassifier

# Prepare features and target
X = dataframe[['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']]
y = dataframe['class']

# Train a RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Plot feature importance
feature_importance = rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_names, y=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()

# Violin plots for magnitudes by class
plt.figure(figsize=(12, 8))
for i, col in enumerate(magnitude_columns, 1):
    plt.subplot(2, 3, i)
    sns.violinplot(data=dataframe, x='class', y=col, inner='quart')
    plt.title(f'{col} Magnitude by Class')
plt.tight_layout()
plt.show()

from sklearn.ensemble import IsolationForest

# Train Isolation Forest to detect outliers based on redshift
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(dataframe[['redshift']])

# Add outlier flag to dataframe
dataframe['is_outlier'] = outliers
outliers_data = dataframe[dataframe['is_outlier'] == -1]

# Plot outliers in redshift
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dataframe, x='ra', y='dec', hue='is_outlier', palette={-1: 'red', 1: 'blue'}, s=60, alpha=0.7)
plt.title('Redshift Outliers (Isolation Forest)')
plt.xlabel('Right Ascension (ra)')
plt.ylabel('Declination (dec)')
plt.show()

# Show outliers
print(outliers_data[['objid', 'ra', 'dec', 'redshift']])

