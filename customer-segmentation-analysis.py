import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# Verify and print current working directory
print("Current Working Directory:", os.getcwd())

# Load the dataset
file_path = 'customer_segmentation_data.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Drop non-numeric columns that are not needed
df.drop('id', axis=1, inplace=True)  # Drop 'id' column

# Encode categorical variables
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['preferred_category'] = label_encoder.fit_transform(df['preferred_category'])

# Check data types
print("Data types of columns after encoding:")
print(df.dtypes)

# Ensure all columns are numeric before scaling
numeric_df = df.select_dtypes(include=[np.number])

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# Optional: Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)
reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])

# Determine the optimal number of clusters using the elbow method
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Apply K-means with the optimal number of clusters (e.g., 5)
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
df['Cluster'] = clusters
reduced_df['Cluster'] = clusters

# Analyze the clusters
cluster_analysis = df.groupby('Cluster').mean()
print("Cluster analysis:")
print(cluster_analysis)

# Plot the clusters in PCA-reduced space
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=reduced_df, palette='viridis')
plt.title('Customer Segments')
plt.show()

# Visualize clusters with respect to original features
plt.figure(figsize=(12, 8))
for cluster in df['Cluster'].unique():
    sns.scatterplot(df[df['Cluster'] == cluster]['last_purchase_amount'], 
                    df[df['Cluster'] == cluster]['age'], 
                    label=f'Cluster {cluster}')

plt.title('Customer Segments by Age and Last Purchase Amount')
plt.xlabel('Last Purchase Amount')
plt.ylabel('Age')
plt.legend()
plt.show()
