# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:12:35 2024

@author: localadmin
"""

from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA
import matplotlib.pyplot as plt

# Start a Dask client
cluster = LocalCluster()
client = Client(cluster)
n_features=2
# Load your data into a Dask DataFrame
df = dd.read_csv('array.csv')
X = df.iloc[:,:-n_features]  # Features (assuming last n columns are target variables)
y = df.iloc[:,-n_features:] # Multi-output targets (last n columns)/to/your/large_dataset.csv')
X = X.to_dask_array(lengths=True)
y = y.to_dask_array(lengths=True)

#print(df.head())

# Assume the last column is the target, and the rest are features

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize PCA
pca = PCA(n_components=2)  # Adjust n_components as needed

# Fit PCA on the scaled data
X_pca = pca.fit_transform(X_scaled)

# Persist the result to avoid recomputation
X_pca = X_pca.persist()

# Convert the result to a Dask DataFrame for easier manipulation
X_pca_df = dd.from_dask_array(X_pca, columns=['PC1', 'PC2'])

# Visualize the first two principal components
X_pca_sample = X_pca_df.sample(frac=0.1).compute()
plt.scatter(X_pca_sample['PC1'], X_pca_sample['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Result')
plt.show()