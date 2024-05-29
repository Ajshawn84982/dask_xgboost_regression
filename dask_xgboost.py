# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:31:55 2024

@author: localadmin
"""

from dask.distributed import Client
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA
import matplotlib.pyplot as plt
import dask.array as da
import xgboost as xgb
from dask_ml.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import dask.array as da


# Start a Dask client (optional, but recommended for distributed computing)
client = Client()
#client = Client(n_workers=4, threads_per_worker=1)


# Read your data into a Dask DataFrame
#f= 'C:/Users/localadmin/downloads/leap-atmospheric-physics-ai-climsim/train.csv'
f='array.csv'

df = dd.read_csv(f)
X = df.iloc[:,:-2]  # Features (assuming last n columns are target variables)
y = df.iloc[:,-2:] # Multi-output targets (last n columns)

# Convert to Dask arrays
X = X.to_dask_array(lengths=True)
y = y.to_dask_array(lengths=True)

# Ensure X and y have the same number of partitions
X = X.rechunk({0: 'auto', 1: None})
y = y.rechunk({0: X.chunks[0]})

if False:
    y_sample = y.head(100000, compute=True)
    
    # Calculate the standard deviation of each column
    stds = y_sample.std()
    # Get the column names of non-constant columns
    non_constant_columns = stds[stds == 0].index
    # Filter y to retain only non-constant columns
    y_filter = y.drop(columns=non_constant_columns) 
    y_filter_value = y_filter.head(100000, compute=True)
    lx,ly=y_filter_value.shape
    
    y_fix = y[[col for col in y.columns if col in non_constant_columns]]
    y_fix_value = y_fix.head(100000, compute=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
models = []
predictions = []

# Loop through each target variable and train a separate model
for i in range(2):
    print(i)
    # Extract the target variable for the current model
    y_train_i = y_train.iloc[:, i]
    y_test_i = y_test.iloc[:, i]
    
    # Convert Dask arrays to DaskDMatrix, which is a Dask-compatible version of XGBoost's DMatrix
    dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train_i)
    dtest = xgb.dask.DaskDMatrix(client, X_test, y_test_i)

    # Set up parameters for the XGBoost model
    params = {
        'objective': 'reg:squarederror',  # for regression task
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    # Train the model
    output = xgb.dask.train(client, params, dtrain, num_boost_round=100)

    # The trained model
    booster = output['booster']
    models.append(booster)
    
    # Make predictions on the test set
    y_pred = xgb.dask.predict(client, booster, dtest)
    predictions.append(y_pred.compute())
    
    # Optionally, evaluate the model
    y_test_computed = y_test_i.compute()
    mse = mean_squared_error(y_test_computed, predictions[-1])
    print(f"Mean Squared Error for target {i+1}: {mse}")

    # Optionally, visualize feature importance
    xgb.plot_importance(booster)
    plt.title(f"Feature Importance for target {i+1}")
    plt.show()