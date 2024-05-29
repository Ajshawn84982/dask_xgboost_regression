# dask_xgboost_regression
A xgboost regressor witht he support of dask for processing large dataset. It directly use the data in dask dataframe to train the xgboost regressor and partition the dataset automatically. I test the input with 1000000 rows with 10 input features and with 2 output features. The current code is still unstable in windows. Waiting for updating of dask.
## Feature importance

<img src='Feature1.jpg' width='250'>
<img src='Feature2.jpg' width='250'>
