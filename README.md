# Monitoring Module

The `monitoring` module provides two classes: `PredictorsMonitor` and `PredictionsMonitor`.

## PredictorsMonitor

The `PredictorsMonitor` class is designed for monitoring and comparing statistical characteristics of predictors between an "etalon" dataset (used for training) and a test dataset. It supports both numerical and categorical predictors, providing insights into data drift or discrepancies that might affect model performance or data quality.

### Features

1. **Initialization (`__init__`)**:
    - `bins_amt`: number of bins for numerical predictors.

2. **Fitting (`fit`)**:
    - `data`: accept Pandas DataFrame as etalon.
    - `checks`: custom checks for predictors.

3. **Monitoring (`monitor`)**:
    - `data`: accept Pandas DataFrame as test.

4. **Utility Methods**:
    - `get_test_stat`: accept Pandas DataFrame as test.

## PredictionsMonitor

The `PredictionsMonitor` class is designed for monitoring and comparing the statistical characteristics of a model's predictions between a reference ("etalon") and a test predictions. It supports both classification and regression tasks, providing insights into prediction stability, outlier detection, and distribution shifts that could indicate model drift or performance degradation.

### Features

1. **Initialization (`__init__`)**:
    - `task`: task that PredictionsMonitor works on.
    - `bins_amt`: number of bins for predictions.

2. **Fitting (`fit`)**:
    - `data`: accept Numpy ndarray as etalon.
    - `checks`: custom checks for predictions.

3. **Monitoring (`monitor`)**:
    - `data`: accept Numpy ndarray as test.

4. **Utility Methods**:
    - `get_test_stat`: accept Numpy ndarray as test.