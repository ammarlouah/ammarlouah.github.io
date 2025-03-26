---
title: "Implementing Linear Regression for Predicting House Prices (From Scratch)"
date: 2025-03-26
categories: [Projects,Machine Learning]
tags: [Projects,ML,scikit-learn,from-scratch]
---

## Introduction

Bellow is my notebook from Kaggle for my project on implementing Linear Regression from scratch for Predicting House Prices and comparing it to scikit-learn predefined one.\
Enjoy!

## Notebook

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/the-boston-houseprice-data/boston.csv
    

### Overview
In this notebook i will predict the house prices using linear regression.
i will implement everything from scratch then compare my results to a predefined algorithm in scikit learn.

### Dataset

first i will load and explore the dataset.
i'm working on Boston House Prices on Kaggle 

https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data


```python
df = pd.read_csv("/kaggle/input/the-boston-houseprice-data/boston.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (506, 14)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 14 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   CRIM     506 non-null    float64
     1   ZN       506 non-null    float64
     2   INDUS    506 non-null    float64
     3   CHAS     506 non-null    int64  
     4   NOX      506 non-null    float64
     5   RM       506 non-null    float64
     6   AGE      506 non-null    float64
     7   DIS      506 non-null    float64
     8   RAD      506 non-null    int64  
     9   TAX      506 non-null    float64
     10  PTRATIO  506 non-null    float64
     11  B        506 non-null    float64
     12  LSTAT    506 non-null    float64
     13  MEDV     506 non-null    float64
    dtypes: float64(12), int64(2)
    memory usage: 55.5 KB
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.613524</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
      <td>22.532806</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.601545</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
      <td>9.197104</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
      <td>17.025000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
      <td>21.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.677083</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Shuffle the data
data = df.sample(frac=1, random_state=42).reset_index(drop=True)
```


```python
# Calculate the number of samples for each set
train_size = int(0.7 * len(data))
val_size = int(0.15 * len(data))

# Split the dataset into training, validation, and test sets
train_data = data[:train_size] # training data (70%)
val_data = data[train_size:train_size+val_size] # validation data (15%)
test_data = data[train_size+val_size:] # test data (15%)
```


```python
# Split the features and target for each set
X_train = train_data.drop('MEDV', axis=1)
y_train = train_data['MEDV']

X_val = val_data.drop('MEDV', axis=1)
y_val = val_data['MEDV']

X_test = test_data.drop('MEDV', axis=1)
y_test = test_data['MEDV']
```


```python
print(f"Training set size: {len(X_train)} samples")
print(f"Validation set size: {len(X_val)} samples")
print(f"Test set size: {len(X_test)} samples")
```

    Training set size: 354 samples
    Validation set size: 75 samples
    Test set size: 77 samples
    

## Model

### Linear Regression from scratch

```python
class LinearRegression:
    def __init__(self, alpha=0.01, iterations=1000, scale=False):
        self.alpha = alpha  # Learning rate
        self.iterations = iterations  # Number of iterations for gradient descent
        self.scale = scale  # Whether to scale the features or not
        self.w = None  # Weights
        self.b = None  # Bias
        self.cost_history = []  # List to track cost history

    def scale_features(self, X):
        """Scale features using mean and std deviation (standardization)."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std

    def fit(self, X, y):
        """Fit the model to the training data using gradient descent."""
        m = len(y)  # Number of training examples
        
        # If scaling is needed, scale the features
        if self.scale:
            X = self.scale_features(X)

        # Initialize weights (w) and bias (b)
        self.w = np.zeros(X.shape[1])
        self.b = 0

        # Perform gradient descent
        for i in range(self.iterations):
            predictions = X.dot(self.w) + self.b
            error = predictions - y

            # Gradient of the cost function with respect to w and b
            dw = (1/m) * np.dot(X.T, error)
            db = (1/m) * np.sum(error)

            # Update weights and bias
            self.w -= self.alpha * dw
            self.b -= self.alpha * db

            # Calculate the cost (Mean Squared Error)
            cost = (1/(2*m)) * np.sum(error ** 2)
            self.cost_history.append(cost)

    def predict(self, X):
        """Make predictions using the trained model."""
        # If scaling was applied, scale the new data as well
        if self.scale:
            X = (X - self.mean) / self.std
        return X.dot(self.w) + self.b

    def get_cost_history(self):
        return self.cost_history
```


```python
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_pred):
    m = len(y_true)
    return (1/m) * np.sum((y_true - y_pred) ** 2)

# Initialize and train the model
model = LinearRegression(alpha=0.01, iterations=1000, scale=True)
model.fit(X_train, y_train)

cost_history = model.get_cost_history()

plt.plot(range(1, len(cost_history) + 1), cost_history, color='blue')
plt.title('Cost History During Training')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

# Make predictions on all datasets
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calculate Mean Squared Error for all datasets
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Training MSE: {train_mse}")
print(f"Validation MSE: {val_mse}")
print(f"Test MSE: {test_mse}")
```


    
![png](assets\img\posts\linear-regression-predicting-house-prices\output_12_0.png)
    


    Training MSE: 22.384617927405444
    Validation MSE: 21.208792092413585
    Test MSE: 22.102794782467647
    


```python
# Plot the actual vs predicted values for all datasets
plt.figure(figsize=(15, 5))

# Training data plot
plt.subplot(1, 3, 1)
plt.scatter(y_train, y_train_pred, color='blue', label="Train Set")
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', label="Ideal")
plt.title('Training Set: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Validation data plot
plt.subplot(1, 3, 2)
plt.scatter(y_val, y_val_pred, color='green', label="Validation Set")
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', label="Ideal")
plt.title('Validation Set: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Test data plot
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_test_pred, color='orange', label="Test Set")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label="Ideal")
plt.title('Test Set: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

plt.tight_layout()
plt.show()
```


    
![png](assets\img\posts\linear-regression-predicting-house-prices\output_13_0.png)
    


### Linear Regression using scikit-learn and comparison


```python
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.metrics import mean_squared_error

# Scikit-learn Linear Regression Model
sklearn_model = SklearnLR()
sklearn_model.fit(X_train, y_train)
y_train_pred_sklearn = sklearn_model.predict(X_train)
y_val_pred_sklearn = sklearn_model.predict(X_val)
y_test_pred_sklearn = sklearn_model.predict(X_test)

# Calculate Mean Squared Error for both models
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_mse_sklearn = mean_squared_error(y_train, y_train_pred_sklearn)
val_mse_sklearn = mean_squared_error(y_val, y_val_pred_sklearn)
test_mse_sklearn = mean_squared_error(y_test, y_test_pred_sklearn)

print("Custom Model MSE:")
print(f"Training MSE: {train_mse}")
print(f"Validation MSE: {val_mse}")
print(f"Test MSE: {test_mse}")

print("\nScikit-learn Model MSE:")
print(f"Training MSE: {train_mse_sklearn}")
print(f"Validation MSE: {val_mse_sklearn}")
print(f"Test MSE: {test_mse_sklearn}")

# Plot the actual vs predicted values for both models

plt.figure(figsize=(15, 10))

# Custom Model - Training Data
plt.subplot(2, 2, 1)
plt.scatter(y_train, y_train_pred, color='blue', label="Custom Model - Train Set")
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', label="Ideal")
plt.title('Custom Model - Training Set: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Scikit-learn Model - Training Data
plt.subplot(2, 2, 2)
plt.scatter(y_train, y_train_pred_sklearn, color='green', label="Scikit-learn Model - Train Set")
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', label="Ideal")
plt.title('Scikit-learn Model - Training Set: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Custom Model - Test Data
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_test_pred, color='orange', label="Custom Model - Test Set")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label="Ideal")
plt.title('Custom Model - Test Set: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Scikit-learn Model - Test Data
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_test_pred_sklearn, color='purple', label="Scikit-learn Model - Test Set")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label="Ideal")
plt.title('Scikit-learn Model - Test Set: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

plt.tight_layout()
plt.show()
```

    Custom Model MSE:
    Training MSE: 22.384617927405444
    Validation MSE: 21.208792092413585
    Test MSE: 22.102794782467647
    
    Scikit-learn Model MSE:
    Training MSE: 22.018971306970627
    Validation MSE: 22.53291330113675
    Test MSE: 22.14619818293375
    


    
![png](assets\img\posts\linear-regression-predicting-house-prices\output_15_1.png)
    

