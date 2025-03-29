---
title: "Implementing Polynomial Regression for Predicting Car Price (From Scratch)"
date: 2025-03-29
categories: [Projects,Machine Learning]
tags: [Projects,ML,scikit-learn,from-scratch,polynomial-regression]
---


## Introduction

Bellow is my [notebook from Kaggle](https://www.kaggle.com/code/ammarlouah/car-price-prediction-polynomial-regression) for my project on implementing Polynomial Regression from scratch for Predicting Car Price and comparing it to scikit-learn predefined one. \
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

    /kaggle/input/car-price-prediction/CarPrice_Assignment.csv
    /kaggle/input/car-price-prediction/Data Dictionary - carprices.xlsx
    

### Overview
In this notebook i will predict Car Price using polynomial regression. i will implement everything from scratch then compare my results to a predefined algorithm in scikit learn.

### Dataset

first i will load and explore the dataset.
i'm working on Car Price Prediction Data Set on Kaggle 

https://www.kaggle.com/datasets/hellbuoy/car-price-prediction


```python
df = pd.read_csv("/kaggle/input/car-price-prediction/CarPrice_Assignment.csv")
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
      <th>car_ID</th>
      <th>symboling</th>
      <th>CarName</th>
      <th>fueltype</th>
      <th>aspiration</th>
      <th>doornumber</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>wheelbase</th>
      <th>...</th>
      <th>enginesize</th>
      <th>fuelsystem</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>alfa-romero giulia</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>alfa-romero stelvio</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>alfa-romero Quadrifoglio</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2</td>
      <td>audi 100 ls</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2</td>
      <td>audi 100ls</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
df.shape
```




    (205, 26)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 205 entries, 0 to 204
    Data columns (total 26 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   car_ID            205 non-null    int64  
     1   symboling         205 non-null    int64  
     2   CarName           205 non-null    object 
     3   fueltype          205 non-null    object 
     4   aspiration        205 non-null    object 
     5   doornumber        205 non-null    object 
     6   carbody           205 non-null    object 
     7   drivewheel        205 non-null    object 
     8   enginelocation    205 non-null    object 
     9   wheelbase         205 non-null    float64
     10  carlength         205 non-null    float64
     11  carwidth          205 non-null    float64
     12  carheight         205 non-null    float64
     13  curbweight        205 non-null    int64  
     14  enginetype        205 non-null    object 
     15  cylindernumber    205 non-null    object 
     16  enginesize        205 non-null    int64  
     17  fuelsystem        205 non-null    object 
     18  boreratio         205 non-null    float64
     19  stroke            205 non-null    float64
     20  compressionratio  205 non-null    float64
     21  horsepower        205 non-null    int64  
     22  peakrpm           205 non-null    int64  
     23  citympg           205 non-null    int64  
     24  highwaympg        205 non-null    int64  
     25  price             205 non-null    float64
    dtypes: float64(8), int64(8), object(10)
    memory usage: 41.8+ KB
    


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
      <th>car_ID</th>
      <th>symboling</th>
      <th>wheelbase</th>
      <th>carlength</th>
      <th>carwidth</th>
      <th>carheight</th>
      <th>curbweight</th>
      <th>enginesize</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>103.000000</td>
      <td>0.834146</td>
      <td>98.756585</td>
      <td>174.049268</td>
      <td>65.907805</td>
      <td>53.724878</td>
      <td>2555.565854</td>
      <td>126.907317</td>
      <td>3.329756</td>
      <td>3.255415</td>
      <td>10.142537</td>
      <td>104.117073</td>
      <td>5125.121951</td>
      <td>25.219512</td>
      <td>30.751220</td>
      <td>13276.710571</td>
    </tr>
    <tr>
      <th>std</th>
      <td>59.322565</td>
      <td>1.245307</td>
      <td>6.021776</td>
      <td>12.337289</td>
      <td>2.145204</td>
      <td>2.443522</td>
      <td>520.680204</td>
      <td>41.642693</td>
      <td>0.270844</td>
      <td>0.313597</td>
      <td>3.972040</td>
      <td>39.544167</td>
      <td>476.985643</td>
      <td>6.542142</td>
      <td>6.886443</td>
      <td>7988.852332</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>-2.000000</td>
      <td>86.600000</td>
      <td>141.100000</td>
      <td>60.300000</td>
      <td>47.800000</td>
      <td>1488.000000</td>
      <td>61.000000</td>
      <td>2.540000</td>
      <td>2.070000</td>
      <td>7.000000</td>
      <td>48.000000</td>
      <td>4150.000000</td>
      <td>13.000000</td>
      <td>16.000000</td>
      <td>5118.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>52.000000</td>
      <td>0.000000</td>
      <td>94.500000</td>
      <td>166.300000</td>
      <td>64.100000</td>
      <td>52.000000</td>
      <td>2145.000000</td>
      <td>97.000000</td>
      <td>3.150000</td>
      <td>3.110000</td>
      <td>8.600000</td>
      <td>70.000000</td>
      <td>4800.000000</td>
      <td>19.000000</td>
      <td>25.000000</td>
      <td>7788.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>103.000000</td>
      <td>1.000000</td>
      <td>97.000000</td>
      <td>173.200000</td>
      <td>65.500000</td>
      <td>54.100000</td>
      <td>2414.000000</td>
      <td>120.000000</td>
      <td>3.310000</td>
      <td>3.290000</td>
      <td>9.000000</td>
      <td>95.000000</td>
      <td>5200.000000</td>
      <td>24.000000</td>
      <td>30.000000</td>
      <td>10295.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>154.000000</td>
      <td>2.000000</td>
      <td>102.400000</td>
      <td>183.100000</td>
      <td>66.900000</td>
      <td>55.500000</td>
      <td>2935.000000</td>
      <td>141.000000</td>
      <td>3.580000</td>
      <td>3.410000</td>
      <td>9.400000</td>
      <td>116.000000</td>
      <td>5500.000000</td>
      <td>30.000000</td>
      <td>34.000000</td>
      <td>16503.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>205.000000</td>
      <td>3.000000</td>
      <td>120.900000</td>
      <td>208.100000</td>
      <td>72.300000</td>
      <td>59.800000</td>
      <td>4066.000000</td>
      <td>326.000000</td>
      <td>3.940000</td>
      <td>4.170000</td>
      <td>23.000000</td>
      <td>288.000000</td>
      <td>6600.000000</td>
      <td>49.000000</td>
      <td>54.000000</td>
      <td>45400.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Shuffle the data 

data = df.sample(frac=1,random_state=42).reset_index(drop=True)
```


```python
# Calculate the number of samples for each set
train_size = int(0.8 * len(data))
test_size = int(0.2 * len(data))

# Split the dataset into training and test sets
train_data = data[:train_size] # training data (80%)
test_data = data[train_size:] # test data (20%)
```


```python
data.head()
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
      <th>car_ID</th>
      <th>symboling</th>
      <th>CarName</th>
      <th>fueltype</th>
      <th>aspiration</th>
      <th>doornumber</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>wheelbase</th>
      <th>...</th>
      <th>enginesize</th>
      <th>fuelsystem</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16</td>
      <td>0</td>
      <td>bmw x4</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>103.5</td>
      <td>...</td>
      <td>209</td>
      <td>mpfi</td>
      <td>3.62</td>
      <td>3.39</td>
      <td>8.00</td>
      <td>182</td>
      <td>5400</td>
      <td>16</td>
      <td>22</td>
      <td>30760.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>0</td>
      <td>audi 5000s (diesel)</td>
      <td>gas</td>
      <td>turbo</td>
      <td>two</td>
      <td>hatchback</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.5</td>
      <td>...</td>
      <td>131</td>
      <td>mpfi</td>
      <td>3.13</td>
      <td>3.40</td>
      <td>7.00</td>
      <td>160</td>
      <td>5500</td>
      <td>16</td>
      <td>22</td>
      <td>17859.167</td>
    </tr>
    <tr>
      <th>2</th>
      <td>101</td>
      <td>0</td>
      <td>nissan nv200</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>97.2</td>
      <td>...</td>
      <td>120</td>
      <td>2bbl</td>
      <td>3.33</td>
      <td>3.47</td>
      <td>8.50</td>
      <td>97</td>
      <td>5200</td>
      <td>27</td>
      <td>34</td>
      <td>9549.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>133</td>
      <td>3</td>
      <td>saab 99e</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.1</td>
      <td>...</td>
      <td>121</td>
      <td>mpfi</td>
      <td>3.54</td>
      <td>3.07</td>
      <td>9.31</td>
      <td>110</td>
      <td>5250</td>
      <td>21</td>
      <td>28</td>
      <td>11850.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>69</td>
      <td>-1</td>
      <td>buick century luxus (sw)</td>
      <td>diesel</td>
      <td>turbo</td>
      <td>four</td>
      <td>wagon</td>
      <td>rwd</td>
      <td>front</td>
      <td>110.0</td>
      <td>...</td>
      <td>183</td>
      <td>idi</td>
      <td>3.58</td>
      <td>3.64</td>
      <td>21.50</td>
      <td>123</td>
      <td>4350</td>
      <td>22</td>
      <td>25</td>
      <td>28248.000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
numerical_features = ['wheelbase', 'carlength', 'carwidth', 'carheight',
                      'curbweight', 'enginesize', 'boreratio', 'stroke',
                      'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']
target = 'price'
```


```python
# Split the features and target for each set
X_train = train_data[numerical_features].values
y_train = train_data[target].values

X_test = test_data[numerical_features].values
y_test = test_data[target].values

```


```python
print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
```

    Training set size: 164 samples
    Test set size: 41 samples
    

## Models
### Polynomial Regression from scratch


```python
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.01, iterations=1000, scale=True):
        """
        Initialize the polynomial regression model.
        
        Parameters:
        - degree: The degree of the polynomial features.
        - learning_rate: The learning rate (alpha) for gradient descent.
        - iterations: Number of iterations for gradient descent.
        - scale: Boolean flag to apply manual feature scaling.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.scale = scale
        self.bias = 0.0
        self.weights = None
        self.cost_history = []
        # To store scaling parameters
        self.means = None
        self.stds = None

    def _scale_features(self, X):
        """
        Scale the features manually: standardize to zero mean and unit variance.
        
        Parameters:
        - X: A numpy array of shape (m, n).
        
        Returns:
        - X_scaled: The scaled features.
        """
        if self.means is None or self.stds is None:
            self.means = np.mean(X, axis=0)
            self.stds = np.std(X, axis=0)
            # Avoid division by zero
            self.stds[self.stds == 0] = 1.0
        return (X - self.means) / self.stds

    def _create_polynomial_features(self, X):
        """
        Create polynomial features for the input data X.
        
        Parameters:
        - X: A numpy array of shape (m, n) where m is the number of samples 
             and n is the number of original features.
             
        Returns:
        - X_poly: A numpy array of shape (m, n * degree) containing polynomial features.
        """
        # Optionally scale the features
        if self.scale:
            X = self._scale_features(X)
        
        m, n = X.shape
        poly_features = []
        for d in range(1, self.degree + 1):
            poly_features.append(np.power(X, d))
        X_poly = np.concatenate(poly_features, axis=1)
        return X_poly

    def _compute_cost(self, X_poly, y):
        """
        Compute the mean squared error cost.
        
        Parameters:
        - X_poly: Feature matrix.
        - y: True target values.
        
        Returns:
        - cost: The computed cost value.
        """
        m = len(y)
        predictions = self.bias + X_poly.dot(self.weights)
        error = predictions - y
        cost = (1 / (2 * m)) * np.sum(np.square(error))
        return cost

    def fit(self, X, y):
        """
        Fit the polynomial regression model using gradient descent,
        calculating bias and weights separately.
        
        Parameters:
        - X: A numpy array of shape (m, n) with the original features.
        - y: A numpy array of shape (m,) with the target values.
        """
        # Generate polynomial features (scaling is applied if self.scale=True)
        X_poly = self._create_polynomial_features(X)
        m, n_poly = X_poly.shape

        # Initialize parameters
        self.bias = 0.0
        self.weights = np.zeros(n_poly)
        self.cost_history = []

        # Gradient Descent loop
        for i in range(self.iterations):
            predictions = self.bias + X_poly.dot(self.weights)
            error = predictions - y

            # Compute gradients
            bias_gradient = (1 / m) * np.sum(error)
            weights_gradient = (1 / m) * X_poly.T.dot(error)

            # Update parameters
            self.bias -= self.learning_rate * bias_gradient
            self.weights -= self.learning_rate * weights_gradient

            cost = self._compute_cost(X_poly, y)
            self.cost_history.append(cost)

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        - X: A numpy array of shape (m, n) with the original features.
        
        Returns:
        - predictions: A numpy array of shape (m,) with the predicted values.
        """
        # Use the same scaling parameters stored during fit
        if self.scale:
            X = (X - self.means) / self.stds
        X_poly = self._create_polynomial_features(X)  # _create_polynomial_features scales again; avoid double scaling
        # To avoid double scaling, we can create a separate method for prediction features:
        m, n = X.shape
        poly_features = []
        for d in range(1, self.degree + 1):
            poly_features.append(np.power(X, d))
        X_poly = np.concatenate(poly_features, axis=1)
        predictions = self.bias + X_poly.dot(self.weights)
        return predictions

    def plot_cost_history(self):
        """
        Plot the cost history over iterations.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.cost_history, label="Cost over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Gradient Descent Convergence")
        plt.legend()
        plt.show()

# Instantiate the polynomial regression model with degree 2
poly_reg = PolynomialRegression(degree=2, learning_rate=0.001, iterations=1000, scale=True)

# Fit the model using the training data
poly_reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = poly_reg.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Test Mean Squared Error:", mse)

# Plot the cost history to observe convergence
poly_reg.plot_cost_history()
```

    Test Mean Squared Error: 59698053.45753728
    


    
![png](/assets/img/posts/car-price-prediction-polynomial-regression/output_16_1.png)
    


### Polynomial Regression using scikit-learn


```python
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

# First, manually scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create polynomial features using scikit-learn
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_test_poly = poly_features.transform(X_test_scaled)

# Fit a LinearRegression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)
y_pred_sklearn = lin_reg.predict(X_test_poly)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
print("scikit-learn Polynomial Regression MSE:", mse_sklearn)
```

    scikit-learn Polynomial Regression MSE: 234990527.49339545
    


```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Compute error metrics for manual implementation
mse_manual = mean_squared_error(y_test, y_pred)
rmse_manual = np.sqrt(mse_manual)
mae_manual = mean_absolute_error(y_test, y_pred)
r2_manual = r2_score(y_test, y_pred)

# Compute error metrics for scikit-learn implementation
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
rmse_sklearn = np.sqrt(mse_sklearn)
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

# Print results
print("Manual Polynomial Regression:")
print(f"   MSE  : {mse_manual:.4f}")
print(f"   RMSE : {rmse_manual:.4f}")
print(f"   MAE  : {mae_manual:.4f}")
print(f"   R²   : {r2_manual:.4f}")
print("\nScikit-Learn Polynomial Regression:")
print(f"   MSE  : {mse_sklearn:.4f}")
print(f"   RMSE : {rmse_sklearn:.4f}")
print(f"   MAE  : {mae_sklearn:.4f}")
print(f"   R²   : {r2_sklearn:.4f}")

```

    Manual Polynomial Regression:
       MSE  : 59698053.4575
       RMSE : 7726.4515
       MAE  : 4168.0573
       R²   : 0.2301
    
    Scikit-Learn Polynomial Regression:
       MSE  : 234990527.4934
       RMSE : 15329.4008
       MAE  : 7138.0699
       R²   : -2.0305
    
