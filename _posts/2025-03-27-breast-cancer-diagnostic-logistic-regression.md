---
title: "Implementing Logistic Regression for Predicting if a Tumor is Malignant or Benign (From Scratch)"
date: 2025-03-27
categories: [Projects,Machine Learning]
tags: [Projects,ML,scikit-learn,from-scratch,logistic-regression]
---

## Introduction

Bellow is my [notebook from Kaggle](https://www.kaggle.com/code/ammarlouah/breast-cancer-diagnostic-logistic-regression) for my project on implementing Logistic Regression from scratch for Predicting if a Tumor is Malignant or Benign and comparing it to scikit-learn predefined one. \
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

    /kaggle/input/breast-cancer-wisconsin-data/data.csv
    

### Overview
In this notebook i will predict if a Tumor is Malignant or Benign using logistic regression. i will implement everything from scratch then compare my results to a predefined algorithm in scikit learn.

### Dataset

first i will load and explore the dataset.
i'm working on Breast Cancer Wisconsin (Diagnostic) Data Set on Kaggle 

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data


```python
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
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
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
      <th>Unnamed: 32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
df.shape
```




    (569, 33)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 33 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   id                       569 non-null    int64  
     1   diagnosis                569 non-null    object 
     2   radius_mean              569 non-null    float64
     3   texture_mean             569 non-null    float64
     4   perimeter_mean           569 non-null    float64
     5   area_mean                569 non-null    float64
     6   smoothness_mean          569 non-null    float64
     7   compactness_mean         569 non-null    float64
     8   concavity_mean           569 non-null    float64
     9   concave points_mean      569 non-null    float64
     10  symmetry_mean            569 non-null    float64
     11  fractal_dimension_mean   569 non-null    float64
     12  radius_se                569 non-null    float64
     13  texture_se               569 non-null    float64
     14  perimeter_se             569 non-null    float64
     15  area_se                  569 non-null    float64
     16  smoothness_se            569 non-null    float64
     17  compactness_se           569 non-null    float64
     18  concavity_se             569 non-null    float64
     19  concave points_se        569 non-null    float64
     20  symmetry_se              569 non-null    float64
     21  fractal_dimension_se     569 non-null    float64
     22  radius_worst             569 non-null    float64
     23  texture_worst            569 non-null    float64
     24  perimeter_worst          569 non-null    float64
     25  area_worst               569 non-null    float64
     26  smoothness_worst         569 non-null    float64
     27  compactness_worst        569 non-null    float64
     28  concavity_worst          569 non-null    float64
     29  concave points_worst     569 non-null    float64
     30  symmetry_worst           569 non-null    float64
     31  fractal_dimension_worst  569 non-null    float64
     32  Unnamed: 32              0 non-null      float64
    dtypes: float64(31), int64(1), object(1)
    memory usage: 146.8+ KB
    


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
      <th>id</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>...</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
      <th>Unnamed: 32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.690000e+02</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>...</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.037183e+07</td>
      <td>14.127292</td>
      <td>19.289649</td>
      <td>91.969033</td>
      <td>654.889104</td>
      <td>0.096360</td>
      <td>0.104341</td>
      <td>0.088799</td>
      <td>0.048919</td>
      <td>0.181162</td>
      <td>...</td>
      <td>25.677223</td>
      <td>107.261213</td>
      <td>880.583128</td>
      <td>0.132369</td>
      <td>0.254265</td>
      <td>0.272188</td>
      <td>0.114606</td>
      <td>0.290076</td>
      <td>0.083946</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.250206e+08</td>
      <td>3.524049</td>
      <td>4.301036</td>
      <td>24.298981</td>
      <td>351.914129</td>
      <td>0.014064</td>
      <td>0.052813</td>
      <td>0.079720</td>
      <td>0.038803</td>
      <td>0.027414</td>
      <td>...</td>
      <td>6.146258</td>
      <td>33.602542</td>
      <td>569.356993</td>
      <td>0.022832</td>
      <td>0.157336</td>
      <td>0.208624</td>
      <td>0.065732</td>
      <td>0.061867</td>
      <td>0.018061</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.670000e+03</td>
      <td>6.981000</td>
      <td>9.710000</td>
      <td>43.790000</td>
      <td>143.500000</td>
      <td>0.052630</td>
      <td>0.019380</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.106000</td>
      <td>...</td>
      <td>12.020000</td>
      <td>50.410000</td>
      <td>185.200000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156500</td>
      <td>0.055040</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.692180e+05</td>
      <td>11.700000</td>
      <td>16.170000</td>
      <td>75.170000</td>
      <td>420.300000</td>
      <td>0.086370</td>
      <td>0.064920</td>
      <td>0.029560</td>
      <td>0.020310</td>
      <td>0.161900</td>
      <td>...</td>
      <td>21.080000</td>
      <td>84.110000</td>
      <td>515.300000</td>
      <td>0.116600</td>
      <td>0.147200</td>
      <td>0.114500</td>
      <td>0.064930</td>
      <td>0.250400</td>
      <td>0.071460</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.060240e+05</td>
      <td>13.370000</td>
      <td>18.840000</td>
      <td>86.240000</td>
      <td>551.100000</td>
      <td>0.095870</td>
      <td>0.092630</td>
      <td>0.061540</td>
      <td>0.033500</td>
      <td>0.179200</td>
      <td>...</td>
      <td>25.410000</td>
      <td>97.660000</td>
      <td>686.500000</td>
      <td>0.131300</td>
      <td>0.211900</td>
      <td>0.226700</td>
      <td>0.099930</td>
      <td>0.282200</td>
      <td>0.080040</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.813129e+06</td>
      <td>15.780000</td>
      <td>21.800000</td>
      <td>104.100000</td>
      <td>782.700000</td>
      <td>0.105300</td>
      <td>0.130400</td>
      <td>0.130700</td>
      <td>0.074000</td>
      <td>0.195700</td>
      <td>...</td>
      <td>29.720000</td>
      <td>125.400000</td>
      <td>1084.000000</td>
      <td>0.146000</td>
      <td>0.339100</td>
      <td>0.382900</td>
      <td>0.161400</td>
      <td>0.317900</td>
      <td>0.092080</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.113205e+08</td>
      <td>28.110000</td>
      <td>39.280000</td>
      <td>188.500000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>0.304000</td>
      <td>...</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>1.058000</td>
      <td>1.252000</td>
      <td>0.291000</td>
      <td>0.663800</td>
      <td>0.207500</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 32 columns</p>
</div>




```python
list(df.columns.values)
```




    ['id',
     'diagnosis',
     'radius_mean',
     'texture_mean',
     'perimeter_mean',
     'area_mean',
     'smoothness_mean',
     'compactness_mean',
     'concavity_mean',
     'concave points_mean',
     'symmetry_mean',
     'fractal_dimension_mean',
     'radius_se',
     'texture_se',
     'perimeter_se',
     'area_se',
     'smoothness_se',
     'compactness_se',
     'concavity_se',
     'concave points_se',
     'symmetry_se',
     'fractal_dimension_se',
     'radius_worst',
     'texture_worst',
     'perimeter_worst',
     'area_worst',
     'smoothness_worst',
     'compactness_worst',
     'concavity_worst',
     'concave points_worst',
     'symmetry_worst',
     'fractal_dimension_worst',
     'Unnamed: 32']



Here i noticed a column named 'Unnamed: 32' and there values are NaN


```python
df.drop(["Unnamed: 32"], axis = 1, inplace=True)
list(df.columns.values)
```




    ['id',
     'diagnosis',
     'radius_mean',
     'texture_mean',
     'perimeter_mean',
     'area_mean',
     'smoothness_mean',
     'compactness_mean',
     'concavity_mean',
     'concave points_mean',
     'symmetry_mean',
     'fractal_dimension_mean',
     'radius_se',
     'texture_se',
     'perimeter_se',
     'area_se',
     'smoothness_se',
     'compactness_se',
     'concavity_se',
     'concave points_se',
     'symmetry_se',
     'fractal_dimension_se',
     'radius_worst',
     'texture_worst',
     'perimeter_worst',
     'area_worst',
     'smoothness_worst',
     'compactness_worst',
     'concavity_worst',
     'concave points_worst',
     'symmetry_worst',
     'fractal_dimension_worst']



Now we are talking :)


```python
# Shuffle the data
data = df.sample(frac=1, random_state=42).reset_index(drop=True)
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
# Split the features and target for each set
X_train = train_data.drop('diagnosis', axis=1)
y_train = train_data['diagnosis']

X_test = test_data.drop('diagnosis', axis=1)
y_test = test_data['diagnosis']

y_train = y_train.map({'M': 1, 'B': 0}).astype(np.float64)
y_test = y_test.map({'M': 1, 'B': 0}).astype(np.float64)
```


```python
print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
```

    Training set size: 455 samples
    Test set size: 114 samples
    

## Models

### Logistic Regression from scratch


```python
import matplotlib.pyplot as plt

class LogisticRegression :
    def __init__(self, alpha=0.01, iterations=1000, scale=False):
        self.alpha = alpha
        self.iterations = iterations
        self.scale = scale
        self.w = None
        self.b = None
        self.cost_history = []

    def scale_features(self,X):
        """Scale features using mean and std deviation (standardization)."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std


    def fit(self, X, y):
        m = len(y) # Number of training examples

        # Scale the features if needed
        if self.scale : 
            X = self.scale_features(X)
        
        # Initialize weights and bias
        self.w = np.zeros(X.shape[1])
        self.b = 0
        
        for i in range(self.iterations):
            
            z = X.dot(self.w) + self.b
            # pred = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
            pred = 1 / (1 + np.exp(-z))
            
            epsilon = 1e-8
            pred = np.clip(pred, epsilon, 1 - epsilon)
            
            error = pred - y

            dw = (1/m) * np.dot(X.T , error)
            db = (1/m) * np.sum(error)

            # Update the weights and bias
            self.w -= self.alpha * dw
            self.b -= self.alpha * db

            # Calculate and store the cost
            cost = (-1/m) * np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
            self.cost_history.append(cost)

    def predict(self,X):
        """Make predictions using the trained model."""
        if self.scale:
            X = (X - self.mean) / self.std
        logits = X.dot(self.w) + self.b
        prob = 1 / (1 + np.exp(-logits))
        return (prob >= 0.5).astype(int)

    def predict_proba(self, X):
        """Return probability estimates for the positive class."""
        if self.scale:
            X = (X - self.mean) / self.std
        logits = X.dot(self.w) + self.b
        return 1 / (1 + np.exp(-logits))

    def get_cost_history(self):
        return self.cost_history
```


```python
# Initialize and train the model
model = LogisticRegression(alpha=0.1, iterations=1000, scale=True)
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
y_test_pred = model.predict(X_test)
```


    
![png](/assets/img/posts/breast-cancer-diagnostic-logistic-regression/output_18_0.png)
    



```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

report = classification_report(y_test, y_test_pred)
print("Classification Report:")
print(report)
```

    Training Accuracy: 0.9912087912087912
    Test Accuracy: 0.9736842105263158
    Confusion Matrix:
    [[66  1]
     [ 2 45]]
    Classification Report:
                  precision    recall  f1-score   support
    
             0.0       0.97      0.99      0.98        67
             1.0       0.98      0.96      0.97        47
    
        accuracy                           0.97       114
       macro avg       0.97      0.97      0.97       114
    weighted avg       0.97      0.97      0.97       114
    
    

### Logistic Regression using scikit-learn


```python
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the sklearn logistic regression model
clf = SklearnLogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)
clf.fit(X_train_scaled, y_train)

# Make predictions on the training and test sets
y_train_pred_sklearn = clf.predict(X_train_scaled)
y_test_pred_sklearn = clf.predict(X_test_scaled)

# Evaluate the performance
train_accuracy = accuracy_score(y_train, y_train_pred_sklearn)
test_accuracy = accuracy_score(y_test, y_test_pred_sklearn)
conf_matrix = confusion_matrix(y_test, y_test_pred_sklearn)
class_report = classification_report(y_test, y_test_pred_sklearn)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
```

    Training Accuracy: 0.9934065934065934
    Test Accuracy: 0.956140350877193
    Confusion Matrix:
    [[66  1]
     [ 4 43]]
    Classification Report:
                  precision    recall  f1-score   support
    
             0.0       0.94      0.99      0.96        67
             1.0       0.98      0.91      0.95        47
    
        accuracy                           0.96       114
       macro avg       0.96      0.95      0.95       114
    weighted avg       0.96      0.96      0.96       114
    
    
