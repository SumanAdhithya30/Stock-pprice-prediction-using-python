#!/usr/bin/env python
# coding: utf-8

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

# Importing the dataset
df = pd.read_csv('TSLA.csv')
print(df.head())

# Dataset shape and info
print(df.shape)
print(df.describe())
print(df.info())

# Exploratory Data Analysis
plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Tesla Close Price', fontsize=15)
plt.ylabel('Price in dollars')
plt.show()

# Check if 'Close' is equal to 'Adj Close'
print(df[df['Close'] == df['Adj Close']].shape)

# Drop 'Adj Close' column
df = df.drop(['Adj Close'], axis=1)

# Check for null values
print(df.isnull().sum())

# Visualize distributions of features
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(20, 10))

for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.histplot(df[col], kde=True)
plt.show()

# Visualize boxplots of features
plt.subplots(figsize=(20, 10))

for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.boxplot(df[col])
plt.show()

# Feature Engineering
splitted = df['Date'].str.split('/', expand=True)
df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')
print(df.head())

# Grouped data by year and mean visualization
data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20, 10))

for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2, 2, i + 1)
    data_grouped[col].plot.bar()
plt.show()

# Adding new features
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
print(df.head())

# Group by 'is_quarter_end'
print(df.groupby('is_quarter_end').mean())

# Adding more features
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Target distribution pie chart
plt.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
plt.show()

# Heatmap of highly correlated features
plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()

# Data Splitting and Normalization
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

# Model Development and Evaluation
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

for model in models:
    model.fit(X_train, Y_train)
    print(f'{model} : ')
    print('Training Accuracy: ', metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1]))
    print('Validation Accuracy: ', metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1]))
    print()
