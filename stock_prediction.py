#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 11:50:14 2019

@author: kalyantulabandu
"""

'''
steps:
    0. import required libraries
    1. import dataset
    2. clean dataset
    3. encode categorical data
    4. feature scaline
    5. train and test set split
    6. fit models
    7. predict
    8. plot
'''
#Import libraries

import pandas as pd
import datetime
import math
import numpy as np

#Import dataset
dataset = pd.read_csv('SCHW.csv')


#calculate moving average of closing price
closing_price = dataset.iloc[:,-2].values
closing_price = dataset['Adj Close']
mavg = closing_price.rolling(window=100).mean()

#creating dataset by date, adj close, volume, hl_pct, pct_change
dfreg = dataset.loc[:,['Date','Adj Close','Volume']]
dfreg['HL_PCT'] = (dataset['High'] - dataset['Low'])/dataset['Close'] * 100.0
dfreg['PCT_Change'] = (dataset['Close'] - dataset['Open']) / dataset['Open'] * 100.0

import datetime as dt
dfreg['Date'] = pd.to_datetime(dfreg['Date'])
dfreg['Date'] = dfreg['Date'].map(dt.datetime.toordinal)


#clearning dataset
#drop missing values
dfreg.fillna(value=-99999, inplace=True)

#separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01*len(dfreg)))

#separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'],1))

#SCale the X so that everyone can have the same distribution for linear regression
import sklearn.preprocessing
X = sklearn.preprocessing.scale(X)

#Finally we want to find data series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

#Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]


# splitting X & y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

#model generation and prediction
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

#Linear Regression
clfreg = LinearRegression(n_jobs=1)
clfreg.fit(X_train,y_train)
y_pred = clfreg.predict(X_test)
confidencereg = clfreg.score(X_test,y_test)

#Ridge Regression
rr = Ridge(alpha=0.01)
rr.fit(X_train,y_train)
y_pred_ridge = rr.predict(X_test)
confidenceridge = rr.score(X_test,y_test)

#Lasso Regression
ls = Lasso()
ls.fit(X_train,y_train)
y_pred_lasso = ls.predict(X_test)
confidencelasso = ls.score(X_test,y_test)

#plotting learning curves for linear regression
import matplotlib.pyplot as plt
plt.plot(y_test[:100])
plt.plot(y_pred[:100])
plt.legend(['Actual', 'Linear Predicted'], loc='upper right')
plt.show()


#plotting learning curves for linear regression
import matplotlib.pyplot as plt
plt.plot(y_test[:100])
plt.plot(y_pred_ridge[:100])
plt.legend(['Actual', 'Ridge Predicted'], loc='upper right')
plt.show()

#plotting learning curves for linear regression
import matplotlib.pyplot as plt
plt.plot(y_test[:100])
plt.plot(y_pred_lasso[:100])
plt.legend(['Actual', 'Lasso Predicted'], loc='upper right')
plt.show()