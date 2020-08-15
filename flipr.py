# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 18:06:47 2020

@author: hp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 
from matplotlib import pyplot
import streamlit as st
#
##encoding the categorical variables
#train_data = pd.get_dummies(train_data, columns=["Industry"]) 
#train_data = pd.get_dummies(train_data, columns=["Index"]) 
#train_data = train_data.drop(['Stock Index'],axis = 1)
#
##train_data['Stock Index'] = train_data['Stock Index'].str.replace(r'[^\d.]+', '')
#
#output = train_data['Stock Price']
#train_data = train_data.drop(['Stock Price'],axis=1)
#train_data = train_data.fillna(train_data.mean())

train_data = pd.read_csv('Train_dataset.csv')
test_data = pd.read_csv('test (1).csv')

output = train_data['Stock Price']
train_data = train_data.drop(['Stock Price'],axis=1)

def preprocessing(data):
    data = pd.get_dummies(data, columns=["Industry"]) 
    data = pd.get_dummies(data, columns=["Index"]) 
    data = data.drop(['Stock Index'],axis = 1)
    data = data.fillna(data.mean())
    
    return data



train = preprocessing(train_data)
test = preprocessing(test_data)



from sklearn.ensemble import RandomForestRegressor
regressorRF = RandomForestRegressor(n_estimators = 100, random_state = 0,max_features = "log2",oob_score = True)
regressorRF.fit(train, output)

predicted_output_rf =regressorRF.predict(test)
predicted_output_rf= pd.DataFrame(predicted_output_rf)
predicted_output_rf.to_csv('output_rf.csv')

from sklearn.tree import DecisionTreeRegressor
regressorDT = DecisionTreeRegressor(random_state = 0)
regressorDT.fit(train, output)

predicted_output_dt =regressorDT.predict(test)

data = pd.read_csv('pull_call_test.csv')
data = data.drop(['Stock Index'],axis = 1)
data = data.iloc[1:]
data['Put-Call Ratio'] = data['Put-Call Ratio'].astype(float)
data['Unnamed: 2'] = data['Unnamed: 2'].astype(float)
data['Unnamed: 3'] = data['Unnamed: 3'].astype(float)
data['Unnamed: 4'] = data['Unnamed: 4'].astype(float)
data['Unnamed: 5'] = data['Unnamed: 5'].astype(float)
data['Unnamed: 6'] = data['Unnamed: 6'].astype(float)
data = data.fillna(data.mean())
series = data['Unnamed: 5']

data.plot()
pyplot.show()

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(data['Unnamed: 5'], order=(1,1,2))
model_fit = model.fit()
print(model_fit.summary())
aug11 = model_fit.predict()

from statsmodels.tsa.vector_ar.var_model import VAR
model_var = VAR(data)
model_var_fit = model_var.fit()
var_predicted = model_var_fit.forecast(model_var_fit.y,steps=50)

from statsmodels.tsa.statespace.varmax import VARMAX
model = VARMAX(data)
model_fit = model.fit() 
aug16_predicted = model_fit.predict()
aug16_predicted.to_csv('output_02_16th_aug.csv')
forecast = model_fit.forecast()
model_fit.plot_diagnostics()
print(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()



