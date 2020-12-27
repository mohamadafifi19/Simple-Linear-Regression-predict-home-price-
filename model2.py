# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 19:40:47 2020

@author: afifi
"""



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import accuracy_score

houses = pd.read_csv("house_data.csv")
houses.isnull().sum() 
corr = houses.corr()
df = houses[['bedrooms','bathrooms','sqft_living','sqft_lot','price']]
sns.pairplot(df, kind="scatter")
plt.show()
x = houses.bedrooms +  houses.bathrooms
x=x/max(x)
y = houses.price
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

cls = linear_model.LinearRegression()
x_train=np.array(x_train)
x_train= x_train.reshape(17290,1)
y_train=np.array(y_train)
y_train=y_train.reshape(17290,1)
x_test=np.array(x_test)
x_test=x_test.reshape(4323,1)

y_test=np.array(x_test)
y_test=y_test.reshape(4323,1)
cls.fit((x_train)**2,(y_train))

y_pred=cls.predict(x_test)

mse = mean_squared_error(y_test,y_pred )
print ('mean square error : '  , np.sqrt(mse))
