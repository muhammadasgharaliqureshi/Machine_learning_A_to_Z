# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:33:55 2021
Simple linear regression
@author: Qureshi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

######Loading Data Set##########
data_set = pd.read_csv('./Salary_Data.csv')
#################################
####Loading data preprocessing template############
###############################################

X = data_set.iloc[:,:-1 ]
Y = data_set.iloc[:, -1]

###############################################
#########Splitting data into train and test set##########
###############################################

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 42) 

##################################################################
#########Creating Simple Linear Regression machine###############
#################################################################
from sklearn.linear_model import LinearRegression
regressor   = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

prediction_Y = regressor.predict(X_test)

################################################################
##########Now Visualizing Data##################################

################Visualization of training data##################
plt.scatter(X_train, Y_train, color='Red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Visualization of train data VS Regression line')
plt.xlabel('Experence')
plt.ylabel('Salary')

plt.show()


################Visualization of testing data##################
plt.scatter(X_test, Y_test, color='Red')
plt.plot(X_test, prediction_Y, color = 'blue')

plt.title('Visualization of train data VS Regression line')
plt.xlabel('Experence')
plt.ylabel('Salary')

plt.show()





