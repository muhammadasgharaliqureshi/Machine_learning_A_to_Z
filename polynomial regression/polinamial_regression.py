# -*- coding: utf-8 -*-
"""
Created on Wed May 12 22:32:33 2021

POLYNOMIAL REGRESSION

@author: Qureshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##################################################################
################# Loading Dataset  ###############################

data_set = pd.read_csv('./Position_Salaries.csv')


##################################################################
####################### Defining X and Y from data set ###########

X = data_set.iloc[:, 1:2].values
Y = data_set.iloc[:, -1].values

##################################################################
################### Making simple linear regression for compare 
# purpose #######################################################

from sklearn.linear_model import LinearRegression
Lin_regressor = LinearRegression()

Lin_regressor.fit(X, Y)
pred_lin_Y  = Lin_regressor.predict(X)

##################################################################
######################  making polynomial regressor #############

## first converting IV into polynomial_IV     ###################

from sklearn.preprocessing import PolynomialFeatures
PF =  PolynomialFeatures(degree = 3)
X_polynomialed = PF.fit_transform(X)

pol_regressor = LinearRegression()
pol_regressor = pol_regressor.fit(X_polynomialed, Y)
pred_pol_Y = pol_regressor.predict(X_polynomialed)

####################################################################
############ Visualizing both models ###############################

plt.scatter(X, Y, color= 'red')
plt.plot(X, pred_lin_Y, 'blue')
#plt.plot(X, pred_pol_Y, 'green')
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
grid_pred_Y = pol_regressor.predict(PF.fit_transform(X_grid)) 
plt.plot(X_grid, grid_pred_Y  ,color =  '#000080')

plt.title('Linear Regression vs polynomial regression')
plt.xlabel('Ranks')
plt.ylabel('Salary')
plt.show()




