# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:49:42 2021
Regression trees
@author: Qureshi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

####################### Reading the dataset ##################################
data_set = pd.read_csv('Position_Salaries.csv')
#print(data_set)


##################### Defining X(Dependent) and Y(Independent) ################

X = data_set.iloc[:, 1:2].values
Y = data_set.iloc[:, 2].values

########## making regressor for regression tree ##############################

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators= 500, random_state = 42)
regressor = regressor.fit(X, Y)
X_t_predict = np.array([[6.5]])
Y_pred = regressor.predict(X_t_predict)

###### Visualization ##############

plt.scatter(X, Y, color= 'red')
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
grid_pred_Y = regressor.predict(X_grid) 
plt.plot(X_grid, grid_pred_Y  ,color =  '#000080')

plt.title('Random Forest regression')
plt.xlabel('Ranks')
plt.ylabel('Salary')
plt.show()
