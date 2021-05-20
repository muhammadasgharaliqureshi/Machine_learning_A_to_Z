# -*- coding: utf-8 -*-
"""
Created on Fri May 14 07:29:27 2021

Support Vector Regression

@author: Qureshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##############################################################################
############################### Loading Data_set #############################
data_set = pd.read_csv('Position_Salaries.csv')
##############################################################################

############################# Defining X and Y ###############################

X = data_set.iloc[:, 1:2].values
Y = data_set.iloc[:, 2:3].values

##############################################################################
######################## Applying Feature scalling ###########################

from sklearn.preprocessing import StandardScaler
stnd_scl = StandardScaler()
stnd_scl_X = stnd_scl.fit(X)
X = stnd_scl_X.transform(X)
stnd_scl_Y = stnd_scl.fit(Y)
Y = stnd_scl_Y.transform(Y)
Y = Y[:, -1]
##############################################################################
######################## Support Vector Regression ###########################
from sklearn.svm import SVR
Vector_regressor = SVR(kernel='rbf')
Vector_regressor = Vector_regressor.fit(X, Y)
pred_y = Vector_regressor.predict(X)
##############################################################################

###################### visualizing SVR #######################################
plt.scatter(stnd_scl_X.inverse_transform(X), stnd_scl_Y.inverse_transform(Y), color = 'red')
plt.plot(stnd_scl_X.inverse_transform(X), stnd_scl_Y.inverse_transform( pred_y), color = 'blue')

plt.show()

print(stnd_scl_Y.inverse_transform( Vector_regressor.predict(stnd_scl_X.transform(np.array([[6.5]])))))
