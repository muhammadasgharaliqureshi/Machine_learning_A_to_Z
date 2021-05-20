# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:42:28 2021
Multiple linear regression
@author: Qureshi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##############################################################################
##############################  LOADING THE DATASET  #########################
data_set = pd.read_csv('./50_Startups.csv')
##############################################################################
###############  Defining dependent and independent variables  ###############

X = data_set.iloc[:, :-1].values
Y = data_set.iloc[:, -1].values
##############################################################################

################## Taking care of catagorical data ###########################
cat_x = X[:, -1]

from sklearn.preprocessing import LabelEncoder
lbl_enc = LabelEncoder()
cat_x = lbl_enc.fit_transform(cat_x)
X[:, -1] = cat_x

##### Now converting independent catagorical data into one hot encode ########
from sklearn.preprocessing import OneHotEncoder
ohc = OneHotEncoder(categories='auto')
ohcx = ohc.fit_transform(X)
cat_x = ohcx.toarray()

cat_x = cat_x[:, 147:150]

X = np.hstack((X[:, 0:3], cat_x[:, 0:3]))

X = X[:, :-1]
##############################################################################
######## Now splitting training and test data ################################

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 42 )

##############################################################################
############# Now making our linear regression model ########################

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)
predict_Y = regressor.predict(X_test)
##############################################################################
################ Visualizing Predictions #####################################

'''plt.scatter(X_test[:, 0], Y_test, color = 'red')
plt.plot(X_test[:, 0], predict_Y)

plt.title("Visualization")
plt.xlabel('Dependent Variables')
plt.ylabel('Independent Variables')
plt.show()'''
##############################################################################

##############################################################################
#########################  Backward Elimination  #############################

import statsmodels.api as sm
X  = np.append(arr = np.ones([50, 1], dtype = int), values = X, axis = 1 ) 

####################### Creating optimal IV ##################################
opt_X = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype = int)
ols_regressor = sm.OLS(endog = Y, exog = opt_X).fit()
print(ols_regressor.summary())


opt_X = np.array(X[:, [0, 1, 2, 3, 5]], dtype = int)
ols_regressor = sm.OLS(endog = Y, exog = opt_X).fit()
print(ols_regressor.summary())

opt_X = np.array(X[:, [0, 1, 2, 3]], dtype = int)
ols_regressor = sm.OLS(endog = Y, exog = opt_X).fit()
print(ols_regressor.summary())


opt_X = np.array(X[:, [0, 1, 3]], dtype = int)
ols_regressor = sm.OLS(endog = Y, exog = opt_X).fit()
print(ols_regressor.summary())


opt_X = np.array(X[:, [0, 1]], dtype = int)
ols_regressor = sm.OLS(endog = Y, exog = opt_X).fit()
print(ols_regressor.summary())