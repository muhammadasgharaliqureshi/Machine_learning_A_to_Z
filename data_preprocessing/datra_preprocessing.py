# -*- coding: utf-8 -*-
"""
Created on Tue May 11 00:43:31 2021

Data preprocessing
@author: Qureshi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_set  = pd.read_csv('./Data.csv')
#creating a independent variable
X = data_set.iloc[0:10, :-1].values
Y = data_set.iloc[0:10, -1].values

# taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[: , 1::])
X[:, 1:3] = imputer.transform(X[:, 1::])

###Now taking care of Catagorical data i.e. encoding data

from sklearn.preprocessing import LabelEncoder
##for independent variable
cat_label = LabelEncoder()
cat_label_X = cat_label.fit(X[:, 0])
cat_label_X = cat_label_X.transform(X[:, 0])
X[:, 0] = cat_label_X
###for dependent cvariable
cat_label_Y = cat_label.fit(Y)
cat_label_Y = cat_label_Y.transform(Y)
Y  =cat_label_Y
##Now catagorical data is being labeled as int, but in order to eliminate a relation of 
#(for example label 0 is < label 1) we weill transform it to one hot encoding
# Remember one hot coding as a truth table type example

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories = 'auto')
#for independent variable
ohe_label_X = ohe.fit_transform(X).toarray()
ohe_label_X = ohe_label_X[:,0:3]
ohe_label_X = np.hstack((ohe_label_X[:, 0:3], X[:, 1:3]))
X = ohe_label_X
#######################################################

## Spliting the data set into training and test set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.25, random_state = 42)


###########################################################
###Now applying feature scalling

from sklearn.preprocessing import StandardScaler
std_sc = StandardScaler()

std_sc_X  = std_sc
std_sc_X = std_sc_X.fit(X_train[:, 3:5])

scaled_X_train = std_sc_X.transform(X_train[:, 3:5])
scaled_X_test = std_sc_X.transform(X_test[:, 3:5])

X_train = np.hstack((X_train[:, 0:3], scaled_X_train[:,:])) 
X_test = np.hstack((X_test[:, 0:3], scaled_X_test[:,:])) 

##################################################################################
####