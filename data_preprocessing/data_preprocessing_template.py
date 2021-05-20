# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:01:53 2021
data preprocessing templete
@author: Qureshi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##############LOAD DATA SET#####################
data_set = pd.read_csv('./Data.csv')

#############DEFINING DEPENDENT AND INDEPENDENT VARIABLES#######################

X = data_set.iloc[0:10,0:-1].values
Y = data_set.iloc[0:10,-1].values

#############################SPLIT TEST AND TRAIN SET########################

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42 )

##############################FEATURE SCALING##########################
'''from sklearn.preprocessing import StandardScaler
std_sc = StandardScaler()

std_sc_X = std_sc
X_train = std_sc_X.fit_transform(X_train)
X_test = std_sc_X.fit(X_test)'''
