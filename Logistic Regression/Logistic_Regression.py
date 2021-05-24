# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:15:13 2021
Logistic Regression
@author: Muhammad Asghar Ali Qureshi
"""


##############################################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

##############################################################################
############################## Loading DataSet ###############################

data_set = pd.read_csv('./Social_Network_Ads.csv')

##############################################################################
####################### Defining I.V and D.V #################################

X = data_set.iloc[:, 2:-1].values
Y = data_set.iloc[:, -1].values

##############################################################################

####################### Splitting Test And Train #############################

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25,
                                                    random_state = 42)


#############################################################################
################### Applying Feature Scalling ###############################

from sklearn.preprocessing import StandardScaler
std_scalar = StandardScaler()
std_scalar_train = std_scalar.fit(X_train)
std_scalar_train = std_scalar_train.transform(X_train)

std_scalar_test = std_scalar.fit(X_test)
std_scalar_test = std_scalar_test.transform(X_test)

X_train, X_test  = std_scalar_train, std_scalar_test

##############################################################################
################# Creating classifier for logistic Regression ################

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=42)
classifier = classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

##############################################################################
############ Analyzing model by Confusion Matrix #############################

from sklearn.metrics import confusion_matrix
cfm = confusion_matrix(Y_test, Y_pred)

##############################################################################
############## Visualizing model #############################################

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('#3f000f', '#02075d')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('#3f000f', '#02075d'))(i), label = j)
plt.title('Logistic Regression (Train set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('#3f000f', '#02075d')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('#3f000f', '#02075d'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show() 

