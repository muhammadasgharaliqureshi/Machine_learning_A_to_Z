# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:03:42 2021
K Nearest Neighbour
@author: Qureshi
"""
##############################################################################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#############################################################################
#####################  Loading Data set #####################################

data_set = pd.read_csv('E:/asghar/MY PYTHON STUFF/Machine_learning_A_to_Z/K Nearest Neighbour/Social_Network_Ads.csv')

##############################################################################
###################### Defining dependent and independent variables ##########

X = data_set.iloc[:,2:-1].values
Y = data_set.iloc[:, -1].values

###############################################################################
########### Spliting training and testing data ################################
###############################################################################

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)

###############################################################################
############## Creating KNN classifier object ################################
##############################################################################

from sklearn.neighbors import KNeighborsClassifier


KNN_classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
KNN_classifier = KNN_classifier.fit(X_train, Y_train)
Y_predict = KNN_classifier = KNN_classifier.predict(X_test)

##############################################################################
################# Confusion matrix ############################################

from sklearn.metrics import confusion_matrix
cfm = confusion_matrix(Y_test, Y_predict)
##############################################################################
###################### Visualizing results ###################################

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, KNN_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('#3f000f', '#02075d')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('#3f000f', '#02075d'))(i), label = j)
plt.title('KNN Classification (Train set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, KNN_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('#3f000f', '#02075d')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('#3f000f', '#02075d'))(i), label = j)
plt.title('KNN_classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show() 






