"""
author: pshivraj@uw.edu

This module generates simlated dataset.

"""
from model.fastalgo import fit, plot_grad_vs_fast_obj, sklearn_compare_fast
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Simulate a dataset with  200 obervations and 10 features
class_1 = []
class_2 = []
for i in range(0,100):
    class_1.append(np.random.normal(10,5,10))
    class_2.append(np.random.normal(15,5,10))
label_1 = [-1]*100
label_2 = [1]*100

features = np.concatenate((class_1,class_2),axis=0)
label = np.concatenate((label_1,label_2),axis=0)


# Random train-test split. Test set contain 20% of the data.
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.20, random_state=2017)
# Standardize the data
scaler = preprocessing.StandardScaler().fit(X_train)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Append a column of 1's to X_train and X_test for the intercept
X_train = preprocessing.add_dummy_feature(X_train)
X_test = preprocessing.add_dummy_feature(X_test)

#Fit model based on both grad and fast grad
grad_data = fit(X_train,y_train,t_init=1,eps=0.001,lamda=0.1, grad_descent='grad')
fast_data = fit(X_train,y_train,t_init=1,eps=0.001,lamda=0.1, grad_descent='fast')

#Analyze training objective with Iteration
plot_grad_vs_fast_obj(grad_data[1],fast_data[1])

#Comapre sklearn and own algo training Objective
sklearn_compare_fast(X_train,y_train,fast_data[0],alpha=0.1, max_iter=50)
