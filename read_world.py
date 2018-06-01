"""
author: pshivraj@uw.edu

This module reads real world  dataset (Spam_data).

"""

# Load the data
from model.fastalgo import fit, plot_grad_vs_fast_obj, sklearn_compare_fast
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


spam = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data',header=None,delim_whitespace=True)
spam = spam.dropna()
spam[57] = spam[57].replace(0, -1)
# seperate feature and target
X = spam.loc[:,0:57]
y = spam[57]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
# # Standardize the data
scaler = preprocessing.StandardScaler().fit(X_train)
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
