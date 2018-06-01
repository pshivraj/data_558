"""
Developed by pshivraj@uw.edu

This module performs l2-regularized logistic regression with both gradient descent and fast gradient descent Algorithm.

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier


def obj(X, y, beta, lamda=0.1):
    """Computes gradient of ojbective funtion.

    Parameters
    ----------
    X: features
    y: labels
    beta: beta parameter
    lamda: lambda parameter
    Returns
    -------

    ojbective value.
    """
    loss = (np.sum(np.log(1 + np.exp(-y * X.dot(beta))))/len(y))
    reg = lamda*(np.linalg.norm(beta))**2
    return loss + reg

def computegrad(X, y, beta, lamda=0.1):
    """Computes gradient of obj funtion.

    Parameters
    ----------
    X: features
    y: labels
    beta: beta parameter
    lamda: lambda parameter
    Returns
    -------

    gradient of ojbective funtion.
    """
    num = np.exp(-y*(X.dot(beta)))
    diag_val = np.diag(num/(1+num))
    return X.T.dot(diag_val.dot(y))*(-1/len(y)) + 2*lamda*beta

def backtracking(X, y, beta, grad, t=1, alpha=0.5, beta1=0.5):
    """ Backtracking rule for step size.
    Parameters
    ----------
    X: features
    y: labels
    beta: beta parameter
    grad: gradient value
    t: prior step size
    alpha: alpha val
    beta1: beta_update
    Returns
    -------
    eta: new step size
    """
    norm_grad_x = np.linalg.norm(grad)  # Norm of the gradient at x
    found_t = False
    max_iter = 0  # Iteration counter
    while found_t is False and max_iter < 50:
        if obj(X, y, beta-t*grad, 0.1) < obj(X, y, beta, 0.1) - alpha*t*norm_grad_x**2:
            found_t = True
        else:
            t *= beta1
            max_iter += 1
    return t

def misclassification(X, y, beta):
    """ Misclassification for test.
    Parameters
    ----------
    X: features
    y: labels
    beta: beta parameter
    Returns
    -------
    eta: misclassification error
    """
    prob_val = np.exp(X.dot(beta))/(1+np.exp(X.dot(beta)))
    y_pred = np.where(prob_val >= 0.5, 1, -1)
    return 1-accuracy_score(y, y_pred)


def graddescent(X, y, beta, t_init=1, eps=0.001, lamda=0.1):
    """ gradient descent algorithm.
    Parameters
    ----------
    X: features
    y: labels
    beta: initialized beta value
    t_init: initialized theta value
    eps: init step size
    lamda: lambda value
    Returns
    -------
    beta: beta value of iterations
    x_vals: obj value of iterations
    """
    x_vals = []
    pred_val = []
    beta_val = []
    grad = computegrad(X, y, beta, lamda)
    max_iter = 0
    while np.linalg.norm(grad) > eps and max_iter < 50:
        t = t_init
        t = backtracking(X, y, beta, grad, t)
        beta = beta -t*grad
        x_vals.append(obj(X, y, beta, lamda))
        grad = computegrad(X, y, beta, lamda)
        pred_val.append(misclassification(X, y, beta))
        beta_val.append(beta)
        max_iter += 1
    return beta, x_vals

def fastgradalgo(X, y, beta, t_init=1, eps=0.001, lamda=0.1):
    """ fast gradient descent algorithm.
    Parameters
    ----------
    X: features
    y: labels
    beta: initialized beta value
    t_init: initialized theta value
    eps: init step size
    lamda: lambda value
    Returns
    -------
    beta: beta value of iterations
    x_vals: obj value of iterations
    """
    x_vals = []
    pred_val = []
    beta_compute = beta
    beta_update = beta
    beta_val = []


    grad = computegrad(X, y, beta_compute, lamda)
    max_iter = 0
    while np.linalg.norm(grad) > eps and max_iter < 50:
        t = t_init
        t = backtracking(X, y, beta_compute, grad, t)
        beta = beta_update
        beta_update = beta_compute -t*grad
        beta_compute = beta_update + (t/(t+3))*(beta_update-beta)
        x_vals.append(obj(X, y, beta_compute, lamda))
        grad = computegrad(X, y, beta_compute, lamda)
        pred_val.append(misclassification(X, y, beta_compute))
        beta_val.append(beta)

        max_iter += 1
    return beta, x_vals

def fit(X, y, t_init=1, eps=0.001, lamda=0.1, grad_descent='grad'):
    """ fit gradient descent.
    Parameters
    ----------
    X: features
    y: labels
    t_init: initialized theta value
    eps: init step size
    lamda: lambda value
    grad_descent: type of gradient descent

    Returns
    -------
    beta: beta value of iterations
    x_vals: obj value of iterations
    """
    np.random.seed(2017)
    beta = np.zeros(X.shape[1])
    if grad_descent == 'grad':
        optimized_data_grad = graddescent(X, y, beta=beta, t_init=1, eps=0.001, lamda=0.1)
    elif grad_descent == 'fast':
        optimized_data_grad = fastgradalgo(X, y, beta=beta, t_init=1, eps=0.001, lamda=0.1)
    return optimized_data_grad

def plot_grad_vs_fast_obj(grad_data, fast_data):
    """Plots the change in objective values with each iteration.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(grad_data, label='Grad')
    plt.plot(fast_data, label='Fast Grad')
    plt.title('Training Objective')
    plt.grid()
    plt.legend()
    plt.show()

def sklearn_compare_fast(X, y, fast_data, alpha=0.1, max_iter=50):
    """Plots the comparison of sklearn and own implimentation.
    """
    clf = SGDClassifier(loss='log', alpha=0.1, fit_intercept=False, max_iter=50)
    clf.fit(X, y)
    sklearn_beta = clf.coef_
    plt.plot(sklearn_beta[0], label='Sklearn Obj')
    plt.plot(fast_data, label='FastGrad Obj')
    plt.title('Training Objective')
    plt.grid()
    plt.legend()
    plt.show()
