import torch.nn as nn
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel
from sklearn.metrics import classification_report
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        # nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def cal_km(params, X_fit, X, type):
    if type == 'interface':
        if params['kernel'] == 'linear':
            km = linear_kernel(X_fit, X)
        elif params['kernel'] == 'rbf':
            km = rbf_kernel(X_fit, X, gamma=params['gamma'])
        elif params['kernel'] == 'poly':
            km = polynomial_kernel(X_fit, X, gamma=params['gamma'], coef0=0.0)
        elif params['kernel'] == 'sigmoid':
            km = sigmoid_kernel(X_fit, X, gamma=params['gamma'], coef0=0.0)
        else:
            print('Unknown kernel')
            km = None
    elif type == 'realize':
        if params['kernel'] == 'linear':
            km = cal_linear(X_fit, X)
        elif params['kernel'] == 'rbf':
            km = cal_rbf(X_fit, X, gamma=params['gamma'])
        elif params['kernel'] == 'poly':
            km = cal_poly(X_fit, X, gamma=params['gamma'])
        elif params['kernel'] == 'sigmoid':
            km = cal_sigmoid(X_fit, X, gamma=params['gamma'])
        else:
            print('Unknown kernel')
            km = None
    else:
        print('Unknown type')
        km = None
    return km


def cal_fx(km, coef, intercept):

    fx = np.sum(coef * km.T, axis=1) + intercept

    return fx


# cal_kernel
def cal_linear(X_fit, X_train):
    return np.dot(X_fit, X_train.T)


def cal_rbf(X_fit, X_train, gamma):
    X_a = X_train.reshape(-1, 1, 123)
    km = np.exp(-gamma * np.sum((X_fit - X_a)**2, axis=2))
    return km.T


def cal_poly(X_fit, X_train, gamma, coef0=0.0, degree=3):
    # return (gamma * np.dot(X_fit, X_train.T) + coef0)**degree
    return np.power(gamma * np.dot(X_fit, X_train.T) + coef0, degree)


def cal_sigmoid(X_fit, X_train, gamma, coef0=0.0):
    return np.tanh(gamma * np.dot(X_fit, X_train.T) + coef0)
