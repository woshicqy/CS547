import numpy as np
import h5py
import time
import copy
from random import randint
# activate function
def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s


def compute_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum

    return L


def feed_forward(X, params):
    cache = {}

    # l1 = w1.dot(x) + b1
    cache['l1'] = np.matmul(params['w1'], X) + params['b1']
    # sum_out = np.sum(cache['l1'],axis = 1,keepdims = True)

    # cache['l1'] = cache['l1']/sum_out
    # l1_out = sigmoid(l1)
    cache['l1_out'] = sigmoid(cache['l1'])

    # l2 = w2.dot(l1_out) + b2
    cache['l2'] = np.matmul(params['w2'], cache['l1_out']) + params['b2']


    # sum_out = np.sum(cache['l2'],axis = 1,keepdims = True)
    # cache['l2'] = cache['l2']/sum_out
    # l2_out = softmax(l2)
    # print(np.max(cache['l2']))
    cache['l2_out'] = np.exp(cache['l2']) / np.sum(np.exp(cache['l2']), axis=0)

    return cache


def back_propagate(X, Y, params, cache, m_batch):

    # error at last layer
    dl2 = cache['l2_out'] - Y

    # gradients at last layer (Py2 need 1. to transform to float)
    dw2 = (1. / m_batch) * np.matmul(dl2, cache['l1_out'].T)
    db2 = (1. / m_batch) * np.sum(dl2, axis=1, keepdims=True)

    # back propgate through first layer
    dl1_out = np.matmul(params['w2'].T, dl2)
    dl1 = dl1_out * sigmoid(cache['l1']) * (1 - sigmoid(cache['l1']))

    # gradients at first layer (Py2 need 1. to transform to float)
    dw1 = (1. / m_batch) * np.matmul(dl1, X.T)
    db1 = (1. / m_batch) * np.sum(dl1, axis=1, keepdims=True)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return grads

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def SoftmaxLoss(X, y):
    m = y.shape[0]
    p = softmax(X)
    # print(p.shape)
    # print(y.shape)
    # exit()
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m

    dx = p
    dx[range(m), y] -= 1
    dx /= m
    return loss, dx



class ReLU():

    def __init__(self):

        self.params = []

    def forward(self, X):

        self.X = X
        return np.maximum(X, 0)

    def backward(self, dout):

        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX, []


class sigmoid():
    def __init__(self):

        self.params = []

    def forward(self, X):

        out = 1.0 / (1.0 + np.exp(X))
        self.out = out
        return out

    def backward(self, dout):

        dX = dout * self.out * (1 - self.out)
        return dX, []
