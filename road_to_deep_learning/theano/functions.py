__author__ = 'du'

import theano.tensor as T

sigmoid = T.nnet.sigmoid
tanh = T.tanh

def identity(x):
    return x

def relu(x):
    return T.maximum(0, x)

activation_functions = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "identity": identity,
    "relu": relu
}
