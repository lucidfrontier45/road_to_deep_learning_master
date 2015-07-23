__author__ = 'du'

import theano.tensor as T

sigmoid = T.nnet.sigmoid
tanh = T.tanh
identity = lambda x: x
relu = lambda x: T.maximum(0, x)
