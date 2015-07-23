__author__ = 'du'

import numpy as np
from .mlp import MLPTransformer


class AutoEncoder(MLPTransformer):
    pass


class DenoisingAutoEncoder(AutoEncoder):
    def __init__(self, input_dim, hidden_dim, noise_ratio=0.1, **params):
        AutoEncoder.__init__(self, input_dim, hidden_dim)
        self.noise_ratio = noise_ratio

    def fit(self, X, y=None):
        X_noisy = X + np.random.randn(*X.shape) * X.std(0) * self.noise_ratio
        AutoEncoder.fit(self, X, X_noisy)


class Stack(object):
    def __init__(self, aes):
        self.aes = aes

    def fit(self, X, y=None):
        z = X
        for ae in self.aes:
            ae.fit(z)
            z = ae.transform(z)

        return self

    def transform(self, X):
        z = X
        for ae in self.aes:
            z = ae.transform(z)
        return z
