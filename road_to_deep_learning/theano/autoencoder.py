__author__ = 'du'

from .mlp import MLPTransformer


class AutoEncoder(MLPTransformer):
    pass


class Stack(object):
    def __init__(self, aes):
        self.aes = aes

    def fit(self, X, y=None):
        z = X
        for ae in self.aes:
            ae.fit(z)
            z = ae.transform(z)
            print(z.shape)

        return self

    def transform(self, X):
        z = X
        for ae in self.aes:
            z = ae.transform(z)
        return z
