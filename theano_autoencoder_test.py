__author__ = 'du'

import numpy as np
from road_to_deep_learning.theano import LogisticRegression, AutoEncoder, functions
from sklearn import datasets

np.random.seed(0)
iris = datasets.load_iris()
X = iris.data
y = iris.target

n_dim = X.shape[1]
n_classes = len(set(y))

ae = AutoEncoder(n_dim, 5, tol=1e-6, lr=0.001, n_iter=100000, report=1000).fit(X)
z = ae.transform(X)

print(z)

model = LogisticRegression(z.shape[1], n_classes, tol=1e-6, lr=0.005, n_iter=100000, report=1000, C=1.0).fit(z, y)
print(model.get_variables())
print(model.score(z, y))
