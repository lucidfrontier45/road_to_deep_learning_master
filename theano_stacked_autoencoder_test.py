__author__ = 'du'

import numpy as np
from road_to_deep_learning.theano import LogisticRegression, AutoEncoder, Stack
from sklearn import datasets, pipeline

np.random.seed(0)
iris = datasets.load_iris()
X = iris.data
y = iris.target

n_dim = X.shape[1]
n_classes = len(set(y))

ae1 = AutoEncoder(n_dim, 20, tol=1e-7, lr=0.0005, n_iter=1000000, report=10000)
ae2 = AutoEncoder(20, 10, tol=1e-7, lr=0.0005, n_iter=1000000, report=10000)
ae3 = AutoEncoder(10, 5, tol=1e-6, lr=0.001, n_iter=1000000, report=10000)

sae = Stack([ae1, ae2, ae3]).fit(X)
z = sae.transform(X)

print(z)

model = LogisticRegression(z.shape[1], n_classes, tol=1e-6, lr=0.005, n_iter=100000, report=1000, C=1.0).fit(z, y)
print(model.get_variables())
print(model.score(z, y))
