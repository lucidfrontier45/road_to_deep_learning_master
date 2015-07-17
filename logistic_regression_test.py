__author__ = 'du'

import numpy as np
from road_to_deep_learning.naive import SGDLogisticRegression, BatchLogisticRegression
from sklearn import datasets

np.random.seed(0)
iris = datasets.load_iris()
X = iris.data
y = iris.target

n_dim = X.shape[1]
n_classes = len(set(y))

model1 = SGDLogisticRegression(n_dim, n_classes, batch_size=50, tol=1e-7, lr=0.001,
                               n_iter=100000, report=1000, C=0.5).fit(X, y)
print(model1.W_)
print(model1.score(X, y))

model2 = BatchLogisticRegression(n_dim, n_classes, tol=1e-7, n_iter=1000000, C=0.5, method="CG").fit(X, y)
print(model2.W_)
print(model2.score(X, y))
