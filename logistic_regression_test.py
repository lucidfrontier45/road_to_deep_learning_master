__author__ = 'du'

import numpy as np
from road_to_deep_learning.utils import optimize
from road_to_deep_learning.naive import SGDLogisticRegression, BatchLogisticRegression, LogisticRegression
from sklearn import datasets

np.random.seed(0)
iris = datasets.load_iris()
X = iris.data
y = iris.target

n_dim = X.shape[1]
n_classes = len(set(y))

model = LogisticRegression(n_dim, n_classes, optimizer=optimize.scipy_minimize, batch_size=100, tol=1e-7, lr=0.001,
                               n_iter=100000, report=1000, C=1.0, method="L-BFGS-B").fit(X, y)
print(model.W_)
print(model.score(X, y))

