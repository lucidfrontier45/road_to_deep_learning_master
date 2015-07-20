__author__ = 'du'

import numpy as np
from road_to_deep_learning.utils import optimize
from road_to_deep_learning.naive import MLPClassifier
from sklearn import datasets

np.random.seed(0)
iris = datasets.load_iris()
X = iris.data
y = iris.target

n_dim = X.shape[1]
n_hidden = 3
n_classes = len(set(y))

model = MLPClassifier(n_dim, n_hidden, n_classes, optimizer=optimize.scipy_minimize, batch_size=100, tol=1e-7, lr=0.001,
                               n_iter=100000, report=1000).fit(X, y)
print(model.w_hidden, model.w_input)
print(model.score(X, y))

