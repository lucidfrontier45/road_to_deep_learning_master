__author__ = 'du'

import numpy as np
from road_to_deep_learning.theano import MLPClassifier
from sklearn import datasets

np.random.seed(0)
iris = datasets.load_iris()
X = iris.data
y = iris.target

n_dim = X.shape[1]
n_classes = len(set(y))

model = MLPClassifier(n_dim, 5, n_classes, tol=1e-6, lr=0.005, n_iter=100000, report=1000, C=1.0).fit(X, y)
print(model.get_variables())
print(model.score(X, y))
