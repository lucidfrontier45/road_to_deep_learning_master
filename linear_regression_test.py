__author__ = 'du'

import numpy as np
from road_to_deep_learning.naive import SGDLinearRegression, BatchLinearRegression

np.random.seed(0)
W = [2, -1]
X = np.random.randn(10000, 2)
y = X.dot(W) + 1 + np.random.randn(len(X))

model1 = SGDLinearRegression(X.shape[1], tol=1e-7, batch_size=200, n_iter=1000000, lr=0.0000005, report=100).fit(X, y)
print(model1.W_)
print(model1.score(X, y))

model2 = BatchLinearRegression(X.shape[1], tol=1e-7, n_iter=1000000, method="L-BFGS-B").fit(X, y)
print(model2.W_)
print(model2.score(X, y))
