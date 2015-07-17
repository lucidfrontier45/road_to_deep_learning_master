__author__ = 'du'

import numpy as np
from road_to_deep_learning.utils import optimize
from road_to_deep_learning.naive import LinearRegression

np.random.seed(0)
W = [2, -1]
X = np.random.randn(10000, 2)
y = X.dot(W) + 1 + np.random.randn(len(X))

optimizer = optimize.scipy_minimize
model = LinearRegression(X.shape[1], tol=1e-7, batch_size=200, n_iter=1000000, optimizer=optimizer,
                         method="L-BFGS-B").fit(X, y)
print(model.W_)
print(model.score(X, y))
