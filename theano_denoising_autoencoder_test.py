__author__ = 'du'

import numpy as np
from road_to_deep_learning.theano import LogisticRegression, DenoisingAutoEncoder
from sklearn.cross_validation import cross_val_score

X = np.random.randn(1000, 2)
y = np.array((X ** 2).sum(1) > 1.0, dtype=int)

n_dim = X.shape[1]
n_classes = len(set(y))

model = LogisticRegression(n_dim, n_classes, tol=1e-6, lr=0.01, n_iter=100000, report=0, C=1.0)
print(cross_val_score(model, X, y, cv=4, n_jobs=-1))

ae = DenoisingAutoEncoder(n_dim, 10, tol=1e-6, lr=0.01, n_iter=1000000, report=1000)
z = ae.transform(X)

print(z)

model = LogisticRegression(z.shape[1], n_classes, tol=1e-6, lr=0.01, n_iter=100000, report=0, C=1.0)
print(cross_val_score(model, z, y, cv=4, n_jobs=-1))
