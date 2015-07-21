__author__ = 'du'

import numpy as np
from road_to_deep_learning.utils import optimize
from road_to_deep_learning.naive import MLPClassifier, DenoisingAutoEncoder
from sklearn import datasets, linear_model, pipeline, cross_validation

np.random.seed(0)
data = datasets.load_digits()
X = data.data
y = data.target

n_dim = X.shape[1]
n_classes = len(set(y))
n_hidden = 100

ae = DenoisingAutoEncoder(n_dim, n_hidden, optimizer=optimize.scipy_minimize, noise_ratio=0.3)
clf = linear_model.LogisticRegression()

s = cross_validation.cross_val_score(clf, X, y, cv=10)
print(s)

ae.fit(X)
z = ae.transform(X)
s = cross_validation.cross_val_score(clf, z, y, cv=10)
print(s)
