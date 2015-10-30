# -*- coding: utf-8 -*-

from sklearn.svm import SVR
import numpy as np
n_samples, n_features = 10, 5
np.random.seed(0)
y = range(10) # np.random.randn(n_samples)
#X = np.random.randn(n_samples, n_features)
#y = [
#    [1, 38],
#    [2, 59],
#    [3, 14],
#]
y = [1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
X = [
    [1, 24],
    [3, 48],
    [3, 63],
    [1, 12],
    [1, 27],
    [1, 31],
    [1, 18],
    [3, 50],
    [3, 73],
    [3, 82],
]

y = [i for i in range(1, 10)]
X = [[i] for i in range(1, 10)]
print y
print X

clf = SVR(kernel='linear')#, C=1.0, epsilon=0.2)
print clf.fit(X, y)
print clf.predict([[i] for i in range(10)])
print clf.set_params(kernel='rbf')
