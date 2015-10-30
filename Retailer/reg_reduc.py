# -*- coding: utf-8 -*-


import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

##### Version prédiction de la consommation individuelle de chaque agents.


class RegretRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, nup=0.01, r=np.arange(0, 1, 0.001)):
        self.nup = nup
        self.r = r

    def fit(self, y):
        self.p = np.array([[0.001]*1000])
        self.loss_e = np.array([[0.]*1000])
        self.loss_a = []
        for i, v in enumerate(y):
                self.loss_e = (v - self.r)**2
                self.p = self.p * np.exp(-self.nup * self.loss_e) / \
                                np.sum(self.p * np.exp(-self.nup * self.loss_e))
                self.loss_a.append((v - self.loss_e*self.p)**2)
        self.total_loss = np.sum(self.loss_a)
        return self

    def predict(self, X):
        u = np.random.uniform(0, 1, len(X))
        s = np.cumsum(self.p)
        self.y = []
        for i in u:
            self.y.append(float(np.abs(i - s).argmin())/1000)
        return np.array(self.y)

    def score(self, y):
        vect_se = (self.y - y)**2
        mse = float(np.sum(vect_se))/float(len(vect_se))
        self.total_loss = np.sum(self.loss_a)
        return mse, self.total_loss
