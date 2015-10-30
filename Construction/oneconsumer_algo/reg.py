import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

class RegretRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, nup=0.01, r=np.arange(0, 1, 0.001)):
        self.nup = nup
        self.r = r

    def fit(self, X, y):
        self.p = np.array([[0.001]*1000]*12)
        self.loss_e = np.array([[0.]*1000]*12)
        self.loss_a = []
        for i, v in enumerate(y):
            self.loss_e[X[i][2]] = (v - self.r)**2
            self.p[X[i][2]] = self.p[X[i][2]] * np.exp(-self.nup * self.loss_e[X[i][2]]) / \
                                np.sum(self.p[X[i][2]] * np.exp(-self.nup * self.loss_e[X[i][2]]))
            self.loss_a.append((v - self.loss_e[X[i][2]]*self.p[X[i][2]])**2)
        self.total_loss = np.sum(self.loss_a)
        return self

    def predict(self, X):
        u = np.random.uniform(0, 1, len(X))
        s = np.array([np.cumsum(self.p[i]) for i in range(11)])
        self.y = []
        for i in u:
            self.y.append(float(np.abs(i - s[X[i][2]]).argmin())/1000)
        return np.array(self.y)

    def score(self, y):
        vect_se = (self.y - y)**2
        mse = float(np.sum(vect_se))/float(len(vect_se))
        self.total_loss = np.sum(self.loss_a)
        return mse, self.total_loss
