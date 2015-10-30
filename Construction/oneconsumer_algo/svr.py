from sklearn.svm import SVR

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

class SVRRegressor(BaseEstimator, RegressorMixin):

    cf = SVR(kernel='rbf', C=1.0, epsilon=0.2)

    def fit(self, X, y):
        cf.fit(X, y)
        return self


    def score(self, y):
        vect_se = (self.yhat - y)**2
        mse = float(np.sum(vect_se))/float(len(vect_se))
        return mse