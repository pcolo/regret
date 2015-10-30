import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin

d_size_train = 96*365
d_size_test = 96*31
d_size = d_size_train + d_size_test
d_skip = 96*365

trs = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_161.csv', delimiter=';', na_values=0,
                    nrows=d_size_train, index_col=0, parse_dates=True, infer_datetime_format=True)

tes = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_161.csv', delimiter=';', na_values=0,
                    skiprows=range(1, d_skip), nrows=d_size_test, index_col=0, parse_dates=True, infer_datetime_format=True)

bdata = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_161.csv', delimiter=';', na_values=0, nrows=d_size,
                    index_col=0, parse_dates=True, infer_datetime_format=True)

# 0: weekday, 1: month, 2: time, 3: monthday
X_1 = []
for k, v in trs.iterrows():
    dow = k.dayofweek
    day = k.day
    mth = k.month
    sec = (k.hour * 3600 + k.minute * 60)/1000
    X_1.append([dow, day, mth, sec])

y_1 = trs.MT_161.values*0.25

Z_1 = []
for k, v in tes.iterrows():
    dow = k.dayofweek
    day = k.day
    mth = k.month
    sec = (k.hour * 3600 + k.minute * 60)/1000
    Z_1.append([dow, day, mth, sec])

w_1 = tes.MT_161.values*0.25



Xmax_1 = np.asarray([x*1.0 for x in np.amax(X_1, axis=0)])
X_1 = X_1/Xmax_1

ymin_1 = np.asarray(np.amin(y_1, axis=0))
ymax_1 = np.asarray(np.amax(y_1, axis=0))
y_1 = (y_1-ymin_1)/(ymax_1-ymin_1)

Zmax_1 = np.asarray([x*1.0 for x in np.amax(Z_1, axis=0)])
Z_1 = Z_1/Zmax_1

wmin_1 = np.asarray(np.amin(w_1, axis=0))
wmax_1 = np.asarray(np.amax(w_1, axis=0))
w_1 = (w_1-wmin_1)/(wmax_1-wmin_1)


class RegretRegressor(BaseEstimator, RegressorMixin):
    """
    RegretRegressor is a regressor that makes predictions using
    simple rules.
    This regressor is useful as a simple baseline to compare with other
    (real) regressors. Do not use it for real problems.
    Parameters
    ----------
    strategy : str
        Strategy to use to generate predictions.
        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by
          the user.
    constant : int or float or array of shape = [n_outputs]
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.
    quantile : float in [0.0, 1.0]
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.
    Attributes
    ----------
    constant_ : float or array of shape [n_outputs]
        Mean or median or quantile of the training targets or constant value
        given by the user.
    n_outputs_ : int,
        Number of outputs.
    outputs_2d_ : bool,
        True if the output at fit is 2d, else false.
    """

    def __init__(self, nup = 0.1, r = np.arange(0, 1, 0.001)):
        self.nup = nup
        self.r = r



    def fit(self, X, y):
        """Fit the random regressor.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target values.
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        self : object
            Returns self.
        """
        self.p = np.array([0.001]*1000)
        self.loss_e = np.array([0.]*1000)
        self.loss_a = [] #np.zeros(y.size) #np.array([0.]*1000)
        for i in y:
            self.loss_e = (i - self.r)**2
            self.p = self.p * np.exp(-self.nup * self.loss_e) / np.sum(self.p * np.exp(-self.nup * self.loss_e))
            self.loss_a.append((i - self.loss_e*self.p)**2)

        return self

    def predict(self, X):
        """
        Perform classification on test vectors X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        y : array, shape = [n_samples]  or [n_samples, n_outputs]
            Predicted target values for X.
        """
        u = np.random.uniform(0, 1, len(X))
        s = np.cumsum(self.p)
        self.y = []
        for i in u:
            self.y.append(float(np.abs(i - s).argmin())/1000)
        return self.y

    def perf(self, X):
        vect_se = (self.y - X)**2
        mse = float(np.sum(vect_se))/float(len(vect_se))
        self.total_loss = np.sum(self.loss_a)
        return mse, self.total_loss

if __name__ == '__main__':

    regret = RegretRegressor(nup= 0.0001)
    regret.fit(Z_1, y_1)
    print type(regret.predict(w_1))
    #print regret.perf(w_1)
    #plt.plot(np.arange(0,1,0.001), regret.p)
    #plt.plot(y, 'b')
    #plt.plot(w_1, 'r')
    #plt.show()

    #A=[]
    #for i in np.arange(0.0001, 0.01, 0.0001):
        #regret = RegretRegressor(nup= i)
        #regret.fit(Z_1, y_1)
        #y = regret.predict(w_1)
        #A.append(regret.perf(w_1)[0])

    #plt.plot(np.arange(0.0001, 0.01, 0.0001), A)
    #plt.show()


