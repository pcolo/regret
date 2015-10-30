# -*- coding: utf-8 -*-

##### Version prédiction de la consommation individuelle de chaque agents. Manque reshape avant le SVR

from nconsumer_algo import nn, reg
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cross_validation

import timeit



## ----------------------- Data Dimension ---------------------------- ##


d_train_e = 2
d_train_a = d_train_e + 2
d_pred = d_train_a + 1

d_test = 0.
## ----------------------- Data Creation ---------------------------- ##

data = pd.read_hdf('/Users/Colo/Google Drive/Projects/regret/data/LD2011_2014.hdf', 'ElectricityLoadDiagrams20112014').resample('M', how='sum')
data = 0.25 * (data - data.min(0)) / (data.max(0)- data.min(0))
data_e = pd.DataFrame(data)[0:d_train_e]
data_a = pd.DataFrame(data)[d_train_e:d_train_a]
data_p = pd.DataFrame(data)[d_train_a:d_pred]

categories = pd.read_csv('/Users/Colo/Google Drive/Projects/regret/data/activityInformation.csv', delimiter=';', index_col=0)
#print categories['Categorie'][1]
#print data_e.iloc[-1]

X_e = []
for i, v in enumerate(data_e):
    x = []
    for k, w in data_e.iterrows():
        mth = k.month
        cat = categories['Categorie'][i+1]
        x.append([cat, mth])
    X_e.append(x)
#print X_e[369][23][1]

y_e = data_e.values
#print y_e[1][369]


X_a = []
for i, v in enumerate(data_a):
    x = []
    for k, w in data_a.iterrows():
        mth = k.month
        cat = categories['Categorie'][i+1]
        x.append([cat, mth])
    X_a.append(x)
y_a = data_a.values

X_p = []
for i, v in enumerate(data_p):
    x = []
    for k, w in data_p.iterrows():
        mth = k.month
        cat = categories['Categorie'][i+1]
        x.append([cat, mth])
    X_p.append(x)
y_p = data_p.values

## ----------------------- Data normalisation ---------------------------- ##


Xmax_e = np.asarray([x*1.0 for x in np.amax(X_e, axis=0)])
X_e = X_e/Xmax_e
print X_e

Xmax_a = np.asarray([x*1.0 for x in np.amax(X_a, axis=0)])
X_a = X_a/Xmax_a


Xmax_p = np.asarray([x*1.0 for x in np.amax(X_p, axis=0)])
X_p = X_p/Xmax_p

## ----------------------- Data split ---------------------------- ##

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    #X, y, test_size=d_test, random_state=0)

## ----------------------- Aggregator ---------------------------- ##

class RegretAggregetor(object):
    def __init__(self, nu=0.0282871250702, r=np.arange(0, 3)):
        self.nu = nu
        self.r = r

    def fit(self, X, y):
        self.p = np.array([1.0/float(self.r.size)]*self.r.size)
        self.loss_a = []
        nn_p = xpr_nn.predict(X)
        reg_p = xpr_regret.predict(X)
        svr_p = xpr_svr.predict(X)
        print nn_p, reg_p, svr_p
        #self.loss_e = np.array([(nn_p-y)**2, (reg_p-y)**2, (svr_p-y)**2,])
        for i, v in enumerate(y):
            self.loss_e = np.array([(nn_p[i]-v)**2, (reg_p[i]-v)**2, (svr_p[i]-v)**2,])
            self.p = self.p * np.exp(-self.nu * self.loss_e) / np.sum(self.p * np.exp(-self.nu * self.loss_e))
            self.loss_a.append((v - self.loss_e*self.p)**2)
        self.total_loss = np.sum(self.loss_a)
        return self

    def predict(self, X):
        self.y = self.p[0]*xpr_nn.predict(X) + self.p[1]*xpr_regret.predict(X) + self.p[2]*xpr_svr.predict(X)
        return self.y

    def expost(self, y):
        vect_se = (self.y - y)**2
        mse = float(np.sum(vect_se))/float(len(vect_se))
        self.total_loss = np.sum(self.loss_a)
        return mse, self.total_loss, self.p

## -----------------------  Fit & Prediction ---------------------------- ##

if __name__ == '__main__':

#### Neural Network ####

    start = timeit.default_timer()

    xpr_nn = nn.NeuralNetwork()
    xpr_nn.fit(X_e, y_e)
    print xpr_nn.predict(X_a)
    #xpr_bay = BayesianRidge()
    #xpr_bay.fit(X_e, y_e)

    #joblib.dump(xpr_nn0, 'pick/nn.pkl') #Saving parameters of the Neural Network
    #xpr_nn = joblib.load('pick/nn.pkl')  #Charge parameters of the Neural Network

    stop = timeit.default_timer()
    print "xpr_nn time:", stop - start

#### Regret ####

    start = timeit.default_timer()

    xpr_regret = reg.RegretRegressor()
    xpr_regret.fit(y_e)
    print xpr_regret.predict(X_a)

    stop = timeit.default_timer()
    print "xpr_regret time", stop - start

#### SVR ####

    start = timeit.default_timer()

    xpr_svr = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    xpr_svr.fit(X_e, y_e)

    #joblib.dump(xpr_svr0, 'pick_svr/svr.pkl') #Saving parameters of the SVR
    #xpr_svr = joblib.load('pick_svr/svr.pkl')

    stop = timeit.default_timer()
    print "xpr_svr time:", stop - start

#### Aggregator ####

    start = timeit.default_timer()

    agg = RegretAggregetor()
    agg.fit(X_a, y_a)
    agg.predict(X_p)

    stop = timeit.default_timer()
    print "agg time:", stop - start

#### Output ####

#    a = agg.predict(X_t)
#    plt.plot(a, 'r')
#    plt.plot(y_t, 'b')
#    plt.show()
#    print a


    print "mse nn:", mean_squared_error(y_p, xpr_nn.predict(X_p))
    print "mse reg:", mean_squared_error(y_p, xpr_regret.predict(X_p))
    print "mse svr:", mean_squared_error(y_p, xpr_svr.predict(X_p))
    print "mse agg:", agg.expost(y_p)[0]
    print "weights:", agg.expost(y_p)[2]

    #A=[]
    #for i in np.arange(0.001, 0.1, 0.001):
        #b = RegretAggregetor(nu=i)
        #b.fit(X, y)
        #y = b.predict(X_t)
        #A.append(b.expost(y_t)[0])

    #print np.array(A).min()