# -*- coding: utf-8 -*-

from regret import nn, reg
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


d_train_e = 96 #96*365*2
d_train_a = d_train_e + 96*365
d_pred = d_train_a + 96*30

d_test = 0.
## ----------------------- Data Creation ---------------------------- ##


data = pd.read_hdf('/Users/Colo/Google Drive/Projects/regret/data/LD2011_2014.hdf', 'ElectricityLoadDiagrams20112014')
data_e = pd.DataFrame(data.MT_161)[0:d_train_e]
data_a = pd.DataFrame(data.MT_161)[d_train_e:d_train_a]
data_p = pd.DataFrame(data.MT_161)[d_train_a:d_pred]
# 0: weekday, 1: month, 2: time, 3: monthday

X_e = []
for k, v in data_e.iterrows():
    dow = k.dayofweek
    day = k.day
    mth = k.month
    sec = float(k.hour * 3600 + k.minute * 60)/1000
    X_e.append([dow, day, mth, sec])
y_e = data_e.values*0.25

X_a = []
for k, v in data_a.iterrows():
    dow = k.dayofweek
    day = k.day
    mth = k.month
    sec = float(k.hour * 3600 + k.minute * 60)/1000
    X_a.append([dow, day, mth, sec])
y_a = data_a.values*0.25

X_p = []
for k, v in data_p.iterrows():
    dow = k.dayofweek
    day = k.day
    mth = k.month
    sec = float(k.hour * 3600 + k.minute * 60)/1000
    X_p.append([dow, day, mth, sec])
y_p = data_p.values*0.25



## ----------------------- Data normalisation ---------------------------- ##


Xmax_e = np.asarray([x*1.0 for x in np.amax(X_e, axis=0)])
X_e = X_e/Xmax_e

ymin_e = np.asarray([x*1.0 for x in np.amin(y_e, axis=0)])
ymax_e = np.asarray([x*1.0 for x in np.amax(y_e, axis=0)])
y_e = (y_e-ymin_e)/(ymax_e-ymin_e)
y_e = np.reshape(y_e, y_e.size)


Xmax_a = np.asarray([x*1.0 for x in np.amax(X_a, axis=0)])
X_a = X_a/Xmax_a

ymin_a = np.asarray([x*1.0 for x in np.amin(y_a, axis=0)])
ymax_a = np.asarray([x*1.0 for x in np.amax(y_a, axis=0)])
y_a = (y_a-ymin_a)/(ymax_a-ymin_a)
y_a = np.reshape(y_a, y_a.size)


Xmax_p = np.asarray([x*1.0 for x in np.amax(X_p, axis=0)])
X_p = X_p/Xmax_p

ymin_p = np.asarray([x*1.0 for x in np.amin(y_p, axis=0)])
ymax_p = np.asarray([x*1.0 for x in np.amax(y_p, axis=0)])
y_p = (y_p-ymin_p)/(ymax_p-ymin_p)
y_p = np.reshape(y_p, y_p.size)

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

    #xpr_nn0 = nn.NeuralNetwork(split_prop=d_test)
    #xpr_nn0.fit(X_e, y_e)
    #xpr_bay = BayesianRidge()
    #xpr_bay.fit(X_e, y_e)

    #joblib.dump(xpr_nn0, 'pick/nn.pkl') #Saving parameters of the Neural Network
    xpr_nn = joblib.load('pick/nn.pkl')  #Charge parameters of the Neural Network

    stop = timeit.default_timer()
    print "xpr_nn time:", stop - start

#### Regret ####

    start = timeit.default_timer()

    xpr_regret = reg.RegretRegressor()
    xpr_regret.fit(X_e, y_e)

    #joblib.dump(xpr_regret0, 'pick_regret/regret.pkl') #Saving parameters of the Regret
    #xpr_regret = joblib.load('pick_regret/regret.pkl')  #Charge parameters of the Regret

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

    #e_prediction = pd.DataFrame(data = agg.predict(X_p))
    #print e_prediction