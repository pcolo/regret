# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.svm import SVR
from sklearn import cross_validation

from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet

## ----------------------- Data ---------------------------- ##

d_base = 96*30
d_test = 0.25

d_skip = 365*96-1
d_pred = 96*2

bdata = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_161.csv', delimiter=';', na_values=0,
                      nrows=d_base, index_col=0, parse_dates=True, infer_datetime_format=True)

tdata = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_161.csv', delimiter=';', na_values=0,
                    skiprows=range(1, d_skip), nrows=d_pred, index_col=0, parse_dates=True, infer_datetime_format=True)

# 0: weekday, 1: month, 2: time, 3: monthday

X = []
for k, v in bdata.iterrows():
    dow = k.dayofweek
    day = k.day
    mth = k.month
    sec = (k.hour * 3600 + k.minute * 60)/1000
    X.append([dow, day, mth, sec])

y = bdata.values*0.25

X_t = []
for k, v in tdata.iterrows():
    dow = k.dayofweek
    day = k.day
    mth = k.month
    sec = (k.hour * 3600 + k.minute * 60)/1000
    X_t.append([dow, day, mth, sec])

y_t = tdata.values*0.25


## ----------------------- Data normalisation ---------------------------- ##


Xmax = np.asarray([x*1.0 for x in np.amax(X, axis=0)])
X = X/Xmax

ymin = np.asarray([x*1.0 for x in np.amin(y, axis=0)])
ymax = np.asarray([x*1.0 for x in np.amax(y, axis=0)])
y = (y-ymin)/(ymax-ymin)
y = np.reshape(y, y.size)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=d_test, random_state=0)


Xmax_t = np.asarray([x*1.0 for x in np.amax(X_t, axis=0)])
X_t = X_t/Xmax_t

ymin_t = np.asarray([x*1.0 for x in np.amin(y_t, axis=0)])
ymax_t = np.asarray([x*1.0 for x in np.amax(y_t, axis=0)])
y_t = (y_t-ymin_t)/(ymax_t-ymin_t)
y_t = np.reshape(y_t, y_t.size)

## ----------------------- SVR results & performance ---------------------------- ##


class NeuralNetwork(BaseEstimator, RegressorMixin):

    def __init__(self, inp_neu=4, hid_neu=3, out_neu=1, learn_rate=0.1, nomentum=0.5, weight_dec=0.0001, epochs=100,
                 split_prop=0.25):
        self.inp_neu = inp_neu
        self.hid_neu = hid_neu
        self.out_neu = out_neu
        self.learn_rate = learn_rate
        self.nomentum = nomentum
        self.weight_dec = weight_dec
        self.epochs = epochs
        self.split_prop = split_prop

    def data(self, X, y=None):
        DS = SupervisedDataSet(self.inp_neu, self.out_neu)
        for i in range(0, len(X)):
            DS.addSample((X[i][0], X[i][1], X[i][2], X[i][3]), y[i]) #ATTENTION pas optimisé pour toutes les tailles
        return DS

    def fit(self, X, y):
        self.n = FeedForwardNetwork()

        self.n.addInputModule(SigmoidLayer(self.inp_neu, name='in'))
        self.n.addModule(SigmoidLayer(self.hid_neu, name='hidden'))
        self.n.addOutputModule(LinearLayer(self.out_neu, name='out'))
        self.n.addConnection(FullConnection(self.n['in'], self.n['hidden'], name='c1'))
        self.n.addConnection(FullConnection(self.n['hidden'], self.n['out'], name='c2'))

        self.n.sortModules() #initialisation

        self.tstdata, trndata = self.data(X,y).splitWithProportion(self.split_prop)

        trainer = BackpropTrainer(self.n, trndata, learningrate=self.learn_rate, momentum=self.nomentum, weightdecay=self.weight_dec)
        trainer.trainUntilConvergence(verbose=True, maxEpochs=self.epochs)

        return self

    def predict(self, X):
        self.yhat = []
        for i in X:
            self.yhat.append(float(self.n.activate(i)))
        return np.array(self.yhat)

    def score(self, y):
        vect_se = (self.yhat - y)**2
        mse = float(np.sum(vect_se))/float(len(vect_se))
        return mse


def expert_elm(demand, feedback):
    forecast_demand_ELM = []
    optimum_price_ELM = []
    return forecast_demand_ELM, optimum_price_ELM

class RegretRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, nup=0.01, r=np.arange(0, 1, 0.001)):
        self.nup = nup
        self.r = r

    def fit(self, X, y):
        self.p = np.array([0.001]*1000)
        self.loss_e = np.array([0.]*1000)
        self.loss_a = []
        for i in y:
            self.loss_e = (i - self.r)**2
            self.p = self.p * np.exp(-self.nup * self.loss_e) / np.sum(self.p * np.exp(-self.nup * self.loss_e))
            self.loss_a.append((i - self.loss_e*self.p)**2)
        self.total_loss = np.sum(self.loss_a)
        return self

    def predict(self, X):
        u = np.random.uniform(0, 1, len(X))
        s = np.cumsum(self.p)
        self.y = []
        for i in u:
            self.y.append(float(np.abs(i - s).argmin())/1000)
        return np.array(self.y)

    def perf(self, X):
        vect_se = (self.y - X)**2
        mse = float(np.sum(vect_se))/float(len(vect_se))
        self.total_loss = np.sum(self.loss_a)
        return mse, self.total_loss

nn = NeuralNetwork(split_prop=d_test)
nn.fit(X, y)

regret = RegretRegressor()
regret.fit(X_train, y_train)

clf = SVR(kernel='rbf', C=1.0, epsilon=0.2)
clf.fit(X_train, y_train)


class RegretAggregetor(object):
    def __init__(self, nu=0.01, r=np.arange(0, 3)):
        self.nu = nu
        self.r = r

    def fit(self, X, y):
        self.p = np.array([1.0/float(self.r.size)]*self.r.size)
        self.loss_a = []
        for i in y:
            self.loss_e = np.array([(nn.predict(X)[i] - i)**2, (regret.predict(X)[i] - i)**2, (clf.predict(X)[i] - i)**2])
            self.p = self.p * np.exp(-self.nu * self.loss_e) / np.sum(self.p * np.exp(-self.nu * self.loss_e))
            self.loss_a.append((i - self.loss_e*self.p)**2)
        self.total_loss = np.sum(self.loss_a)
        return self

    def predict(self, X):
        self.y = self.p[0]*nn.predict(X) + self.p[1]*regret.predict(X) + self.p[2]*clf.predict(X)
        return self.y

    def expost(self, y):
        vect_se = (self.y - y)**2
        mse = float(np.sum(vect_se))/float(len(vect_se))
        self.total_loss = np.sum(self.loss_a)
        return mse, self.total_loss, self.p


def get_consumption():
    data = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_161.csv', delimiter=';', na_values=0, nrows=96*3,
                       index_col=0, parse_dates=True, infer_datetime_format=True, header=1)

    for index in data.index:
        consumption = data.loc[index] * 0.25
        consumption = consumption[~np.isnan(consumption)]
        #epsilon = np.random.uniform(-2.0, 2.0)
        yield index, consumption #+ (epsilon)



agg = RegretAggregetor()
agg.fit(X_train, y_train)

a = agg.predict(X_t)


plt.plot(a, 'r')
plt.plot(y_t, 'b')
plt.show()
print agg.expost(y_t)[0]
print agg.expost(y_t)[1]
print agg.expost(y_t)[2]

if __name__ == '__main__':
    for step, demand in enumerate(get_consumption()):
        #print demand[0], demand[1], len(demand[1])
        # Display header
        print "Run: {:03d}".format(step)




        #print demand[1]
        #print agg.predict(demand[1])


        #forecast_demand_SVR = expert_svr(X, y, Z)
        # forecast_demand_NN, optimum_price_NN = expert_nn(demand[1], feedback)
        # forecast_demand_ELM, optimum_price_ELM = expert_elm(demand[1], feedback)
        # forecast_demand_R, optimum_price_R = expert_r(demand[1], feedback)
        #aggregator_price = aggregator(forecast_demand_SVR, loss_e)
        #loss_e, loss_a, feedback = expost(aggregator_price, nu_a, nu_b)
        #total_loss += loss_a



        # Condition de sortie
        if step == 10:
            break

