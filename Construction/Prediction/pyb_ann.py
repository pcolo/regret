# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet

bdata = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_161.csv', delimiter=';', na_values=0, nrows=96*3,
                     index_col=0, parse_dates=True, infer_datetime_format=True)

# 0: weekday, 1: month, 2: time, 3: monthday

X = []

for k, v in bdata.iterrows():
    dow = k.dayofweek
    day = k.day
    mth = k.month
    sec = (k.hour * 3600 + k.minute * 60)/1000
    X.append([dow, day, mth, sec])

Y = bdata.values

#Xmin=np.asarray([x*1.0 for x in np.amin(X,axis=0)])
Xmax=np.asarray([x*1.0 for x in np.amax(X,axis=0)])
#X = (X-Xmin)/(Xmax-Xmin)
X=X/Xmax
Ymin=np.asarray([x*1.0 for x in np.amin(Y,axis=0)])
Ymax=np.asarray([x*1.0 for x in np.amax(Y,axis=0)])
Y = (Y-Ymin)/(Ymax-Ymin)

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
        return self.yhat

    def score(self, y):
        vect_se = (self.yhat - y)**2
        mse = float(np.sum(vect_se))/float(len(vect_se))
        return mse



nn = NeuralNetwork()
nn.fit(X,Y)
print type(nn.predict(X))
#print nn.score(Y)
print (Y[1] - 4)*4