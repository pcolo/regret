# -*- coding: utf-8 -*-

## ----------------------- Data ---------------------------- ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from pybrain.datasets import SupervisedDataSet
DS = SupervisedDataSet(4, 1)
for i in range(0, Y.size):
        DS.addSample((X[i][0], X[i][1], X[i][2], X[i][3]), (float(Y[i]),))

## ----------------------- ANN ---------------------------- ##

from pybrain.structure import RecurrentNetwork
n = RecurrentNetwork()

from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

n.addInputModule(SigmoidLayer(4, name='in'))
n.addModule(SigmoidLayer(3, name='hidden'))
n.addOutputModule(LinearLayer(1, name='out'))
n.addConnection(FullConnection(n['in'], n['hidden'], name='c1'))
n.addConnection(FullConnection(n['hidden'], n['out'], name='c2'))

n.sortModules() #initialisation


## ----------------------- Trainer ---------------------------- ##

from pybrain.supervised.trainers import BackpropTrainer

tstdata, trndata = DS.splitWithProportion(0.25)

# print len(tstdata)
# print len(trndata)

trainer = BackpropTrainer(n, DS, learningrate=0.1, momentum=0.5, weightdecay=0.0001)
trainer.trainUntilConvergence(verbose=True, maxEpochs=100)

# print trainer.trainUntilConvergence()
# trainer.trainOnDataset(trndata, 100)

#print n.activate((2, 1, 3, 0))
#print n.activate((2, 1, 3, 90))

## ----------------------- Results & Performance mesurements ---------------------------- ##

yhat = []
yhat = n.activateOnDataset(tstdata)

#print yhat
#print tstdata['target']

def vect_se(X,y):
    vect_se = []
    for i in range(len(X)):
        vect_se.append((float(y[i]) - float(X['target'][i]))**2)
    return vect_se


def mse(X,y):
    return float(sum(vect_se(X, y)))/float(len(vect_se(X, y)))

print mse(tstdata, yhat)

plt.plot(yhat, color='r')
plt.plot(tstdata['target'], color='b')
#plt.plot(vect_se(tstdata, yhat))
plt.show()




