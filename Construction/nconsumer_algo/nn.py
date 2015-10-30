# -*- coding: utf-8 -*-
##### Version prédiction de la consommation individuelle de chaque agents.
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet

class NeuralNetwork(BaseEstimator, RegressorMixin):

    def __init__(self, inp_neu=2, hid_neu=3, out_neu=1, learn_rate=0.1, nomentum=0.5, weight_dec=0.0001, epochs=100,
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
            for j in range(len(X[i])):
                DS.addSample((X[i][j][0], X[i][j][1]), y[j][i]) #ATTENTION pas optimisé pour toutes les tailles
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
        for i in range(0, len(X)):
            for j in range(len(X[i])):
                self.yhat.append(float(self.n.activate(j)))
        self.yhat = np.array(self.yhat)
        return self.yhat

    def score(self, y):
        vect_se = (self.yhat - y)**2
        mse = float(np.sum(vect_se))/float(len(vect_se))
        return mse

