
## ----------------------- Data ---------------------------- ##
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd

bdata = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_162.csv', delimiter=';', na_values=0, nrows=96*365*3,
                     index_col=0, parse_dates=True, infer_datetime_format=True)

# 0: weekday, 1: month, 2: time, 3: monthday

trainX = []

for k, v in bdata.iterrows():
    dow = k.dayofweek
    day = k.day
    mth = k.month
    sec = k.hour * 3600 + k.minute * 60
    trainX.append([dow, day, mth, sec])
trainX = np.array(trainX, dtype=float)
trainY = np.array(bdata.values, dtype=float)


bdata = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_162.csv', delimiter=';', skiprows=range(1, 96*365*3), na_values=0, nrows=96*365,
                     index_col=0, parse_dates=True, infer_datetime_format=True)

# 0: weekday, 1: month, 2: time, 3: monthday

testX = []

for k, v in bdata.iterrows():
    dow = k.dayofweek
    day = k.day
    mth = k.month
    sec = k.hour * 3600 + k.minute * 60
    testX.append([dow, day, mth, sec])
testX = np.array(testX, dtype=float)
testY = np.array(bdata.values, dtype=float)

#Normalize:
trainX = trainX/np.amax(trainX, axis=0)
trainY = trainY/100 #Max test score is 100

#Normalize:
testX = testX/np.amax(testX, axis=0)
testY = testY/100 #Max test score is 100

## ----------------------- Part 5 ---------------------------- ##

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 4
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        
        return dJdW1, dJdW2

    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad


NN = Neural_Network()
numgrad = computeNumericalGradient(NN, trainX, trainY)
print numgrad

grad = NN.computeGradients(trainX, trainY)
print grad
