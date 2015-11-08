# -*- coding: utf-8 -*-

##### Version prédiction de la consommation globale.

from Aggregator import nn_agg, reg_reduc
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


from sklearn import cross_validation

import timeit



## ----------------------- Data Dimension ---------------------------- ##


d_train_e = 5
d_train_a = d_train_e + 20
d_pred = d_train_a + 20

d_test = 0.
## ----------------------- Data Creation ---------------------------- ##

data = pd.read_hdf('/Users/Colo/Google Drive/Projects/regret/data/LD2011_2014.hdf', 'ElectricityLoadDiagrams20112014').resample('M', how='sum')


## ----------------------- Generator  ---------------------------- ##

r_price = 0.1433
a_price = np.arange(0.14, 0.15, 0.001) #range of possible prices 0.14 < p^{agg} < 0.15 = \bar(p)
beta = 0.4
iter = 10 #Number of iteration to estimate the probability of stabilising the coalition

# Omega, c and d are to be calibrated


def generator(X, a_price):

    cons = 370
    c = 0.01
    d = 0.6
    dates = pd.date_range(start='01/2011', freq= 'M', periods=d_pred)

    d_r = [] #demand of the retailer
    d_a = [] #demand of the aggregator
    C = [] #the aggregator's client list

    for a in a_price:

        c_r = range(1, cons+1) #coalition of the retailer
        c_a = [] #coalition of the aggregator

        b = np.array([np.random.uniform(c, d, cons)]*d_pred)
        s = 7*np.sum(np.random.normal(r_price - a, 2*(b)**2), axis=1) #s(i,m)

        for i, v in enumerate(s):
            if v < 0 and len(c_a) >= int(-1*v/beta): # if the size is sufficient, pick s(i,m) consumers from c_a and take them to c_r
                x = random.sample(c_a, int(-1*v/beta)) # picking int(v/beta) in c_a equivalent to picking  int(v) in beta*c_a. But easier to code
                c_r += x
                for j in x:
                    c_a.remove(j)
            elif v >= 0 and len(c_r) >= int(v/beta): # if the size is sufficient, pick s(i,m) consumers from c_r and take them to c_a
                y = random.sample(c_r, int(v/beta))
                c_a += y
                for j in y:
                    c_r.remove(j)


            d_r.append(sum(X.iloc[i+1, np.array(c_r)-1])) # compute de corresponding demand to c_r
            d_a.append(sum(X.iloc[i+1, np.array(c_a)-1])) # compute de corresponding demand to c_a

            C.append(c_a)

    out = pd.DataFrame(np.array(d_a).reshape(d_pred, len(a_price)), index=dates)

    out=  (out - out.min(0)) / zero(out.max(0)[0], out.min(0)[0]) #Normalized demand of the aggregator and the retailer
    return out, np.array(C)

def obj_margin(iter, C): #iter number of \sum\eta_i - \Pi_C plotted to calculate probability
#def obj_price(iter, alpha, c_a): #alpha \in [0,1] level of confidence and iter number of \sum\eta_i - \Pi_C plotted to calculate probability Alpha version
    p_plus = 1.48 #tbc
    p_minus = 1.46 #tbc
    p_f =  1.44#tbc
    c = 0.01
    d = 0.6

    D = [] # total eta core - covering profit
    for i in range(iter):

        b = np.array([np.random.uniform(c, d, len(C))]*(d_pred-d_train_a))
        s_mu_r = 7*np.sum(np.random.normal(0, len(C)*(b)**2), axis=1)

        sigma = np.array([[0.1]*len(C)]*(d_pred-d_train_a)) #tbc
        epsilon = np.random.normal(0, (sigma)**2)
        epsilon_plus = np.amax([epsilon, [[0]*len(C)]*len(epsilon)], axis=0)
        epsilon_minus = np.amax([-epsilon, [[0]*len(C)]*len(epsilon)], axis=0)

        D_plus = np.amax([np.sum(epsilon, axis=1), [0]*len(np.sum(epsilon, axis=1))], axis=0)
        D_minus = np.amax([- np.sum(epsilon, axis=1), [0]*len(np.sum(epsilon, axis=1))], axis=0)
        s_epsilon_plus = np.sum(epsilon_plus, axis=1)
        s_epsilon_minus = np.sum(epsilon_minus, axis=1)


        D.append(p_plus*s_epsilon_plus - p_minus*s_epsilon_minus +(r_price - s_mu_r - p_f - p_minus*(1 - 1.0/zero(len(C),0)))*D_minus - \
            (r_price - s_mu_r - p_f - p_plus*(1 - 1.0/zero(len(C),0))*D_plus))

    F = []
    for i, v in enumerate(D):
        F.append(np.sort(np.array(D)[i]))
        #E.append(np.cumsum(np.transpose(np.array(D))[i])[int(iter*alpha-1)]) Version with alpha level

    return F[-1], F # returns the vector of values of Pi_A such as p(Pi_A \geq \sum \eta_j - \Pi_C) = 1 p.s. for each period AND the entire distribution for each period

def zero(a, b):
    if a == b:
        return 1
    if a != b:
        return a - b


## ----------------------- Data split ---------------------------- ##

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    #X, y, test_size=d_test, random_state=0)

## ----------------------- Aggregator ---------------------------- ##

class RegretAggregetor(object):
    def __init__(self, nu=0.02, r=np.arange(0, 3)):
        self.nu = nu
        self.r = r
        self.nn_p = np.array(pred_nn_a)
        self.reg_p = np.array(pred_reg_a)
        self.svr_p = np.array(pred_svr_a)

    def fit(self, X, y): #fitting on experts predictions of optimal prices on every training period -> defining corresponding regret
        self.p = np.array([1.0/float(self.r.size)]*self.r.size)
        self.loss_a = []
        for i, v in enumerate(y):
            for k, w in enumerate(v):
                self.loss_e = np.array([(self.nn_p[i][k]-w)**2, (self.reg_p[i][k]-w)**2, (self.svr_p[i][k]-w)**2,])
                self.p = self.p * np.exp(-self.nu * self.loss_e) / np.sum(self.p * np.exp(-self.nu * self.loss_e))
                self.loss_a.append((w - self.loss_e*self.p)**2)
        self.total_loss = np.sum(self.loss_a)
        return self

    def predict(self): #prediction according to the fitted regret distribution
        self.y = self.p[0]*np.array(pred_nn_p) + self.p[1]*np.array(pred_reg_p) + self.p[2]*np.array(pred_svr_p)
        return self.y

    def price(self, X):
        a = np.array([self.y[i]*a_price[i] for i in range(len(a_price))]).reshape(d_pred - d_train_a, len(a_price)) #EXPECTED profit vector for a given period
        # on the entire range of prices
        self.m = [] #EXPECTED profit
        b = [] #index of the EXPECTED optimal price for a given period
        p = [] #EXPECTED optimal price for a given period
        c = np.array([np.array(X[i])*a_price[i] for i in range(len(a_price))]).reshape( d_pred - d_train_a, len(a_price)) #REAL profit vector for a given period
        # on the entire range of prices
        n = [] #REAL maximum profit on a given period
        coal = [] #Real coalition size
        o = [] #distance of real profit to stabilising distribution
        d = [] #probabilitiy of stabilising the coalition
        o_Pi_A = obj_margin(iter, generator(data, a_price)[1])[0] #Vector of objective \Pi_A for all periods
        o_Pi_D = obj_margin(iter, generator(data, a_price)[1])[1] #Vector of distribution of the probability for stability

        # Profit estimation and maximisation

        for i, v in enumerate(a):
            self.m.append(v[np.abs(v - obj_margin(iter, generator(data, a_price)[1][i])[0][i]).argmin()]) #Objective: profit as near as possible to proba = 1 p.s.
            for w, j in enumerate(v):
                if w == np.abs(v - obj_margin(iter, generator(data, a_price)[1][i])[0][i]).argmin(): #Here is infered the optimal price in order to fulfill the objective
                    b.append(w)
        print b
        for i in b:
            p.append(a_price[i])

        # Real profit computation and maximisation

        for i, v in enumerate(c):
            n.append(v[np.abs(v - obj_margin(iter, generator(data, a_price)[1][i])[0][i]).argmin()])

        self.d = np.array([np.array(X[v][i])*p[v] for i,v in enumerate(b)])

        #for i,v in enumerate(b):
            #coal.append(dyn[v].iat[i,0])

        # Probability of stabilising the coalition

        #for i, v in enumerate(self.d):
            #o.append(v[np.abs(v - o_Pi_D).argmin()])
            #for w, j in enumerate(v):
                #if w == np.abs(v -  o_Pi_D).argmin(): #Position in vector o, ie probability of stabilizing the coalition in base 1
                    # d.append(100*float(w)/len(iter))


        return "Coalition size:", coal, "Real profit:", self.d, "Total real profit:", sum(self.d), "Expected Profit:", self.m, "Expected optimal price:", p, \
               "Expected total profit:", sum(self.m), "Real maximum profit:", n, #"Real profit - Obj profit", self.d - o_Pi_A, "Real max profit - Obj profit", \
               #np.array(n) - o_Pi_A, "probability of stabilising the coalition at every period", d


    def expost(self, y):
        #prediction error
        self.total_loss = np.sum(self.loss_a)
        #profit error
        vect_se_p = np.array(self.d) - np.array(self.m)
        total_loss_p = sum(vect_se_p)
        return self.total_loss, self.p, vect_se_p, total_loss_p

## -----------------------  Fit & Prediction ---------------------------- ##

if __name__ == '__main__':

    pred_reg_a = []
    pred_svr_a = []
    pred_nn_a = []
    pred_reg_p = []
    pred_svr_p = []
    pred_nn_p = []
    target_a = []
    target_p = []
    dyn = []

    m_share = generator(data, a_price)[0] #market shares on all periods for a given price of the aggregator


## ----------------------- Data normalisation ---------------------------- ##

    X = []
    for i, v in m_share.iterrows():
        mth = i.month
        X.append([mth])
    Xmax = np.amax(X, axis=0)*1.0
    X = X/Xmax

    X_e = X[:d_train_e]
    X_a = X[d_train_e:d_train_a]
    X_p = X[d_train_a:d_pred]

    for j in range(len(a_price)):

        y = []
        for i in m_share[j]:
            y.append(i)

        y_e = y[:d_train_e]
        y_a = y[d_train_e:d_train_a]
        y_p = y[d_train_a:d_pred]

        target_a.append(y_a)
        target_p.append(y_p)



#### Neural Network ####

        start = timeit.default_timer()

        xpr_nn = nn_agg.NeuralNetwork()
        xpr_nn.fit(X_e, y_e)

        pred_nn_a.append(xpr_nn.predict(X_a))

        pred_nn_p.append(xpr_nn.predict(X_p))


        #xpr_bay = BayesianRidge()
        #xpr_bay.fit(X_e, y_e)

        #joblib.dump(xpr_nn0, 'pick/nn.pkl') #Saving parameters of the Neural Network
        #xpr_nn = joblib.load('pick/nn.pkl')  #Charge parameters of the Neural Network

        stop = timeit.default_timer()
        print "xpr_nn time:", stop - start

#### Regret ####

        start = timeit.default_timer()

        xpr_regret = reg_reduc.RegretRegressor()
        xpr_regret.fit(y_e)

        pred_reg_a.append(xpr_regret.predict(X_a))
        pred_reg_p.append(xpr_regret.predict(X_p))

        stop = timeit.default_timer()
        print "xpr_regret time", stop - start

#### SVR ####

        start = timeit.default_timer()

        xpr_svr = SVR(kernel='rbf', C=1.0, epsilon=0.2)
        xpr_svr.fit(X_e, y_e)

        pred_svr_a.append(xpr_svr.predict(X_a))
        pred_svr_p.append(xpr_svr.predict(X_p))



        #joblib.dump(xpr_svr0, 'pick_svr/svr.pkl') #Saving parameters of the SVR
        #xpr_svr = joblib.load('pick_svr/svr.pkl')

        stop = timeit.default_timer()
        print "xpr_svr time:", stop - start

#### Aggregator ####
    start = timeit.default_timer()

    agg = RegretAggregetor()
    agg.fit(X_a, target_a)
    agg.predict()
    print agg.price(target_p)

    stop = timeit.default_timer()
    print "agg time:", stop - start

#### Output ####

    a = agg.price(target_p)[-3]
    plt.plot(a)
#    plt.plot(y_t, 'b')
    plt.show()
#    print a

#### Performance ####

    print "mse nn:", mean_squared_error(target_a, pred_nn_a)
    print "mse reg:", mean_squared_error(target_a, pred_reg_a)
    print "mse svr:", mean_squared_error(target_a, pred_svr_a)
    print "mse agg:", mean_squared_error(target_a, agg.predict())
    print "weights:", agg.expost(target_p)[1]
    print "profit loss due to misetimation:", agg.expost(target_p)[2], "total profit loss:", agg.expost(target_p)[3]

    #A=[]
    #for i in np.arange(0.001, 0.1, 0.001):
        #b = RegretAggregetor(nu=i)
        #b.fit(X, y)
        #y = b.predict(X_t)
        #A.append(b.expost(y_t)[0])

    #print np.array(A).min()