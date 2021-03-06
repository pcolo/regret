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
d_train_a = d_train_e + 19
d_pred = d_train_a + 24

d_test = 0.
## ----------------------- Data Creation ---------------------------- ##

data = pd.read_hdf('/Users/Colo/Google Drive/Projects/regret/data/LD2011_2014.hdf', 'ElectricityLoadDiagrams20112014').resample('M', how='sum')


## ----------------------- Generator  ---------------------------- ##

r_price = 0.1433
a_price = np.arange(0.14, 0.15, 0.001) #range of possible prices 0.14 < p^{agg} < 0.15 = \bar(p)
beta = 0.9
cons = 370
iter = 100

# Omega, c and d are to be calibrated


def generator(X, a):
    """
    :param X: entry data
    :param a: chosen price by the company
    :return:
    """
    c = 0.01
    d = 0.6
    dates = pd.date_range(start='01/2011', freq= 'M', periods=len(a))

    b = np.array([np.random.uniform(c, d, cons)])

    s = []
    for i in a:
        s.append(np.sum(np.random.normal(r_price - i, 2*(b)**2), axis=1)) #\sum_{i} s(i,m) ie number of changing players

    c_r = range(1, cons+1) #construction variable
    c_a = [] #construction variable
    d_r = [] #demand of the retailer
    d_a = [] #demand of the aggregator
    e_r = [] #list of retailer's clients at every m
    e_a = [] #list of aggregator's clients at every m

    for i, v in enumerate(s):
        if v < 0 and len(c_a) >= int(-1*v*beta): # if the size is sufficient, pick s(i,m) consumers from c_a and take them to c_r
            x = random.sample(c_a, int(-1*v*beta)) # picking int(v/beta) in c_a equivalent to picking  int(v) in beta*c_a. But easier to code
            c_r += x
            for j in x:
                c_a.remove(j)
        elif v >= 0 and len(c_r) >= int(v*beta): # if the size is sufficient, pick s(i,m) consumers from c_r and take them to c_a
            y = random.sample(c_r, int(v*beta))
            c_a += y
            for j in y:
                c_r.remove(j)

        e_r.append(tuple(c_r)) #immutables are necessary here. Lists are mutable types !
        e_a.append(tuple(c_a))

        d_r.append(sum(X.iloc[i+1, np.array(c_r)-1])) # compute de corresponding demand to c_r
        d_a.append(sum(X.iloc[i+1, np.array(c_a)-1])) # compute de corresponding demand to c_a

    out = pd.DataFrame(np.array(d_a).reshape(len(a), 1),  index=dates,  columns=['Aggregator']) #demand of the aggregator
    tou = pd.DataFrame(np.array(d_r).reshape(len(a), 1),  index=dates,  columns=['Retailer']) #demand of the retailer

    outt = (out - out.min(0)[0]) / zero(out.max(0)[0], out.min(0)[0]) #Normalized demand of the aggregator

    return outt, e_a, out.min(0)[0], out.max(0)[0] - out.min(0)[0], tou, e_r,  tou.max(0)[0] - tou.min(0)[0]


def eta(e_a, p, iter):
    p_plus = 1.48 #tbc
    p_minus = 1.46 #tbc
    p_f =  1.44#tbc

    c = 0.01
    d = 0.6

    D = [] # total eta core - covering profit

    b = [np.random.uniform(c, d, cons)]*iter

    for w, k in enumerate(np.array(b)):

        epsilon = []
        epsilon_plus = []
        epsilon_minus = []
        s_mu_r = []

        for i in p:
            s_mu_r.append(np.sum(np.random.normal(i - r_price, 2*(k)**2))) #\sum_{i} s(i,m) ie number of changing players

        for i, v in enumerate(e_a):
            norm = np.random.normal(0, [(0.05)**2]*len(v))
            epsilon.append(norm)

            epsilon_plus.append(np.amax([norm, [0]*len(v)], axis=0))
            epsilon_minus.append(np.amax([-norm, [0]*len(v)], axis=0))

        D_plus = np.amax([sum_variable(epsilon), [0]*len(sum_variable(epsilon))], axis=0)
        D_minus = np.amax([- sum_variable(epsilon), [0]*len(sum_variable(epsilon))], axis=0)
        s_epsilon_plus = sum_variable(epsilon_plus)
        s_epsilon_minus = sum_variable(epsilon_minus)


        D.append(p_plus*s_epsilon_plus - p_minus*s_epsilon_minus +(r_price - np.array(s_mu_r) - p_f - p_minus*(1 - 1.0/zero(len(e_a),0)))*D_minus - \
            (r_price - np.array(s_mu_r) - p_f - p_plus*(1 - 1.0/zero(len(e_a),0))*D_plus)) # total eta core - covering profit

    E = []
    for i in range(len(p)):
        for j in range(iter):
            E.append(D[j][i])

    F = np.array(E).reshape(len(p), iter)

    G = []  #Distribution of "total eta core - covering profit" for each period
    for i, v in enumerate(F):
        G.append(np.sort(F)[i])

    H = [] # H: values of Pi_A such as p(Pi_A \geq \sum \eta_j - \Pi_C) = 1 p.s.
    for i, v in enumerate(G):
        H.append(v[-1])

    return np.array(H), G

def covering(e_a):
    p_plus = 1.48 #tbc
    p_minus = 1.46 #tbc

    epsilon = []
    epsilon_plus = []
    epsilon_minus = []
    for i, v in enumerate(e_a):
        norm = np.random.normal(0, [(0.05)**2]*len(v))
        epsilon.append(norm)

        epsilon_plus.append(np.amax([norm, [0]*len(v)], axis=0))
        epsilon_minus.append(np.amax([-norm, [0]*len(v)], axis=0))

    D_plus = np.amax([sum_variable(epsilon), [0]*len(sum_variable(epsilon))], axis=0)
    D_minus = np.amax([- sum_variable(epsilon), [0]*len(sum_variable(epsilon))], axis=0)
    s_epsilon_plus = sum_variable(epsilon_plus)
    s_epsilon_minus = sum_variable(epsilon_minus)


    D = (p_minus*(s_epsilon_minus - D_minus/zero(len(e_a),0)) - p_plus*(s_epsilon_plus - D_plus/zero(len(e_a),0)))

    return D  #returns the covering profit (including the cost of the coalition)

### Annexe functions for construction

def zero(a, b):
    if a == b:
        return 1
    if a != b:
        return a - b

def sum_variable(x):
    sum_x = []
    for i in x:
        sum_x.append(sum(i))
    return np.array(sum_x)


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
        #self.loss_e = np.array([(nn_p-y)**2, (reg_p-y)**2, (svr_p-y)**2,])
        for i, v in enumerate(y):
            for k, w in enumerate(v):
                self.loss_e = np.array([(self.nn_p[i][k]-w)**2, (self.reg_p[i][k]-w)**2, (self.svr_p[i][k]-w)**2,])
                print self.loss_e.shape
                self.p = self.p * np.exp(-self.nu * self.loss_e) / np.sum(self.p * np.exp(-self.nu * self.loss_e))
                self.loss_a.append((w - self.loss_e*self.p)**2)
        self.total_loss = np.sum(self.loss_a)
        return self

    def predict(self): #prediction according to the fitted regret distribution
        self.y = self.p[0]*np.array(pred_nn_p) + self.p[1]*np.array(pred_reg_p) + self.p[2]*np.array(pred_svr_p)
        return self.y

    def price(self):
        self.m = [] #EXPECTED profit
        self.b = [] #index of the EXPECTED optimal price for a given period
        self.p = [] #EXPECTED optimal price for a given period
        a = []
        c = []

        for i, v in enumerate(a_price):
            a.append(self.y[i]*v*generator(data, [v]*(d_pred))[3])
            b = eta(generator(data, [v]*(d_pred))[1], [v]*(d_pred), iter)[0]*generator(data, [v]*(d_pred))[3]
            c.append(b[d_train_a:])

        A = np.array([a]).reshape(d_pred - d_train_a, len(a_price))
        C = np.array([c]).reshape(d_pred - d_train_a, len(a_price))

        for i in range(d_pred - d_train_a):
            self.b.append(np.abs(A[i] - C[i]).argmin())

        for i, v in enumerate(self.b):
            self.p.append(a_price[v])
            self.m.append(A[i][v])

        return self.m, self.p

    def profit(self, X):
        n = [] #REAL maximum profit on a given period
        c = np.array([np.array(X[i])*v*generator(data, [v]*(d_pred))[3] for i,v in enumerate(a_price)]).reshape( d_pred - d_train_a, len(a_price)) #REAL profit vector for a given period
        # on the entire range of prices

        for i, v in enumerate(c):
            n.append(np.amax(v))

        #REAL profit vector for a given period
        self.d = np.array([np.array(X[v][i])*a_price[v]*generator(data, self.p)[3] for i,v in enumerate(self.b)])

        return self.d, n

    def coal_size(self):

        coal = []
        for i in generator(data, self.p)[1]:
            coal.append(float(len(i))/float(cons))

        return coal

    def expost(self):

        self.total_loss = np.sum(self.loss_a)  #prediction error
        vect_se_p = np.array(self.d) - np.array(self.m) #profit error
        total_loss_p = sum(vect_se_p)

        return self.total_loss, self.p, vect_se_p, total_loss_p

    def stab_proba(self):
        f = []
        e = np.array(eta(generator(data, self.p)[1], self.p, iter)[1])*generator(data, self.p)[3]

        for i in range(d_pred - d_train_a):
            f.append(float(np.abs(np.array(e)[i] - self.m[i]).argmin()))

        return np.array(f)/iter


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


    for a in a_price:


        ## ----------------------- Data normalisation ---------------------------- ##

        m_share = generator(data, [a]*d_pred)[0] #market shares on all periods for a given price of the aggregator

        X = []
        for i, v in m_share.iterrows():
            mth = i.month
            X.append([mth])
        Xmax = np.amax(X, axis=0)*1.0
        X = X/Xmax

        y = []
        for i in m_share['Aggregator']:
            y.append(i)

        X_e = X[:d_train_e]
        X_a = X[d_train_e:d_train_a]
        X_p = X[d_train_a:d_pred]

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
    agg.price()
    agg.profit(target_p)
    agg.coal_size()
    agg.stab_proba()
    #agg.c_profit(data)

#### Retailer ####

    profit_r = [] #Activity profit of the retailer
    for i in generator(data, [r_price]*d_pred)[4]['Retailer']:
        profit_r.append(i)

    #Covering profit of the retailer
    f_cov_ret = np.array(covering(generator(data, [r_price]*d_pred)[5])*generator(data, [r_price]*d_pred)[6])[d_train_a-1:d_pred-1]

    #Total profit of the retailer
    profit_ret = np.array(profit_r[d_train_a-1:d_pred-1])*r_price #+ np.array(f_cov_ret)[d_train_a:]


    stop = timeit.default_timer()
    print "agg time:", stop - start

#### Output ####

    a = agg.profit(target_p)[0] #Real activity profit
    b = a - agg.price()[0] #Difference between real and potential activity profit
    c = agg.coal_size() #Coalition size
    d = agg.stab_proba() #Probability of stabilizing the coalition
    e = agg.price()[1] #Chosen price by the aggregator

    x = np.arange(0, 24, 1)

    plt.figure(1)
    plt.subplot(221)
    plt.bar(x, a, color='b', )
    plt.xticks(np.linspace(0, 23, 24,endpoint=True))
    plt.title('Real profit')
    plt.subplot(222)
    plt.bar(x, d, color='b')
    plt.xticks(np.linspace(0, 23, 24, endpoint=True))
    plt.title('Probability of stabilizing the coalition')
    #plt.axis([0, 24, 0, 0.15])
    plt.subplot(223)
    plt.bar(x, c, color='b')
    plt.xticks(np.linspace(0, 23, 24,endpoint=True))
    plt.xlabel('Months')
    plt.ylabel('Market share')
    plt.title('Coalition size')
    plt.subplot(224)
    plt.bar(x, e, color='b')
    plt.xticks(np.linspace(0, 23, 24,endpoint=True))
    plt.xlabel('Months')
    plt.ylabel('Euros per kWh')
    plt.title('Estimated optimal price')
    plt.ylim(ymax = 0.15, ymin=0.14)
    plt.show()




    #plt.plot(x, a, color='b') #Agregator total profit
    #plt.plot(x, profit_ret, color='r') #Retailer total profit

    #plt.bar(x,c)

    #plt.bar(x, a + d, color='r', label='Aggregator profit')
    #plt.bar(x, profit_ret + f_cov_ret, color='b', alpha=0.4, label='Retailer profit')
    #plt.xticks(np.linspace(0, 23, 24,endpoint=True))
    #plt.xlabel('Months')
    #plt.ylabel('Profits')
    #plt.ylim(ymax = max(profit_ret + f_cov_ret))
    #plt.legend()
    #plt.show()


#### Performance ####

    print "mse nn:", mean_squared_error(target_a, pred_nn_a)
    print "mse reg:", mean_squared_error(target_a, pred_reg_a)
    print "mse svr:", mean_squared_error(target_a, pred_svr_a)
    print "mse agg:", mean_squared_error(target_p, agg.predict())
    print "weights:", agg.expost()[1]
    print "profit loss due to misetimation:", agg.expost()[2], "total profit loss:", agg.expost()[3]
