# -*- coding: utf-8 -*-

from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def expert_svr(X, y, Z):
    forecast_demand_SVR = []

    from sklearn.svm import SVR
    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(X, y)

    forecast_demand_SVR = clf.predict(Z)

    return forecast_demand_SVR

def get_consumption():
    training = pd.read_csv('data/MT_161.csv', delimiter=';', na_values=0, nrows=96*365*1,
                       index_col=0, parse_dates=True, infer_datetime_format=True)
    monthly = training.resample('M', how='sum')

    print monthly

    for i in cycle(range(len(monthly.index))):
        count = i
        consumption = monthly.iloc[i].sum() * 0.25
        epsilon = np.random.uniform(-2.0, 2.0)
        yield count, consumption + (epsilon * 1e3)

if __name__ == '__main__':

    trs = pd.read_csv('data/MT_162.csv', delimiter=';', na_values=0, nrows=96*31,
                     index_col=0, parse_dates=True, infer_datetime_format=True)

    tes = pd.read_csv('data/MT_162.csv', delimiter=';', na_values=0, skiprows=range(1,1*365*96), nrows=96*31,
                     index_col=0, parse_dates=True, infer_datetime_format=True)

    # 0: weekday, 1: month, 2: time, 3: monthday
    X = []
    for k, v in trs.iterrows():
        dow = k.dayofweek
        day = k.day
        mth = k.month
        sec = k.hour * 3600 + k.minute * 60
        X.append([dow, day, mth, sec])

    y = trs.MT_162.values

    Z = []
    for k, v in tes.iterrows():
        dow = k.dayofweek
        day = k.day
        mth = k.month
        sec = k.hour * 3600 + k.minute * 60
        Z.append([dow, day, mth, sec])

    # w = tes.MT_162.values

    from sklearn.svm import SVR
    from sklearn import cross_validation

    print expert_svr(X, y, X)
    print y