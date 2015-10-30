# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation

plt.style.use('ggplot')

from sklearn.svm import SVR
clf = SVR(kernel='rbf', C=1.0, epsilon=0.2)


def cross_val(X, Z):

    clf = SVR(kernel='rbf', C=1.0, epsilon=0.9)
    scores = cross_validation.cross_val_score(clf, X, Z, cv=5)

    return scores

## ----------------------- Data ---------------------------- ##


if __name__ == '__main__':

    trs = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_161.csv', delimiter=';', na_values=0,
                      nrows=96*365, index_col=0, parse_dates=True, infer_datetime_format=True)


    # 0: weekday, 1: month, 2: time, 3: monthday
    X = []
    for k, v in trs.iterrows():
        dow = k.dayofweek
        day = k.day
        mth = k.month
        sec = (k.hour * 3600 + k.minute * 60)/1000
        X.append([dow, day, mth, sec])

    y = trs.MT_161.values

print X[1002]
## ----------------------- Data normalisation ---------------------------- ##

Xmax = np.asarray([x*1.0 for x in np.amax(X, axis=0)])
X = X/Xmax

ymin = np.asarray(np.amin(y, axis=0))
ymax = np.asarray(np.amax(y, axis=0))
y = (y-ymin)/(ymax-ymin)


## ----------------------- SVR results & performance ---------------------------- ##

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.25, random_state=0)

#print y.shape

from sklearn.svm import SVR
from sklearn import cross_validation

clf.fit(X_train, y_train)
print clf.predict(X_test)
#print cross_val(Z, w)


def vect_se(X, y, Z, w):
    return (clf.fit(X, y) - w)**2

def mse(X, y, Z, w):
    return float(sum(vect_se(X, y, Z, w)))/float(len(vect_se(X, y, Z, w)))

#print vect_se(X, y, Z, w)
#print mse(X, y, Z, w)


#plt.plot(expert_svr(X, y, Z))
#plt.plot(y)
#plt.plot(vect_se(X, y, Z, w))
#plt.show()
