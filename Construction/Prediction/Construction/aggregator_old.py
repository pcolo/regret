import numpy as np
import pandas as pd

d_size_train = 96*3
d_size_test = 96
d_size = d_size_train + d_size_test
d_skip = 365*96-1

trs = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_161.csv', delimiter=';', na_values=0,
                    nrows=d_size_train, index_col=0, parse_dates=True, infer_datetime_format=True)

tes = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_161.csv', delimiter=';', na_values=0,
                    skiprows=range(1, d_skip), nrows=d_size_test, index_col=0, parse_dates=True, infer_datetime_format=True)

bdata = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_161.csv', delimiter=';', na_values=0, nrows=d_size,
                    index_col=0, parse_dates=True, infer_datetime_format=True)

# 0: weekday, 1: month, 2: time, 3: monthday
X_1 = []
for k, v in trs.iterrows():
    dow = k.dayofweek
    day = k.day
    mth = k.month
    sec = (k.hour * 3600 + k.minute * 60)/1000
    X_1.append([dow, day, mth, sec])

y_1 = trs.MT_161.values*0.25

Z_1 = []
for k, v in tes.iterrows():
    dow = k.dayofweek
    day = k.day
    mth = k.month
    sec = (k.hour * 3600 + k.minute * 60)/1000
    Z_1.append([dow, day, mth, sec])

w_1 = tes.MT_161.values*0.25


B = np.array([0.]*len(w_1))
C = np.array([0.]*len(w_1))
D = np.array([0.]*len(w_1))

class RegretAggregetor(object):
    def __init__(self, nu=0.01, r=np.arange(0, 3)):
        self.nu = nu
        self.r = r

    def fit(self, B, C, D, y):
        self.p = np.array([1.0/float(self.r.size)]*self.r.size)
        self.loss_a = []
        for i in y:
            self.loss_e = np.array([(B[i] - i)**2, (C[i] - i)**2, (D[i] - i)**2])
            print self.loss_e
            self.p = self.p * np.exp(-self.nu * self.loss_e) / np.sum(self.p * np.exp(-self.nu * self.loss_e))
            self.loss_a.append((i - self.loss_e*self.p)**2)
        self.total_loss = np.sum(self.loss_a)
        return self

    def predict(self, X):
        u = np.random.uniform(0, 1, len(X))
        s = np.cumsum(self.p)
        self.y = []
        for i in u:
            self.y.append(float(np.abs(i - s).argmin())/self.r.size)
        return self.y

    def expost(self, X):
        vect_se = (self.y - X)**2
        mse = float(np.sum(vect_se))/float(len(vect_se))
        self.total_loss = np.sum(self.loss_a)
        return mse, self.total_loss

agg = RegretAggregetor()
agg.fit(B, C, D, w_1)

print agg.predict(w_1)