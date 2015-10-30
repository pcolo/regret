# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import timeit



## ----------------------- Data Dimension ---------------------------- ##


d_base = 96*365*2
d_test = 0.25

d_skip = 365*96
d_pred = 96*1

## ----------------------- Data Creation ---------------------------- ##

bdata = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\MT_161.csv', delimiter=';', na_values=0,
                      nrows=d_base, index_col=0, parse_dates=True, infer_datetime_format=True)

# 0: weekday, 1: month, 2: time, 3: monthday

X = []
for k, v in bdata.iterrows():
    dow = k.dayofweek
    day = k.day
    mth = k.month
    sec = float(k.hour * 3600 + k.minute * 60)/1000
    X.append([dow, day, mth, sec])

y = bdata.values*0.25

#print bdata.ix['Type of consumer']
#a = pd.DataFrame.mean(bdata, axis=0).plot()
pd.DataFrame.plot(bdata)
#print a

#plt.plot(y[0], 'b')
plt.show()