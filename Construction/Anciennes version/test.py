# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import cycle

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


for step, demand in enumerate(get_consumption()):

    print demand

    # Display header
    print "Run: {:03d}".format(step)



    if step == 24:
        break

