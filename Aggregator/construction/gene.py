import numpy as np
import random
import pandas as pd

a_price = 0.14
r_price = 0.1433
beta = 0.4

data = pd.read_hdf('/Users/Colo/Google Drive/Projects/regret/data/LD2011_2014.hdf', 'ElectricityLoadDiagrams20112014').resample('M', how='sum')
data = (0.25 *data - data.min(axis=0)) / (0.25 * (data.max(axis=0)- data.min(axis=0)))

# Omega, c and d are to be calibrated

def generator(X):

    cons = 370
    c = 0.01
    d = 0.6
    periods = 12

    b = np.array([np.random.uniform(c, d, cons)]*periods)
    s = 7*np.sum(np.random.normal(r_price - a_price, 2*(b)**2), axis=1) #s(i,m)
    c_r = range(1, cons+1) #coalition of the retailer
    c_a = [] #coalition of the aggregator
    d_r = [] #demand of the retailer
    d_a = [] #demand of the aggregator

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

    return "ret's demand:", d_r, "agg's demand:", d_a, s

print generator(data)[4].mean()

dates = pd.date_range(start='01/2011', freq= 'M', periods=12)

o = []
for i in range(12):
    o.append([generator(data)[1][i], generator(data)[3][i]])

out = pd.DataFrame(np.array(o).reshape(12,2), index=dates,  columns=['Retailer', 'Aggregator']).fillna(0)
print out.iloc[1]

#Calibration

c = 0.1433333 - 10**(-2)*((6.72+10.57+6.38+5.99+5.46+5.86+9.14+5.3+11.97+6.08+5.26+6.29+5.67+5.86+7.25+5.76)/16 - 5.26)
#print c

