import numpy as np
import pandas as pd
import math
from scipy.special import factorial
import matplotlib.pyplot as plt


## ----------------------- Data Creation ---------------------------- ##

data = pd.read_hdf('/Users/Colo/Google Drive/Projects/regret/data/LD2011_2014.hdf', 'ElectricityLoadDiagrams20112014').resample('M', how='sum')

#a = np.sqrt(data.ix[:,'MT_300':].var().sum())
#b = np.sqrt(data.ix[:,'MT_301':].var().sum())
a = 0.01
d = 0.6

b = np.random.uniform(a, d)
p_plus = 1.48 #tbc
p_minus = 1.46 #tbc
#c = 140
N = 150

p_f =  1.44#tbc
r_price = 0.1433




j = data['MT_300'][0]
j = 1.0


S= []
E= []
for c in range(3,150):
    sigma_b = (np.array([[0.1]*(c+1)])**2)*j
    sigma_a = (np.array([[0.1]*c])**2)*j

    epsilon_b = np.random.normal(0, (sigma_b)**2)
    D_plus_b = np.amax([np.sum(epsilon_b), 0], axis=0)
    D_minus_b = np.amax([- np.sum(epsilon_b), 0], axis=0)

    epsilon_a = np.random.normal(0, (sigma_a)**2)
    D_plus_a = np.amax([np.sum(epsilon_a), 0], axis=0)
    D_minus_a = np.amax([- np.sum(epsilon_a), 0], axis=0)


    shap = j/float(np.sqrt(sigma_b.sum()) - np.sqrt(sigma_a.sum())) - float(factorial(N - c)*factorial(c-1)*(p_minus - p_plus + b*np.sqrt(2*math.pi)))/float(np.sqrt(2*math.pi)*(factorial(N-1)*r_price*(N-1)-factorial(N-c)*factorial(c-1)*p_f))
    S.append(shap)

    fin = factorial(N-c)*factorial(c-1)/(factorial(N))*(p_f*j+p_minus*(D_minus_b - D_minus_a) - p_plus*(D_plus_b - D_plus_a))+ (r_price*j)/N
    E.append(fin)

print E

#plt.plot(E, color='b')
#plt.xlabel('Size of the coalition')
#plt.ylabel('Shapley value')
#plt.title('What is the Shapley value?')
#plt.axis([0, 151, 422, 423])
#plt.show()

plt.plot(S, color='b')
plt.xlabel('Size of the coalition')
plt.title('Is the Shapley value in the core?')
#plt.axis([0, 151, -1000, 1000])
plt.show()

