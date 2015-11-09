import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_hdf('/Users/Colo/Google Drive/Projects/regret/data/LD2011_2014.hdf', 'ElectricityLoadDiagrams20112014')
ts = data['MT_161'].iloc[96*370*3:96*370*3+96*7-1] #logisitics
a = data['MT_370'].iloc[96*370*3:96*370*3+96*7-1] #weekly labouring
b = data['MT_095'].iloc[96*370*3:96*370*3+96*7-1] #School
c = data['MT_110'].iloc[96*370*3:96*370*3+96*7-1] #hotel


plt.figure(1)
plt.subplot(221)
ts.plot(color='r')
plt.ylabel('Consumption')
plt.title('Logistics')
plt.subplot(222)
a.plot(color='r')
plt.ylabel('Consumption')
plt.title('Weekly business')
plt.subplot(223)
b.plot(color='r')
plt.xlabel('Days')
plt.ylabel('Consumption')
plt.title('School')
plt.subplot(224)
c.plot(color='r')
plt.xlabel('Days')
plt.ylabel('Consumption')
plt.title('Hotel')

plt.show()