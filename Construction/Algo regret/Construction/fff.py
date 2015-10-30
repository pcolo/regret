import numpy as np
import pandas as pd
from itertools import cycle

def get_consumption():
    data = pd.read_csv(r'C:\Users\philippe.colo\Projects\regret\data\LD2014.csv', delimiter=';', na_values=0, nrows=14496-1,
                       index_col=0, parse_dates=True, infer_datetime_format=True, header=1)
    # monthly = data.resample('M', how='sum')

    for index in data.index:
        consumption = data.loc[index] * 0.25
        consumption = consumption[~np.isnan(consumption)]
        epsilon = np.random.uniform(-2.0, 2.0)
        yield index, consumption + (epsilon * 1e3)

    #for d in data:
    #    yield d
#    for i in cycle(range(len(monthly.index))):
#        count = monthly.iloc[i].count()
#        consumption = monthly.iloc[i].sum() * 0.25
#        epsilon = np.random.uniform(-2.0, 2.0)
#        yield count, consumption + (epsilon * 1e3)


if __name__ == '__main__':
    for step, demand in enumerate(get_consumption()):
        print step, demand[0], len(demand[1])

        if step == 10:
            break
