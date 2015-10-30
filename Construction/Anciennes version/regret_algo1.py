# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle



def expert_svr(demand, feedback, step):
    forecast_demand_SVR = []
    optimum_price_SVR = []

    from sklearn.svm import SVR
    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(feedback, step)
    x_a = clf.predict(nu_a)

    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(feedback, step)
    x_b = clf.predict(nu_b)

    for aggregator_price in range(0,0.2,0.0001):
        forecast_demand_SVR.append((1-BETA)*demand + BETA*demand*x_b*(x_a + retailer_price - aggregator_price))

    m = max(forecast_demand_SVR)
    optimum_price_SVR = [i for i, j in enumerate(forecast_demand_SVR) if j == m]

    return forecast_demand_SVR, optimum_price_SVR


def expert_nn(demand, feedback):
    forecast_demand_NN = []
    optimum_price_NN = []
    return forecast_demand_NN, optimum_price_NN


def expert_elm(demand, feedback):
    forecast_demand_ELM = []
    optimum_price_ELM = []
    return forecast_demand_ELM, optimum_price_ELM


def expert_r(demand, feedback):
    forecast_demand_R = []
    optimum_price_R = []
    return forecast_demand_R, optimum_price_R


def aggregator(forecast_demand, loss_e):
    aggregator_price = 0.
    return aggregator_price


def expost(aggregator_price, epsilon):
    loss_e = 0.
    loss_a = 0.
    feedback = []
    return loss_e, loss_a, feedback


def get_consumption():
    data = pd.read_csv('data/MT_161.csv', delimiter=';', na_values=0, nrows=14496-1,
                       index_col=0, parse_dates=True, infer_datetime_format=True)
    monthly = data.resample('M', how='sum')

    for i in cycle(range(len(monthly.index))):
        count = monthly.iloc[i].count()
        consumption = monthly.iloc[i].sum() * 0.25
        epsilon = np.random.uniform(-2.0, 2.0)
        yield count, consumption + (epsilon * 1e3)


if __name__ == '__main__':
    total_loss = 0.

    for step, demand in enumerate(get_consumption()):

        # Display header
        print "Run: {:03d}".format(step)

        # init nu
        BETA = 0.4
        retailer_price = 0.1433
        aggregator_price = 0.14
        nu_a = np.random.normal(np.random.uniform(0, 0.2), np.random.uniform(-0.01, 0.01))
        nu_b = np.random.normal(np.random.uniform(0, 0.2), np.random.uniform(-0.01, 0.01))

        feedback = [0.]
        loss_e = 0.

        forecast_demand_SVR, optimum_price_SVR = expert_svr(demand[1], feedback, step)
        # forecast_demand_NN, optimum_price_NN = expert_nn(demand[1], feedback)
        # forecast_demand_ELM, optimum_price_ELM = expert_elm(demand[1], feedback)
        # forecast_demand_R, optimum_price_R = expert_r(demand[1], feedback)
        aggregator_price = aggregator(forecast_demand_SVR, loss_e)
        loss_e, loss_a, feedback = expost(aggregator_price, nu_a, nu_b)
        total_loss += loss_a

        # Condition de sortie
        if step == 10:
            break

