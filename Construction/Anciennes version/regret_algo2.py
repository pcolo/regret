# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle


def expert_svr(demand, step):
    forecast_demand_SVR = []

    from sklearn.svm import SVR
    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(demand, step)

    forecast_demand_SVR = clf.predict(nu_a)

    return forecast_demand_SVR


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
        count = i
        consumption = monthly.iloc[i].sum() * 0.25
        # epsilon = np.random.uniform(-2.0, 2.0)
        yield count, consumption # + (epsilon * 1e3)


if __name__ == '__main__':
    total_loss = 0.

    for step, demand in enumerate(get_consumption()):

        # Display header
        print "Run: {:03d}".format(step)

        # init nu


        feedback = [0.]
        loss_e = 0.

        forecast_demand_SVR = expert_svr(demand, step)
        # forecast_demand_NN, optimum_price_NN = expert_nn(demand[1], feedback)
        # forecast_demand_ELM, optimum_price_ELM = expert_elm(demand[1], feedback)
        # forecast_demand_R, optimum_price_R = expert_r(demand[1], feedback)
        aggregator_price = aggregator(forecast_demand_SVR, loss_e)
        loss_e, loss_a, feedback = expost(aggregator_price, nu_a, nu_b)
        total_loss += loss_a

        # Condition de sortie
        if step == 10:
            break

