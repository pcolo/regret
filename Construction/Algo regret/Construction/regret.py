# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pyplot as plt

ITER_MAX = 1
EPEX_SPOT = 50.90


#def cover(eps, spot):
   # """Algorithme de calcul des coûts.

   # :param float eps: description
   # :param float spot: description
   # :return description
   # :rtype float
    # """
   # cost = spot *
   # cost = 0.
   # return cost


def expert_nr(nu, eps, cost, back_price, back_demand, feedback):
    """Algorithme de réseau de neurone prédisant la demande à venir pour l'aggrégateur.

    :param float nu: description
    :param float eps: description
    :param float cost: description
    :param float back_price: description
    :param float back_demand: description
    :param float feedback: description
    :return description
    :rtype tuple
    """
    ret = 0.
    return ret


def expert_svr(nu, eps, cost, back_price, back_demand, feedback):
    """Algorithme de réseau de support vector r prédisant la demande à venir pour l'aggrégateur.

    :param float nu: description
    :param float eps: description
    :param float cost: description
    :param float back_price: description
    :param float back_demand: description
    :param float feedback: description
    :return description
    :rtype tuple
    """
    ret = 0.
    return ret


def expert_elm(nu, eps, cost, back_price, back_demand, feedback):
    """Algorithme de extreme learning machine prédisant la demande à venir pour l'aggrégateur.

    :param float nu: description
    :param float eps: description
    :param float cost: description
    :param float price: description
    :param float demand: description
    :param float feedback: description
    :return description
    :rtype tuple
    """
    ret = 0.
    return ret


if __name__ == '__main__':
    back_price = 0.
    back_demand = 0.
    feedback =0.

    # Init
    nu = np.random.normal()
    eps = np.random.normal()
    cost = cover(eps, EPEX_SPOT)

    # Algo
    for i in range(ITER_MAX):
        D_nr = expert_nr(nu, eps, cost, back_price, back_demand, feedback)
        print(D_nr)
        D_svr = expert_svr(nu, eps, cost, back_price, back_demand, feedback)
        print(D_svr)
        D_elm = expert_elm(nu, eps, cost, back_price, back_demand, feedback)
        print(D_elm)

