import numpy as np
d_train_e = 5
d_train_a = d_train_e + 20
d_pred = d_train_a + 20

r_price = 0.1433
c = 0.01
d = 0.6
cons = 370

c_a = range(1, cons+1)


def obj_price(iter, alpha, c_a): #alpha \in [0,1] level of confidence and iter number of \sum\eta_i - \Pi_C plotted to calculate probability
    p_plus = 0. #tbc
    p_minus = 0. #tbc
    p_f = 0. #tbc

    D = []
    for i in range(iter):

        b = np.array([np.random.uniform(c, d, len(c_a))]*(d_pred - d_train_a))
        s_mu_r = 7*np.sum(np.random.normal(0, len(c_a)*(b)**2), axis=1)

        sigma = np.array([[0.1]*len(c_a)]*(d_pred - d_train_a)) #tbc
        epsilon = 7*np.random.normal(0, (sigma)**2)
        epsilon_plus = np.amax([epsilon, [[0]*len(c_a)]*len(epsilon)], axis=0)
        epsilon_minus = np.amax([-epsilon, [[0]*len(c_a)]*len(epsilon)], axis=0)

        D_plus = np.amax([np.sum(epsilon, axis=1), [0]*len(np.sum(epsilon, axis=1))], axis=0)
        D_minus = np.amax([- np.sum(epsilon, axis=1), [0]*len(np.sum(epsilon, axis=1))], axis=0)
        s_epsilon_plus = np.sum(epsilon_plus, axis=1)
        s_epsilon_minus = np.sum(epsilon_minus, axis=1)


        D.append(p_plus*s_epsilon_plus - p_minus*s_epsilon_minus +(r_price - s_mu_r - p_f - p_minus*(1 - 1.0/len(c_a)))*D_minus - \
            (r_price - s_mu_r - p_f - p_plus*(1 - 1.0/len(c_a)))*D_plus)

    E = []
    for i, v in enumerate(np.transpose(D)):
        E.append(np.cumsum(np.transpose(np.array(D))[i])[iter*alpha-1])

    return E # returns the vector of minimum values of Pi_A such as p(Pi_A \geq \sum \eta_j - \Pi_C) = \alpha for each period

#print len(obj_price(1000, 0.8, c_a))
print obj_price(1000, 0.8, c_a)

