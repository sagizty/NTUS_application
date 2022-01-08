"""
ref
https://perso.crans.org/besson/publis/notebooks/Runge-Kutta_methods_for_ODE_integration_in_Python.html#Runge-Kutta-method-of-order-4,-%22RK4%22

"""
import matplotlib

matplotlib.use('Agg')  # use non-GUI backend

import numpy as np
from scipy.integrate import odeint  # for comparison
import matplotlib.pyplot as plt


def odefun(x, t):
    """

    :param x: vector of variables
    :param t: step

    :return: divedent equations
    """
    # divedent equation:
    # d P / d t = k3 * ES
    # d S / d t = k2 * ES - k1 * S * E
    # d ES / d t = k1 * S * E - (k2+k3) * ES
    # d E / d t = (k3+k2) * ES - k1 * S * E

    return np.array([150 * x[2],
                     600 * x[2] - 100 * x[1] * x[3],
                     100 * x[1] * x[3] - (600 + 150) * x[2],
                     (600 + 150) * x[2] - 100 * x[1] * x[3]])


def RK4(f, y0, t):
    n = len(t)

    y = np.zeros((n, len(y0)))

    y[0] = y0

    for i in range(n - 1):
        # step
        h = t[i + 1] - t[i]

        k1 = f(y[i], t[i])
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2.)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2.)
        k4 = f(y[i] + k3 * h, t[i] + h)

        y[i + 1] = y[i] + (h / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


if __name__ == '__main__':
    x0 = np.array([0, 10, 0, 1])  # P, S, ES, E

    t = np.linspace(0, 0.5, 100000)  # sample sequence of 0.5 min and 0.02 min

    # output a sequence of results
    print('RK4:')
    output = RK4(odefun, x0, t)
    # print(output)

    # comparing with scipy
    # print('\ncomparing with scipy')
    # output = odeint(odefun, x0, t)
    # print(output)

    # enzymatic reaction velocity V
    V = 150 * output[:, 2]
    # concentrations of P
    P = output[:, 0]
    # concentrations of S
    S = output[:, 1]
    # concentrations of ES
    ES = output[:, 2]
    # concentrations of E
    E = output[:, 3]

    # figure settings
    plt.rcParams['figure.figsize'] = (25.0, 8.0)
    plt.rcParams['figure.dpi'] = 150

    # plot concentrations of species fig
    plt.plot(t, P, 'r', label=r'P (t)')
    plt.plot(t, S, 'b', label=r'S (t)')
    plt.plot(t, ES, 'g', label=r'ES (t)')
    plt.plot(t, E, 'y', label=r'E (t)')
    plt.legend(loc='best', fontsize=20)
    plt.xlabel('time step (min)', fontsize=20)
    plt.ylabel('concentrations of species (μM)', fontsize=20)
    plt.grid()
    plt.show()
    plt.savefig('./concentrations of species (0.5 min).png')
    plt.close()

    # plot V-S VM fig
    plt.plot(S, V, 'b', label=r'reaction velocity V')
    VM = max(V) * np.ones_like(V)
    plt.plot(S, VM, 'r', label=r'maximum reaction velocity VM: '+str(max(V)))
    plt.legend(loc='best', fontsize=20)
    plt.xlabel('concentrations of S (μM)', fontsize=20)
    plt.ylabel('enzymatic reaction velocity V', fontsize=20)
    plt.grid()
    plt.show()
    plt.savefig('./V-S (0.5 min).png')
    plt.close()
