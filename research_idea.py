import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import optimize
from collections import defaultdict


a = 83
b = 42
N = 100
P = 157

mod_func = lambda x: x % P
hash_func = lambda x: (a * x + b) % P % N


def find_cycle(array):
    # Find Cycle
    cycle = [array[0]]
    is_cycle = False
    for idx in range(1, len(array)):
        if array[idx] == cycle[0]:
            is_cycle = True
            for idx2 in range(idx, len(array)):
                period = len(cycle)
                if cycle[idx2 % period] != array[idx2]:
                    is_cycle = False
                    break

        if not is_cycle:
            cycle.append(array[idx])
        else:
            break

    if not is_cycle:
        return False
    else:
        return cycle, period

# print(find_cycle([2, 3, 4, 2, 3, 4, 5, 2, 3, 4, 2, 3, 4, 5]))


def plot_mod():
    x = range(500)
    y1 = list(map(mod_func, x))

    plt.plot(x, y1)
    plt.show()


def plot_hash():
    x = range(1000)
    y2 = list(map(hash_func, x))
    print(y2)

    # Count Occurrence
    # count_dict = defaultdict(int)
    # for y in y2:
    #     count_dict[y] += 1
    # print(count_dict)

    print(find_cycle(y2))

    plt.plot(x, y2)
    plt.show()


def learn_sine():
    theta = 5.1
    w = 5
    iteration = 1000
    stepsize = 0.001

    target_func = lambda x, ww: np.sin(ww * x)
    loss_func = lambda y1, y2: np.sum(np.square(y1 - y2))
    # gradient_func = lambda ww: np.multiply(x, np.cos(ww * x))

    x_data = np.linspace(0, 20, 1000)
    y_data = np.sin(w * x_data)

    param, param_cov = optimize.curve_fit(target_func, x_data, y_data, p0=4.8)
    print(param)

    plt.plot(x_data, y_data)
    plt.show()


def learn_hash():
    x_data = range(100)
    y_data = np.array(list(map(hash_func, x_data)))
    print(y_data.shape)
    b_hat = y_data[0]
    P_hat = np.amax(y_data)
    a_hat = y_data[1] - y_data[0]

    print(a_hat, b_hat, P_hat)

if __name__ == '__main__':
    # learn_sine()
    plot_hash()
    # learn_hash()
