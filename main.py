import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from numpy import pi
from ha import HA
from multiprocessing import Pool
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt


def ga_demo(x):
    f2 = ((x[0] - 5) / 1.2) ** 2 + (x[1] - 6) ** 2 < 16
    f1 = 6.452 * (x[0] + 0.125 * x[1]) * (np.cos(x[0]) - np.cos(2 * x[1])) ** 2
    f1 = f1 / np.sqrt(0.8 + (x[0] - 4.2) ** 2 + 2 * (x[1] - 7) ** 2)
    f1 = f1 + 3.226 * x[1]
    f = 100 - f1 * f2
    return -100 * (100 - f) / 100


def ackley(x):
    a = 20
    b = 0.2
    c = 2 * pi
    d = len(x)
    f = -a * np.exp(-b * np.sqrt(1 / d * np.sum(x ** 2))) - np.exp(1 / d * np.sum(np.cos(c * x))) + a + np.exp(1)
    return -100 * (22.3203 - f) / 22.3203


# def griewank(x):
#     f = 1 + 1 / 4000 * np.sum(x ** 2) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
#     return -100 * (27010 - f) / 27010


if __name__ == '__main__':
    # HA(ackley, 30, -32, 32).ha(0)

    # 并行计算 ha
    pool_num = 30
    with Pool(pool_num) as p:
        res = p.map(HA(ackley, 30, -32, 32).ha, range(pool_num))
    # 转为pd
    res = pd.DataFrame(np.hstack(res), columns=['nfev', 'fitness'] * pool_num)
    # 保存
    res.to_csv('ha.csv', index=False)
