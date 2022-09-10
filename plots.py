# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 22:08:36 2022

@author: andriy
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
import time
import pandas as pd
import seaborn as sns
#sns.set_palette("husl", 10)
plt.style.use("seaborn-bright")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = "true"
plt.rcParams["ytick.right"] = "true"
plt.rcParams["lines.markerfacecolor"] = 'none'

def read_data():
    df = pd.read_csv("data.dat", sep=",", header=0, index_col=None)
    t = df["t"].values
    x1 = df["x1"].values
    y1 = df["y1"].values
    z1 = df["z1"].values
    x2 = df["x2"].values
    y2 = df["y2"].values
    z2 = df["z2"].values
    x3 = df["x3"].values
    y3 = df["y3"].values
    z3 = df["z3"].values
    H = df["H"].values
    K = df["K"].values
    U = df["U"].values
    return t, H, K, U
# end


def plot_energy():
    t, H, K, U = read_data()
    plt.plot(t[1:], H[1:], ".", label="H", mfc="C0", alpha=0.5,
             ms=10)
    plt.plot(t[1:], K[1:], "s", label="K", ms=10)
    plt.plot(t[1:], U[1:], "o", label="U", ms=12)
    plt.xlabel("t")
    plt.legend()
    plt.show()
# end
    
plot_energy()
