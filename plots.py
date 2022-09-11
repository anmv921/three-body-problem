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

from read_data import read_data

plt.style.use("seaborn-bright")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = "true"
plt.rcParams["ytick.right"] = "true"
plt.rcParams["lines.markerfacecolor"] = 'none'

def plot_energy():
    t, H, K, U, x1, y1, z1, x2, y2, z2, x3, y3, z3 = read_data()
    
    plt.plot(t[1:], H[1:], "D", label="H", ms=5, alpha=1)
    plt.plot(t[1:], K[1:], "s", label="K", ms=5, alpha=1)
    plt.plot(t[1:], U[1:], "o", label="U", ms=5, alpha=1)
    
    plt.xlabel("t")
    plt.legend()
    plt.tight_layout()
    plt.margins(x=0.01, y=0.1)
    plt.savefig("E.png")
    plt.show()
# end