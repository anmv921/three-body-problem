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
plt.rcParams["xtick.top"] = "false"
plt.rcParams["ytick.right"] = "false"
plt.rcParams["lines.markerfacecolor"] = 'none'

def plot_energy():
    t, H, K, U, x1, y1, x2, y2, x3, y3 = read_data()
    
    #plt.plot(t[1:], H[1:], "D", label="H", ms=5, alpha=1)
    #plt.plot(t[1:], K[1:], "s", label="K", ms=5, alpha=1)
    #plt.plot(t[1:], U[1:], "o", label="U", ms=5, alpha=1)
    
    plt.plot(t[1:], H[1:], "-o", label="H", ms=5, alpha=0.5)
    plt.plot(t[1:], K[1:], "-o", label="K", ms=5, alpha=0.5)
    plt.plot(t[1:], U[1:], "-o", label="U", ms=5, alpha=0.5)
    
    plt.xlabel("t")
    plt.legend()
    plt.tight_layout()
    plt.margins(x=0.01, y=0.1)
    plt.savefig("E.png", dpi=1000)
    plt.show()
# end

def plot_trajectory():
    t, H, K, U, x1, y1, x2, y2, x3, y3 = read_data()
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.plot(x3, y3)
    plt.savefig("trajectory.png", dpi=1000)
    plt.show()
# end