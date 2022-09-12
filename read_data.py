# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 13:30:21 2022

@author: andriy
"""

import pandas as pd

def read_data():
    df = pd.read_csv("data.dat", sep=",", header=0, index_col=None)
    t = df["t"].values
    x1 = df["x1"].values
    y1 = df["y1"].values
    x2 = df["x2"].values
    y2 = df["y2"].values
    x3 = df["x3"].values
    y3 = df["y3"].values
    H = df["H"].values
    K = df["K"].values
    U = df["U"].values
    return t, H, K, U, x1, y1, x2, y2, x3, y3
# end