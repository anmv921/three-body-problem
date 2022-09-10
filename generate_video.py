import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
import time
import pandas as pd
plt.style.use("seaborn-bright")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = "true"
plt.rcParams["ytick.right"] = "true"
plt.rcParams["lines.markerfacecolor"] = 'none'

def clear_folder(inFolder):
    files = os.listdir(inFolder)
    for fileName in files:
        fullPath = os.path.join(inFolder, fileName)
        os.remove(fullPath)
    # endfor
# end

def generate_video():
    
    print("A gerar o vídeo...")

    start = time.time()
    
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

    folder = "imagens"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # endif
    
    clear_folder(folder)
    for i, _ in enumerate(t):
        imagepath = os.path.join(folder, str(i).zfill(4) + ".jpg")
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        
        plt.plot(x1[:i+1], y1[:i+1], z1[:i+1], c="C0")
        ax.scatter(x1[i], y1[i], z1[i], c="C0")
        
        plt.plot(x2[:i+1], y2[:i+1], z2[:i+1], c="C1")
        ax.scatter(x2[i], y2[i], z2[i], c="C1")
        
        plt.plot(x3[:i+1], y3[:i+1], z3[:i+1], c="C2")
        ax.scatter(x3[i], y3[i], z3[i], c="C2")
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.view_init(30, 50)
        
        plt.title("t="+str(t[i]))
        
        plt.savefig(imagepath)
        
        plt.close(fig)
    # endfor

    nome_vid = "video.mp4"
    if os.path.exists(nome_vid):
        os.remove(nome_vid)
    # endif

    # make video
    strFfmpegCommand = \
    r"ffmpeg -loglevel error -framerate 30 -start_number 0 -i imagens\%04d.jpg -vcodec libx264 video.mp4"
    os.system(strFfmpegCommand)

    print("Video gerado")

    end = time.time()
    print(end - start)
# end