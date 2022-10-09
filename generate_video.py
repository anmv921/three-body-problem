import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
import time
import pandas as pd

from read_data import read_data

plt.style.use("seaborn-bright")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = "false"
plt.rcParams["ytick.right"] = "false"
#plt.rcParams["lines.markerfacecolor"] = 'none'


def clear_folder(inFolder):
    files = os.listdir(inFolder)
    for fileName in files:
        fullPath = os.path.join(inFolder, fileName)
        os.remove(fullPath)
    # endfor
# end

def generate_video(folder):
    
    print("A gerar o vídeo...")

    start = time.time()
    
    t, H, K, U, x1, y1, x2, y2, x3, y3 = read_data()

    #folder = "imagens"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # endif
    
    clear_folder(folder)
    for i, _ in enumerate(t):
        imagepath = os.path.join(folder, str(i).zfill(4) + ".jpg")
        
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        plt.plot(x1[:i+1], y1[:i+1], c="C0")
        ax.scatter(x1[i], y1[i], c="C0")
        
        plt.plot(x2[:i+1], y2[:i+1], c="C1")
        ax.scatter(x2[i], y2[i], c="C1")
        
        plt.plot(x3[:i+1], y3[:i+1], c="C2")
        ax.scatter(x3[i], y3[i], c="C2")
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        plt.title(f't={t[i]:.4f}')
        
        plt.savefig(imagepath)
        
        plt.close(fig)
    # endfor

    nome_vid = "video.mp4"
    if os.path.exists(nome_vid):
        os.remove(nome_vid)
    # endif
    
    
    # make video
    #r"ffmpeg -loglevel error -framerate 30 -start_number 0 -i imagens\%04d.jpg -vcodec libx264 video.mp4"
    strFfmpegCommand = \
    r"ffmpeg -loglevel error -framerate 30 -start_number 0 -i " + \
        folder + r"\%04d.jpg -vcodec libx264 video.mp4"
    os.system(strFfmpegCommand)

    print("Video gerado")

    end = time.time()
    print(end - start)
# end