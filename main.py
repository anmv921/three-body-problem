import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
import time
import pandas as pd
import sys
plt.style.use("seaborn-bright")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = "true"
plt.rcParams["ytick.right"] = "true"
np.random.seed(43567362)


def init():
    print("A inicializar o processo...")
    global m1, m2, m3, N, r1, r2, r3, maxSteps, dt, v1, v2, v3, t,\
        sampleStep, start, end, E
    start = time.time()
    m1 = 3
    m2 = 4
    m3 = 5
    
    if len(sys.argv)>1 and sys.argv[1] != "":
        maxSteps = int(sys.argv[1])
    else:
        maxSteps = 1000000
    
    sampleStep = 100
    
    dt = 0.0001
    
    t = np.arange(maxSteps)*dt
    
    E =  np.zeros(maxSteps)
    
    r1 = np.zeros([maxSteps, 3])
    r2 = np.zeros([maxSteps, 3])
    r3 = np.zeros([maxSteps, 3])
    
    v1 = np.zeros([maxSteps, 3])
    v2 = np.zeros([maxSteps, 3])
    v3 = np.zeros([maxSteps, 3])
    
    r1[0] = np.array([1, 3, 0])
    r2[0] = np.array([-2, 1, 0])
    r3[0] = np.array([1,-1, 0])
    
    v1[0] = np.array([0, 0, 0])
    v2[0] = np.array([0, 0, 0])
    v3[0] = np.array([0, 0, 0])
    
# end

def rand_vec(A, B):
    """
    returns a random 3-vector where each element is between A and B
    """
    vec = np.random.random(3) * (B - A) + A
    return vec
# end


def norm(v):
    return np.sqrt(np.sum(v*v))
# end
    
def fij(mi, mj, ri, rj):
    F = - mi*mj * (ri - rj) / norm(ri - rj)**3
    return F
# end


def force(n):
    F1 = fij(m1, m2, r1[n,:], r2[n,:]) + fij(m1, m2, r1[n,:], r2[n,:])
    F2 = fij(m2, m3, r2[n,:], r3[n,:]) + fij(m2, m1, r2[n,:], r1[n,:])
    F3 = fij(m3, m1, r3[n,:], r1[n,:]) + fij(m3, m2, r3[n,:], r2[n,:])
    return F1, F2, F3
# end
    

def integrate(F1, F2, F3, i):
    leapfrog(F1, i, v1, r1)
    leapfrog(F2, i, v2, r2)
    leapfrog(F3, i, v3, r3)
# end
    
def leapfrog(F, i, v, r):
    v[i+1] = v[i] + F * dt # v(-h/2) -> v(h/2)
    r[i+1] = r[i] + dt*v[i+1] # r(0) -> r(h)
# end
    
def singleStep(step):
    F1, F2, F3 = force(step)
    integrate(F1, F2, F3, step)
# end
    
def endProcess():
    sample()
    print("Fim do processo.")
    end = time.time()
    print(end - start)
# end
    
def sample():
    data = {'t': t[::sampleStep], 'x1': r1[:,0][::sampleStep],\
            'y1': r1[:,1][::sampleStep],\
            'z1': r1[:,2][::sampleStep], 'x2': r2[:,0][::sampleStep], \
            'y2': r2[:,1][::sampleStep],'z2': r2[:,2][::sampleStep], \
            'x3': r3[:,0][::sampleStep],'y3': r3[:,1][::sampleStep], \
            'z3': r3[:,2][::sampleStep]}
    df = pd.DataFrame(data=data)
    df.to_csv("data.dat", sep=",", index=False)
# end

def main():
    init()
    step = 0
    while step < maxSteps-1:
        singleStep(step)
        step += 1
    # endwhile
    endProcess()
# end
    
if __name__ == "__main__":
    main()
    

