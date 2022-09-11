import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
import time
import pandas as pd
import sys
import json

plt.style.use("seaborn-bright")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = "true"
plt.rcParams["ytick.right"] = "true"

np.random.seed(43567362)

def init():
    print("A inicializar o processo...")
    #global m1, m2, m3, N, r1, r2, r3, maxSteps, dt, v1, v2, v3, t,\
    #    sampleStep, start, end, E, H, tMax, U, K
    start = time.time()
    
    with open("config.json", "r") as read_file:
        config = json.load(read_file)
    # end
    
    m1 = config["m1"]
    m2 = config["m2"]
    m3 = config["m3"]
    
    #if len(sys.argv)>1 and sys.argv[1] != "":
    #    tMax = float(sys.argv[1])
    
    tMax = config["tMax"]
    
    sampleStep = config["sampleStep"]
    
    dt = config["dt"]
    
    maxSteps = int(np.ceil(tMax / dt))
    
    t = np.arange(maxSteps)*dt
    
    H =  np.zeros(maxSteps)
    U = np.zeros(maxSteps)
    K = np.zeros(maxSteps-2)
    
    R1 = np.zeros([maxSteps, 3])
    R2 = np.zeros([maxSteps, 3])
    R3 = np.zeros([maxSteps, 3])
    
    V1 = np.zeros([maxSteps, 3])
    V2 = np.zeros([maxSteps, 3])
    V3 = np.zeros([maxSteps, 3])
    
    R1[0] = np.asarray(config["r10"])
    R2[0] = np.asarray(config["r20"])
    R3[0] = np.asarray(config["r30"])
    
    V1[0] = np.asarray(config["v10"])
    V2[0] = np.asarray(config["v20"])
    V3[0] = np.asarray(config["v30"])
    
    
    H[0] = hamiltonian(m1, m2, m3, R1[0], R2[0], R3[0])
    return m1, m2, m3, tMax, sampleStep, dt, maxSteps, t, H, K, U, \
        R1, R2, R3, V1, V2, V3, start
# end

def hamiltonian(m1, m2, m3, r1, r2, r3):
    """
    @In m1, m2, m3
    @In R1(t), R2(t), R3(3)
    """
    return -m1*m2/norm(r1 - r2) - m2*m3/norm(r3 - r2) - m3*m1/norm(r3 - r1)

def rand_vec(A, B):
    """
    Devolve um vetor de 3 elementos
    com números aleatórios uniformes entre A e B
    """
    return np.random.random(3) * (B - A) + A
# end

def norm(v):
    """
    Norma de um vetor
    """
    return np.sqrt(np.sum(v*v))
# end
    
def fij(mi, mj, ri, rj):
    return - mi*mj * (ri - rj) / norm(ri - rj)**3
# end

def force(r1, r2, r3, m1, m2, m3):
    """
    @In R1(t), R2(t), R3(t)
    @Out F1(t) = (F1x, F1y, F1z)(t), F2, F3
    """
    f1 = fij(m1, m2, r1, r2) + fij(m1, m2, r1, r2)
    f2 = fij(m2, m3, r2, r3) + fij(m2, m1, r2, r1)
    f3 = fij(m3, m1, r3, r1) + fij(m3, m2, r3, r2)
    return f1, f2, f3
# end
    

def integrate(F1, F2, F3, v1, v2, v3, r1, r2, r3, dt):
    r1Next, v1Next = leapfrog(F1, v1, r1, dt)
    r2Next, v2Next = leapfrog(F2, v2, r2, dt)
    r3Next, v3Next = leapfrog(F3, v3, r3, dt)
    return r1Next, r2Next, r3Next, v1Next, v2Next, v3Next
# end
    
def leapfrog(F, v, r, dt):
    """
    @In r(0), v(-h/2)
    @Out r(t), v(h/2)
    
    """
    #v[i+1] = v[i] + F * dt
    #r[i+1] = r[i] + dt*v[i+1] # r(0) -> r(h)
    vNext = v + F*dt
    rNext = r + dt*vNext
    return rNext, vNext
# end
    
def singleStep(m1, m2, m3, r1, r2, r3, v1, v2, v3, dt):
    """
    @Out h = H(t)
    """
    h = hamiltonian(m1, m2, m3, r1, r2, r3)
    f1, f2, f3 = force(r1, r2, r3, m1, m2, m3)
    r1Next, r2Next, r3Next, v1Next, v2Next, v3Next = \
        integrate(f1, f2, f3, v1, v2, v3, r1, r2, r3, dt)
    return h, r1Next, r2Next, r3Next, v1Next, v2Next, v3Next
# end
    
def endProcess(U, H, K, start, t, R1, R2, R3, \
               sampleStep, m1, m2, m3, maxSteps, dt):
    np.copyto(U, H)
    np.copyto(K, ecin(R1, R2, R3, maxSteps, m1, m2, m3, dt)) # !
    H[1:-1] += K
    sample(t, R1, R2, R3, H, U, K, sampleStep)
    print("Fim do processo.")
    end = time.time()
    print(end - start)
# end
    
def sample(t, R1, R2, R3, H, U, K, sampleStep):
    data = {
        't': t[::sampleStep],
        'x1': R1[:,0][::sampleStep], \
        'y1': R1[:,1][::sampleStep], \
        'z1': R1[:,2][::sampleStep], \
        'x2': R2[:,0][::sampleStep], \
        'y2': R2[:,1][::sampleStep], \
        'z2': R2[:,2][::sampleStep], \
        'x3': R3[:,0][::sampleStep], \
        'y3': R3[:,1][::sampleStep], \
        'z3': R3[:,2][::sampleStep], \
        'H' : H[::sampleStep], \
        'U' : U[::sampleStep], \
        'K' : K[::sampleStep]
        }
    df = pd.DataFrame(data=data)
    df.to_csv("data.dat", sep=",", index=False)
# end

def calculate_velocity(pos, dt):
    # TODO forward and back differences for 1st and last values
    vel = \
        np.array([ (pos[i+1]-pos[i-1])/(2*dt) \
                  for i in range(1, len(pos)-2) ])
    return vel
# end

def sumSqr(vec):
    return np.sum(vec*vec)
# end

def ecin(R1, R2, R3, maxSteps, m1, m2, m3, dt):
    # todo test
    vel1 = calculate_velocity(R1, dt)
    vel2 = calculate_velocity(R2, dt)
    vel3 = calculate_velocity(R3, dt)
    
    ec = np.empty(maxSteps-2) 
    
    for j in np.arange(maxSteps-3): # whyyyy
        ec[j] = 0.5 * ( m1 * sumSqr(vel1[j]) + m2 * sumSqr(vel2[j]) + \
                     m3 * sumSqr(vel3[j]) )
    # endfor
    return ec
# end
        

def main():
    m1, m2, m3, tMax, sampleStep, dt, maxSteps, \
        t, H, K, U, R1, R2, R3, V1, V2, V3, start = init()
    step = 0
    print("A correr o ciclo principal...")
    while step < maxSteps-1: # whyyyyy
        # print(step)
        H[step], R1[step+1], R2[step+1], R3[step+1], \
            V1[step+1], V2[step+1], V3[step+1] = \
            singleStep(m1, m2, m3, R1[step], R2[step], R3[step], \
                       V1[step], V2[step], V3[step], dt)
        step += 1
    # endwhile
    print("Fim do ciclo principal")
    endProcess(U, H, K, start, t, R1, R2, R3, sampleStep, \
               m1, m2, m3, maxSteps, dt)
# end            
        

if __name__ == "__main__":
    main()
# endif

