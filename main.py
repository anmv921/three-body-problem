import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
import time
import pandas as pd
import sys
import json
from datetime import datetime
from video import generate_video
from plots import read_data, plot_energy, plot_trajectory

np.random.seed(43567362)

# TODO move data to Desktop/ThreeBodyData/current time
# copy config to the same folder
# TODO restricted problem

def init():
    print("A inicializar o processo...")
    
    start = time.time()
    
    with open("config.json", "r") as read_file:
        config = json.load(read_file)
    # end
    
    dtmNow = datetime.now()
    strNow = dtmNow.strftime("%d-%m-%Y_%H-%M-%S")
    strFullDataPath = os.path.join(config["datapath"], strNow)
    os.mkdir(strFullDataPath)
    imageFolder = os.path.join(strFullDataPath, "imagens")
    os.mkdir(imageFolder)
    
    dim = 2
    
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
    
    N = 3
    
    R = []
    V = []
    
    for n in range(N):
        R.append(np.zeros([maxSteps, dim]))
        V.append(np.zeros([maxSteps, dim]))
    # endfor
    
    R[0][0] = np.asarray(config["r10"])
    R[1][0] = np.asarray(config["r20"])
    R[2][0] = np.asarray(config["r30"])
    
    V[0][0] = np.asarray(config["v10"])
    V[1][0] = np.asarray(config["v20"])
    
    eps = config["eps"]
    
    r12 = norm(R[0][0] - R[1][0])
    
    rcm = (m1 * R[0][0] + m2 * R[1][0])/(m1 + m2)
    
    rho1 = norm(R[0][0] - rcm)
    rho2 = norm(R[1][0] - rcm)
    
    V[0][0][1] = np.sqrt(m2 * (rho1) / r12**2)
    #np.sqrt(m2 * r12 * r12 / (r12*r12 + eps*eps)*(3/2) )
    V[1][0][1] = -np.sqrt(m1 * (rho2) / r12**2)
    #-np.sqrt(m1 * r12 * r12 / (r12*r12 + eps*eps)*(3/2))
    
    
    
    
    V[2][0] = np.asarray(config["v30"])
    
    
    
    
    H[0] = hamiltonian(m1, m2, m3, R[0][0], R[1][0], R[2][0], eps)
    return m1, m2, m3, tMax, sampleStep, dt, maxSteps, t, H, K, U, \
        R[0], R[1], R[2], V[0], V[1], V[2], start,\
            dim, strFullDataPath, eps, imageFolder
# end


def hamiltonian(m1, m2, m3, r1, r2, r3, eps):
    """
    @In m1, m2, m3
    @In R1(t), R2(t), R3(3)
    """
    
    nr12 = norm(r1 - r2)
    nr32 = norm(r3 - r2)
    nr31 = norm(r3 - r1)
    
    #return -m1*m2/norm(r1 - r2) - m2*m3/norm(r3 - r2) - m3*m1/norm(r3 - r1)
    return -m1*m2/np.sqrt(nr12*nr12 + eps*eps) - m2*m3/np.sqrt(nr32*nr32 + eps*eps) - m3*m1/np.sqrt(nr31*nr31 + eps*eps)
#end

@njit
def rand_vec(A, B, dim):
    """
    Devolve um vetor de 3 elementos
    com números aleatórios uniformes entre A e B
    """
    return np.random.random(dim) * (B - A) + A
# end

@njit
def norm(v  ):
    """
    Norma de um vetor
    """
    return np.sqrt(np.sum(v*v))
# end
    
@njit
def fij(mi, mj, ri, rj, eps):
    #eps = 0.1
    #return - mi*mj * (ri - rj) / norm(ri - rj)**3
    nr = norm(ri - rj)
    return - mi*mj * (ri - rj) / (( nr*nr  + eps*eps)**(3/2))
# end

@njit
def force(r1, r2, r3, m1, m2, m3, eps):
    """
    @In R1(t), R2(t), R3(t)
    @Out F1(t) = (F1x, F1y, F1z)(t), F2, F3
    """
    f1 = fij(m1, m2, r1, r2, eps) + fij(m1, m3, r1, r3, eps)
    f2 = fij(m2, m3, r2, r3, eps) + fij(m2, m1, r2, r1, eps)
    f3 = fij(m3, m1, r3, r1, eps) + fij(m3, m2, r3, r2, eps)
    return f1, f2, f3
# end
    
def integrate(f1, f2, f3, v1, v2, v3, r1, r2, r3, dt):
    r1Next, v1Next = leapfrog(f1, v1, r1, dt)
    r2Next, v2Next = leapfrog(f2, v2, r2, dt)
    r3Next, v3Next = leapfrog(f3, v3, r3, dt)
    return r1Next, r2Next, r3Next, v1Next, v2Next, v3Next
# end
    
@njit
def leapfrog(F, vLeap, r, dt):
    """
    @In r(0), v(-h/2)
    @Out r(t), v(h/2)
    
    """
    #v[i+1] = v[i] + F * dt
    #r[i+1] = r[i] + dt*v[i+1] # r(0) -> r(h)
    vNext = vLeap + F*dt
    rNext = r + dt*vNext
    return rNext, vNext
# end
    

def singleStep(m1, m2, m3, r1, r2, r3, v1, v2, v3, dt, eps):
    """
    @Out h = H(t)
    """
    h = hamiltonian(m1, m2, m3, r1, r2, r3, eps)
    f1, f2, f3 = force(r1, r2, r3, m1, m2, m3, eps)
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
        'x2': R2[:,0][::sampleStep], \
        'y2': R2[:,1][::sampleStep], \
        'x3': R3[:,0][::sampleStep], \
        'y3': R3[:,1][::sampleStep], \
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

@njit
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
     
def main(in_boolVideo=False):
    """
    Simulação do problema de 3 corpos em 3d
    """
    m1, m2, m3, tMax, sampleStep, dt, maxSteps, \
        t, H, K, U, R1, R2, R3, V1, V2, V3, start, dim, \
            datapath, eps, imageFolder = init()
    step = 0
    print("A correr o ciclo principal...")
    while step < maxSteps-1: # whyyyyy
        # print(step)
        H[step], R1[step+1], R2[step+1], R3[step+1], \
            V1[step+1], V2[step+1], V3[step+1] = \
            singleStep(m1, m2, m3, R1[step], R2[step], R3[step], \
                       V1[step], V2[step], V3[step], dt, eps)
        step += 1
    # endwhile
    print("Fim do ciclo principal")
    endProcess(U, H, K, start, t, R1, R2, R3, sampleStep, \
               m1, m2, m3, maxSteps, dt)
    plot_energy()
    plot_trajectory()
    if in_boolVideo: 
        generate_video(imageFolder)
# end            
        
if __name__ == "__main__":
    main()
# endif

