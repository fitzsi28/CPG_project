#Trep and SAC simulation of walker
#Authors: Katie Fitzsimons, Maddy ??

import numpy as np
import trep
from trep import tx,ty,tz,rx,ry,rz
import sactrep
import matplotlib.pyplot as plt
from math import sin

tf=10
dt = 0.01

g = 0
B=0.002 #damping

MH = 0.05 #kg
MW = 0.05 #kg
Hp = 0.025 #m
d = 0.01 #m
dw = 0.1 #m
LEFTWHISK = "Left whisker"
RIGHTWHISK = "Right whisker"
A = np.pi/3

q0= np.array([0,0,0]) #
dq0 = np.array([0,0,0])


system = trep.System()
frames = [#1
    rx('theta', name ="neck",kinematic=True),[#2
        tz(Hp, name = 'head', mass = MH),[#3
            ty(-d,name = "Lcheek"),[
                rx('phiL',name = LEFTWHISK, kinematic = True),[
                    ty(-dw, name = "Lwhisk", mass = MW)]],
            ty(d,name = "Rcheek"),[
                rx('phiR',name = RIGHTWHISK, kinematic = True),[
                    ty(dw, name = "Rwhisk", mass = MW)]]]]]
system.import_frames(frames)

trep.potentials.Gravity(system,(0,0,-g))
trep.forces.Damping(system,B)

print "configs",system.nu

def proj_func(x):
    x[0] = np.fmod(x[0]+np.pi, 2.0*np.pi)
    if(x[0] < 0):
        x[0] = x[0]+2.0*np.pi
    x[0] = x[0] - np.pi
    x[1] = np.fmod(x[1]+np.pi, 2.0*np.pi)
    if(x[1] < 0):
        x[1] = x[1]+2.0*np.pi
    x[1] = x[1] - np.pi
    x[2] = np.fmod(x[2]+np.pi, 2.0*np.pi)
    if(x[2] < 0):
        x[2] = x[2]+2.0*np.pi
    x[2] = x[2] - np.pi

def whiskpath(t):
    q = np.array([0,0,0])
    q[0] = 0
    q[1] = A*sin(t)
    q[2] = -A*sin(t)
    return q


mvi=trep.MidpointVI(system)
mvi.initialize_from_configs(0, q0, dt, q0)

# set initial conditions:
system.q = q0
system.dq = dq0
T = [mvi.t1]
Q = [system.q]


while mvi.t1 < tf:
    mvi.step(mvi.t2+dt, k2=whiskpath(mvi.t2+dt))
    T.append(mvi.t1)
    qtemp = system.q
    proj_func(qtemp)
    Q.append(qtemp)
    if np.abs(mvi.t1%1)<0.1:
        print "time = ",mvi.q1
                        
plt.plot(T,Q) 
plt.show()
# Visualize the system in action
trep.visual.visualize_3d([ trep.visual.VisualItem3D(system, T, Q) ])
            

