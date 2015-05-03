#Trep and SAC simulation of walker
#Authors: Katie Fitzsimons, Maddy ??

import numpy as np
import trep
from trep import tx,ty,tz,rx,ry,rz
import sactrep
import pylab
from math import sin

tf=1.0
dt = 0.01
tc = 0.0

g = 0
B=0.00 #damping

MH = 0.05 #kg
MW = 0.05 #kg
Hp = 0.025 #m
d = 0.01 #m
dw = 0.1 #m
LEFTWHISK = "Left whisker"
RIGHTWHISK = "Right whisker"
A = np.pi/3.0

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

print "configs",system.configs

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
    q[0] = (np.pi/2.0)*sin(t)
    q[1] = A*sin(16.0*np.pi*t)
    q[2] = -A*sin(16.0*np.pi*t)
    return q


mvi=trep.MidpointVI(system)
mvi.initialize_from_configs(0, q0, dt, q0)

# set initial conditions:
system.q = q0
system.dq = whiskpath(np.pi/2)
T = [mvi.t1]
Q = [system.q]


while mvi.t1 < tf:
    tc = tc+dt
    mvi.step(mvi.t2+dt, k2=whiskpath(tc))
    #system.q = whiskpath(mvi.t1)
    T.append(mvi.t1)
    qtemp = system.q
    proj_func(qtemp)
    Q.append(qtemp)
    if np.abs(mvi.t1%1)<0.1:
        print "time = ",mvi.t1
                        
pylab.plot(T,Q) 
pylab.title("Trep whisking")
pylab.ylabel("q(radians)")
pylab.legend(["Neck","Left Whisker","Right Whisker"])
pylab.show()
# Visualize the system in action
trep.visual.visualize_3d([ trep.visual.VisualItem3D(system, T, Q) ])
            

