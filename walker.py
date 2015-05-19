#Trep and SAC simulation of walker
#Authors: Katie Fitzsimons, Mahdieh

import numpy as np
import trep
from trep import tx,ty,tz,rx,ry,rz
import pylab
import sactrep

tf=10.0
dt = 0.01
tc = 0.0

g = 9.81
B=0.002 #damping

MH = 0.5 #kg
MUL =0.2 #kg
MLL = 0.1 #kg
UL = 0.25 #m
LL = 0.25 #m
HIP = "hip joint"
ULEG = "UpperLeg"
LLEG = "LowerLeg"



system = trep.System()
frames = [#1
    ty('yb', name ="body", mass = MH),[#2
        rx('theta1', name = HIP,),[#3
            tz(-UL ,name = ULEG, mass = MUL),[
                rx('theta2', name = 'knee'),[
                    tz(-LL, name = LLEG, mass = MLL)]]]]]
system.import_frames(frames)

trep.potentials.Gravity(system,(0,0,-g))
trep.forces.Damping(system,B)
trep.forces.ConfigForce(system, 'theta1','hip-torque')


print "configs",system.nu

q0= np.array([0,-np.pi/2,0]) #
dq0 = np.array([0,0,0])

mvi=trep.MidpointVI(system)
mvi.initialize_from_configs(0, q0, dt, q0)

T = [mvi.t1]
Q = [system.q]

while mvi.t1 < tf:
    tc = tc+dt
    mvi.step(mvi.t2+dt, [0.0])
    T.append(mvi.t1)
    Q.append(mvi.q1)
    if np.abs(mvi.t1%1)<0.1:
        print "time = ",mvi.t1
                        
pylab.plot(T,Q) 
pylab.title("Scooter")
pylab.ylabel("q(meters/radians)")
pylab.legend(["Body","Hip","Knee"])
pylab.show()
# Visualize the system in action
trep.visual.visualize_3d([ trep.visual.VisualItem3D(system, T, Q) ])
                        
                
            

