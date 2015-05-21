# Trep implemention of the feedforward cotrol system for rythmic movements as described in Kuo et. al 2002
# Authors: Mahdieh Nejati and Katie Fitzsimons

import numpy as np
import trep
from trep import tx,ty,tz,rx,ry,rz
from pylab import *
from numpy import dot
import trep.discopt


## System parameters: 

m = 12 # Mass of pendulum
l = 1.0 # Length of pendulum

t0 = 0.0 # Initial time
tf = 10.0 # Final time
dt = 0.01 # Timestep
B = 8.4 # Damping coefficient
g = 9.81 #potential due to gravity and springs


#From KUO
tau = 0.398 #time constant from KUO
q0 = 0.3 # Initial angle of pendulum
dq0 = -0.510*np.sqrt(g/l) # Initial velocity of pendulum
#print 1.2/np.sqrt(g/l)
u = 3195.0#impulse force


system = trep.System() # Initialize system

frames = [
    rx('theta', name="pendulumShoulder"), [
        tz(-l, name="pendulumArm", mass=m)]]
system.import_frames(frames) # Add frames

## Add forces to the system
trep.potentials.Gravity(system, (0, 0, -g)) # Add gravity
trep.forces.Damping(system, B) # Add damping
trep.forces.ConfigForce(system, 'theta', 'theta-torque') # Add input (externally applied torque)

system.q = q0
system.dq = dq0
## Create and initialize the variational integrator
mvi = trep.MidpointVI(system)
#mvi.initialize_from_configs(t0, np.array([q0]), t0+dt, np.array([q0+dq0*dt]))
mvi.initialize_from_state(t0,np.array([q0]),np.array([m*l*dq0]))

prev = 0
#feed-forward force function
def FF(t,prev):
    time = t-prev
    if(time>(tau-0.005) and time<(tau+0.005)):
       tor=u
    elif(time>(2*tau-0.005) and time<(2*tau+0.005)):
       tor=-1*u
    elif(time>(3*tau-0.005) and time<(3*tau+0.005)):
       tor=u
    elif(time>(4*tau-0.005) and time<(4*tau+0.005)):
       tor=-1*u
       prev = t-dt
    else:
       tor = 0
    return[tor, prev]

## Simulate system forward in time
T = [mvi.t1] # List to hold time values
Q = [system.q] # List to hold configuration values
dQ = [system.dq] # List to hold velocities 
U = [0.0]

while mvi.t1 < tf:
    torque = np.zeros(system.nu)
    control_val = FF(mvi.t2+dt,prev)
    torque[0] = control_val[0] #+SAC <--plug this into mvi where [0.0] is
    prev = control_val[1]
    #mvi.step(mvi.t2+dt, [0.0]) # no control
    mvi.step(mvi.t2+dt, u1 = torque) # Step the system forward by one time step
       
    T.append(mvi.t1)
    Q.append(mvi.q1)
    dQ.append(system.dq)
    U.append(torque)


# Plot results
ax1 = subplot(211,autoscale_on = False,xlim =[-0.4,0.4],ylim =[-1.75,1.75])
plot(Q, dQ)
title("FeedForward Controller")
ylabel("dtheta")
xlabel("theta")
legend(["qd","p"])
subplot(212,autoscale_on=False,xlim=[0,tf],ylim=[-1.75,1.75])
plot(T,Q)
plot(T,dQ)
plot(T, U)
xlabel("T")
legend(["Q","dQ","U"])
show()

## Visualize the system in action
trep.visual.visualize_3d([ trep.visual.VisualItem3D(system, T, Q) ])


