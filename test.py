# Trep implemention of the feedforward cotrol system for rythmic movements as described in Kuo et. al 2002
# Authors: Mahdieh Nejati and Katie Fitzsimons

import numpy as np
import trep
from trep import tx,ty,tz,rx,ry,rz
from pylab import *
from numpy import dot
import trep.discopt


## System parameters: 

m = 1.0 # Mass of pendulum
l = 1.0 # Length of pendulum
q0 = 0.3 # Initial angle of pendulum
dq0 = -0.510 # Initial velocity of pendulum
t0 = 0.0 # Initial time
tf = 5.0 # Final time
dt = 0.1 # Timestep
B = 0.1 # Damping
g = -9.8

system = trep.System() # Initialize system

frames = [
    rx('theta', name="pendulumShoulder"), [
        tz(-l, name="pendulumArm", mass=m)]]
system.import_frames(frames) # Add frames

## Add forces to the system
trep.potentials.Gravity(system, (0, 0, g)) # Add gravity
trep.forces.Damping(system, B) # Add damping
trep.forces.ConfigForce(system, 'theta', 'theta-torque') # Add input (externally applied torque)

## Create and initialize the variational integrator
mvi = trep.MidpointVI(system)
mvi.initialize_from_configs(t0, np.array([q0]), t0+dt, np.array([q0]))

## Simulate system forward in time
T = [mvi.t1] # List to hold time values
Q = [mvi.q1] # List to hold configuration values
dQ = [mvi.q2_dq1]
while mvi.t1 < tf:
    mvi.step(mvi.t2+dt, [0.0]) # Step the system forward by one time step
    T.append(mvi.t1)
    Q.append(mvi.q1)
    dQ.append(mvi.q2)
    print "Q: ", mvi.q1
    print "dQ: ", mvi.q2

## Phase Plane Limit Cycel
# plot(Q, dQ)
# ylabel(" dtheta ")
# xlabel(" theta ")
# show()


## Visualize the system in action
trep.visual.visualize_3d([ trep.visual.VisualItem3D(system, T, Q) ])