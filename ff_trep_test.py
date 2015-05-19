# Trep implemention of the feedforward cotrol system for rythmic movements as described in Kuo et. al 2002
# Authors: Mahdieh Nejati and Katie Fitzsimons

import numpy as np
import trep
from trep import tx,ty,tz,rx,ry,rz
from pylab import *
from numpy import dot
import trep.discopt


## System Parameters

ts = 0.0 	# Start time
tf = 10.0 	# Final time
dt = 0.01 	# Timestep
<<<<<<< HEAD
tc = 0.0 	# Current time
=======
tc = 0.0 	# t_calc for SAC
>>>>>>> c2156ba6d4624e5886ed638ec990b70a11e89f9e

g = 9.81 	# Acceleration due to gravity
B = 0.01  	# Damping

<<<<<<< HEAD
MH = 0.5 	# Mass of Hip (kg)
MUL = 0.2 	# Mass of upper leg (kg)
MLL = 0.1 	# Mass of lower leg (kg)
UL = 0.25 	# Length of upper leg (m)
LL = 0.25 	# Length of lower leg (m)
=======
M = 0.5 	# Mass of Hip (kg)
L = 0.25 	# Length of upper leg (m)
>>>>>>> c2156ba6d4624e5886ed638ec990b70a11e89f9e

HIP = "hip joint"
LEG = "Leg"

system = trep.System() 			# Initialize system

frames = [#1
    rx('theta1', name = HIP,),[#2
        tz(-L ,name = LEG, mass = M)]]

system.import_frames(frames) 	# Add frames

## Add forces to the system
trep.potentials.Gravity(system,(0,0,-g))	# Add gravity
trep.forces.Damping(system,B)				# Add damping
trep.forces.ConfigForce(system, 'theta1','hip-torque') # Add input


## Create and initialize the variational integrator

# Set up configuration 
q0= np.array([0,0,0.3]) 	# Initial positions
dq0 = np.array([0,0.354,0.354])	# Initial velocities

mvi=trep.MidpointVI(system)
mvi.initialize_from_configs(ts, q0, dt, q0)


## Simulate system forward in time
T = [mvi.t1] 	# List to hold time values
Q = [system.q[2]]	# List to hold configuration values
dQ = [system.dq[2]] # List to hold 
# print system.dq

while mvi.t1 < tf:
    tc = tc+dt
    mvi.step(mvi.t2+dt, [0.0]) 	# Step the system forward by one time step
    T.append(mvi.t1)
    Q.append(mvi.q1[2])
    dQ.append(mvi.q2[2])

    # if np.abs(mvi.t1%1)<0.1:
    #     print "time = ",mvi.t1

# Plot                        
# plot(T,Q)
plot(Q, dQ)
# title("Scooter")
ylabel(" dtheta ")
xlabel(" theta ")
# legend(["Body","Hip","Knee"])
show()

## Visualize the system in action
# trep.visual.visualize_3d([ trep.visual.VisualItem3D(system, T, Q)])
