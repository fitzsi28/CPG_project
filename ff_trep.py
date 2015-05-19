# Trep implemention of the feedforward cotrol system for rythmic movements as described in Kuo et. al 2002
# Authors: Mahdieh Nejati and Katie Fitzsimons

import numpy as np
import trep
from trep import tx,ty,tz,rx,ry,rz
from pylab import * 


# System Parameters

tf=10.0 	# Final time
dt = 0.01 	# Timestep
tc = 0.0 	# t_calc for SAC

g = 9.81 	# acceleration due to gravity
B=0.002  	#damping

M = 0.5 	# Mass of Hip (kg)
L = 0.25 	# Length of upper leg (m)

HIP = "hip joint"
LEG = "Leg"

system = trep.System() 			# Initialize system

frames = [#1
    rx('theta1', name = HIP,),[#2
        tz(-L ,name = LEG, mass = M)]]

system.import_frames(frames) 	# Add frames

# Add forces to the system
trep.potentials.Gravity(system,(0,0,-g))	# Add gravity
trep.forces.Damping(system,B)				# Add damping
trep.forces.ConfigForce(system, 'theta1','hip-torque') # Add input