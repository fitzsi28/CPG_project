# Trep implemention of the feedforward cotrol system for rythmic movements as described in Kuo et. al 2002
# Authors: Mahdieh Nejati and Katie Fitzsimons

import numpy as np
import trep
from trep import tx,ty,tz,rx,ry,rz
from pylab import * 


# System Parameters

tf=10.0 	# Final time
dt = 0.01 	# Timestep
tc = 0.0 	# ???

g = 9.81 	# acceleration due to gravity
B=0.002  	#damping

MH = 0.5 	# Mass of Hip (kg)
MUL =0.2 	# Mass of upper leg (kg)
MLL = 0.1 	# Mass of lower leg (kg)
UL = 0.25 	# Length of upper leg (m)
LL = 0.25 	# Length of lower leg (m)

HIP = "hip joint"
ULEG = "UpperLeg"
LLEG = "LowerLeg"

system = trep.System() 			# Initialize system

frames = [#1
    ty('yb', name ="body", mass = MH),[#2
        rx('theta1', name = HIP,),[#3
            tz(-UL ,name = ULEG, mass = MUL),[
                rx('theta2', name = 'knee'),[
                    tz(-LL, name = LLEG, mass = MLL)]]]]]

system.import_frames(frames) 	# Add frames

# Add forces to the system
trep.potentials.Gravity(system,(0,0,-g))	# Add gravity
trep.forces.Damping(system,B)				# Add damping
trep.forces.ConfigForce(system, 'theta1','hip-torque') # Add input
