#Trep and SAC simulation of walker
#Authors: Katie Fitzsimons, Mahdieh

import numpy as np
import trep
from trep import tx,ty,tz,rx,ry,rz

g = 9.81
MH = 25.0 #kg
MUL = 15 #kg
MLL = 8 #kg
L = 0.25 #m
HIPMASS = "hip"
RULEG = "rightUpperLeg"
RLLEG = "rightLowerLeg"
LULEG = "leftUpperLeg"
LLLEG = "leftLowerLeg"


system = trep.System()
frames = [#1
    tx('xh', name ="xhip"),[#2
        ty('yh', name = HIPMASS, mass = MH),[#3
            rz('thetahR',name = "hipJoint"),[
                tx(-L,name = RULEG,mass = MUL),[
                    rz('thetakr', name = "RkneeJoint"),[
                        tx(-L,name = RLLEG, mass = MLL)]]],
            rz('thetahL',name = "LhipJoint"),[
                tx(-L,name = LULEG,mass = MUL),[
                    rz('thetakl', name = "LkneeJoint"),[
                        tx(-L,name = LLLEG, mass = MLL)]]]]]]
system.import_frames(frames)

print "configs",system.configs
                        
                
            

