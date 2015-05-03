#Trep and SAC simulation of walker
#Authors: Katie Fitzsimons, Maddy ??

import numpy as np
import trep
from trep import tx,ty,tz,rx,ry,rz

g = 9.81
MH = 0.05 #kg
MW = 10^-7 #kg
Hp = 0.025 #m
d = 0.01 #m
dw = 0.005 #m
LEFTWHISK = "Left whisker"
RIGHTWHISK = "Right whisker"


system = trep.System()
frames = [#1
    rz('theta', name ="neck", kinematic = True),[#2
        ty(Hp, name = 'head', mass = MH),[#3
            tx(-d,name = "Lcheek"),[
                rz('phiL',name = LEFTWHISK, kinematic = True),[
                    tx(-dw, name = "Lwhisk", mass = MW)]],
            tx(d,name = "Rcheek"),[
                rz('phiR',name = RIGHTWHISK, kinematic = True),[
                    tx(dw, name = "Rwhisk", mass = MW)]]]]]
system.import_frames(frames)

print "configs",system.configs
                        
                
            

