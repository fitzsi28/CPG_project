import pylab
import math
import numpy as np

t = 0
dt = 0.001
tf = 1.0

Aw = math.pi/3
An = 1.0 #math.pi/3 #(math.pi/2.0)
def whiskpath(t):
    q = np.array([0.,0.,0.])
    q[0] = math.sin(t)
    q[1] = Aw*math.sin(16.0*math.pi*t)
    q[2] = -Aw*math.sin(16.0*math.pi*t)
    return q

T = [t]
Q = [whiskpath(t)]

while t < tf:
   #print "sin(",t,") : ",  math.sin(16*math.pi*t)
   T.append(t)
   Q.append(whiskpath(t))
   #print "whisk", q
   t = t+dt 

pylab.plot(T,Q) 
pylab.title("Sin(t)")
pylab.ylabel("q(radians)")
pylab.legend(["Neck","Left Whisker","Right Whisker"])
pylab.show()

