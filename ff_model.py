import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from pylab import * 
from sympy.plotting import plot_parametric

def limb(Y, t):
	d = 0.1 #damping ratio
	u = 0 	#externally applied torque

	return [Y[1], u-2*d*Y[1]-Y[0]]


def main():

	# First swing
	ts = 0 	#start time
	te = 1.2 	# end time (half-period)
	numsteps = 1000
	time1 = np.linspace(ts, te, numsteps)
	a = 0.3 #
	o = 0.510 # speed
	e = 0.731 # speed retained
	y0 = np.array([-a, -o*e])
	y1 = integrate.odeint(limb, y0, time1)


	# Second swing
	ts = 1.2
	te = 2.4
	time2 = np.linspace(ts, te, numsteps)
	y0 = np.array([-a, o])
	y2 = integrate.odeint(limb, y0, time2)


	# Back
	ts = 2.4
	te = 3.6
	time3 = np.linspace(ts, te, numsteps)
	y0 = np.array([a, -o])
	y3 = integrate.odeint(limb, y0, time3)

	print(y1)
	print(y2)
	print(y3)


	subplot(311)
	plot(y1[:,1],y1[:,0])
	#pylab.plot_parametric((y1[:,1],y1[:,0], (t, 0, 1.2))
	axis('equal')
	grid('on')
	
	subplot(312)
	plot(y2[:,1],y2[:,0])
	axis('equal')
	grid('on')
	
	subplot(313)
	plot(y3[:,1],y3[:,0])
	axis('equal')
	grid('on')

	title(" Phase Plane Limit Cycle ")
	xlabel(" theta ")
	ylabel(" theta dot ")
	show()

	# fig.savefig('phase_plane_limit_cycle.png')

if __name__ == '__main__':
	main()




