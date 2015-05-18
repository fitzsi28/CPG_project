import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from pylab import * 
from sympy.plotting import plot_parametric

def limb(Y, t):
	d = 0.1 #damping ratio
	u = 0 	#externally applied torque
	tau = 1.2
	k = 1
	o = 0.510
	e = 0.731

	return [Y[1], u-2*d*Y[1]-Y[0]]


def main():

	tau = 1.2 	# half-period
	n = 1 		# 

	ts = 0 		# start time
	te = n*tau	# end time
	Tsim = 3.6 	# Total simulation time

	numsteps = 1000 	#

	a = 0.3 	#
	o = 0.510 	# speed
	e = 0.731 	# speed retained

	theta = []			# array of joint angles
	thetaDot = []		  #array of joint velocities

	# Initial swing
	ts = 0
	te = tau
	time1 = np.linspace(ts, te, numsteps)
	y0 = np.array([a, -o])
	y1 = integrate.odeint(limb, y0, time1)	
	theta = np.append(theta, y1[:,0])
	thetaDot = np.append(thetaDot, y1[:, 1])

	# First Impulse
	theta = np.append(theta, -a)
	thetaDot = np.append(thetaDot, -o*e)

	# Second swing
	ts = tau
	te = 2*tau
	time3 = np.linspace(ts, te, numsteps)
	y0 = np.array([-a, o])
	y3 = integrate.odeint(limb, y0, time3)
	theta = np.append(theta, y3[:,0])
	thetaDot = np.append(thetaDot, y3[:, 1])

	# Second Impulse
	theta = np.append(theta, a)
	thetaDot = np.append(thetaDot, -o)

	plot(theta, thetaDot)

	axis('equal')
	grid('on')

	title(" Phase Plane Limit Cycle ")
	xlabel(" theta ")
	ylabel(" theta dot ")
	show()

	# fig.savefig('phase_plane_limit_cycle.png')

if __name__ == '__main__':
	main()




