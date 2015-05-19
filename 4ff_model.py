import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from pylab import * 
from sympy.plotting import plot_parametric

def limb(Y, t):
	d = 0.1 #damping ratio
	u = 0 	#externally applied torque
	tau = 1.2
	o = 0.510
	e = 0.731

	return [Y[1], u-2*d*Y[1]-Y[0]]


def main():

	tau = 1.2 	# half-period
	k = 1 		# 

	ts = 0 		# start time
	te = tau	# end time
	Tsim = 2.4	# Total simulation time

	numsteps = 1000 	#

	a = 0.3 	#
	o = 0.510 	# speed
	e = 0.731 	# speed retained

	theta = []			# array of joint angles
	thetaDot = []		  #array of joint velocities

	velocity = o

	# Initial 
	time = np.linspace(ts, te, numsteps)
	angle = a
	velocity = -o
	y0 = np.array([angle, velocity])
	y = integrate.odeint(limb, y0, time)	
	theta = np.append(theta, y[:,0])
	thetaDot = np.append(thetaDot, y[:, 1])

	ts = ts + tau
	te = te + tau

	while te <= Tsim:

		if ts == (2*k-1)*tau: 

			# First half cycle

			ts = ts + tau
			te = te + tau

			angle = -a

			time = np.linspace(ts, te, numsteps)
			# velocity = velocity +(o*(1+e))
			velocity = o
			y0 = np.array([angle, velocity])
			y = integrate.odeint(limb, y0, time)
			theta = np.append(theta, y[:,0])
			thetaDot = np.append(thetaDot, y[:, 1])

		if ts == 2*k*tau: 

			# Second half cycle 

			ts = ts + tau
			te = te + tau

			angle= a

			time = np.linspace(ts, te, numsteps)
			velocity = velocity + (-o*(1+e))
			y0 = np.array([angle, velocity])
			y = integrate.odeint(limb, y0, time)
			theta = np.append(theta, y[:,0])
			thetaDot = np.append(thetaDot, y[:, 1])

			k=k+1

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




