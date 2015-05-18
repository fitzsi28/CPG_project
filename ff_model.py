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

	if t == (2*k-1)*tau:
		u = o*(1+e)

	if t == 2*k*tau: 
		u = -o*(1+e)

	return [Y[1], u-2*d*Y[1]-Y[0]]


def main():

	tau = 1.2 	# half-period
	n = 1 		# 

	ts = 0 		# start time
	te = n*tau	# end time
	T = 3.6 	# total sim time

	numsteps = 1000 	#

	a = 0.3 	#
	o = 0.510 	# speed
	e = 0.731 	# speed retained

	theta = []			# array of joint angles
	thetaDot = []		  #array of joint velocities

	# while te <= T:  
	# 	if ts == (n-1)*tau: # first half-peirod
	# 		# First swing
	# 		# te = n*tau	# end time (half-period)

	# 		time1 = np.linspace(ts, te, numsteps)
	# 		# y0 = np.array([-a, -o*e])
	# 		# y1 = integrate.odeint(limb, y0, time1)
	# 		# theta.append(y1[:,1])
	# 		# thetaDot.append(y1[:,0])

	# 		time2 = np.linspace(ts, te, numsteps)
	# 		y0 = np.array([-a, o])
	# 		y2 = integrate.odeint(limb, y0, time2)
	# 		theta = theta + y2[:,1]
	# 		thetaDot = thetaDot + y2[:, 0]

	# 		ts = te 
	# 		n = n+1
	# 		te = n*tau

	# 	if ts == n*tau:

	# 		# time2 = np.linspace(ts, te, numsteps)
	# 		# y0 = np.array([-a, o])
	# 		# y2 = integrate.odeint(limb, y0, time2)
	# 		# theta.append(y2[:,1])
	# 		# thetaDot.append(y2[:,0])

	# 		time3 = np.linspace(ts, te, numsteps)
	# 		y0 = np.array([a, -o])
	# 		y3 = integrate.odeint(limb, y0, time3)
	# 		theta = theta + y3[:,1]
	# 		thetaDot = thetaDot + y3[:, 0]
	# 		ts = te 
	# 		n = n+1
	# 		te = n*tau


	# Second swing
	ts = 1.2
	te = 2.4
	time2 = np.linspace(ts, te, numsteps)
	y0 = np.array([a, -o])
	y2 = integrate.odeint(limb, y0, time2)


	# Back
	ts = 2.4
	te = 3.6
	time3 = np.linspace(ts, te, numsteps)
	y0 = np.array([-a, o])
	y3 = integrate.odeint(limb, y0, time3)

	# print(y1)
	print("y2=", y2[:,0])
	print("y3=", y3)

	theta = np.append(y2[:,0], y3[:,0])
	thetaDot = np.append(y2[:,1], y3[:,1])


	# plot(y2[:,0], y2[:,1])
	# plot(y3[:,0],y3[:,1])
	# print "theta", theta
	# plot(thetaDot, theta)
	plot(thetaDot, theta)
	axis('equal')
	grid('on')

	title(" Phase Plane Limit Cycle ")
	xlabel(" theta ")
	ylabel(" theta dot ")
	show()

	# fig.savefig('phase_plane_limit_cycle.png')

if __name__ == '__main__':
	main()




