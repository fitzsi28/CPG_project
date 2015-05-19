# Implementing the feedforward cotrol system for rythmic movements as described in Kuo et. al 2002
# Authors: Mahdieh Nejati and Katie Fitzsimons

import numpy as np
from scipy import integrate
from pylab import * 

# Numercial Differential Equation Sovler 
def limb(Y, t):
	d = 0.1 #damping ratio
	u = 0.05 	#externally applied torque
	tau = 1.2
	o = 0.510
	e = 0.731

	return [Y[1], u-2*d*Y[1]-Y[0]]


def main():

	tau = 1.2 	# half-period

	ts = 0 		# start time
	te = tau	# end time
	Tsim = 10*tau	# Total simulation time

	numsteps = 1000 	# timesteps to solve DE 

	a = 0.3 	# angle
	o = 0.510 	# speed
	e = 0.731 	# speed retained

	theta = []			# array of joint angles
	thetaDot = []		  #array of joint velocities

	# Initial Conditions: 
	angle = a
	velocity = -o

	while te <= Tsim:

		# First Half-Cycle: 
		time = np.linspace(ts, te, numsteps)
		y0 = np.array([angle, velocity])
		y = integrate.odeint(limb, y0, time)
		theta = np.append(theta, y[:,0])
		thetaDot = np.append(thetaDot, y[:, 1])	

		# Imulse: 
		impulseAng= len(theta)-1
		theta = np.append(theta, theta[impulseAng])
		nextVel = len(thetaDot)-1
		impulseVel = thetaDot[nextVel] + o*(1+e)
		thetaDot = np.append(thetaDot, impulseVel)

		# Setting up conditions for next half-cycle
		ts = ts + tau
		te = te + tau	

		lenAng = len(theta)-1
		lenVel = len(thetaDot)-1

		angle = theta[lenAng]
		velocity = thetaDot[lenVel]

		# Second Half-Cycle: 
		time = np.linspace(ts, te, numsteps)
		y0 = np.array([angle, velocity])
		y = integrate.odeint(limb, y0, time)
		theta = np.append(theta, y[:,0])
		thetaDot = np.append(thetaDot, y[:, 1])	

		# Impulse: 
		impulseAng= len(theta)-1
		theta = np.append(theta, theta[impulseAng])
		nextVel = len(thetaDot)-1
		impulseVel = thetaDot[nextVel] - o*(1+e)
		thetaDot = np.append(thetaDot, impulseVel)

		# Setting up conditions for next half-cycle
		ts = ts + tau
		te = te + tau	

		lenAng = len(theta)-1
		lenVel = len(thetaDot)-1

		angle = theta[lenAng]
		velocity = thetaDot[lenVel]


# Plot of results: 
	plot(theta, thetaDot)

	axis('equal')
	grid('on')
	# title(" Phase Plane Limit Cycle ")
	title(" feedforward Disturbance Response ")
	xlabel(" theta ")
	ylabel(" theta dot ")
	show()

	# fig.savefig('phase_plane_limit_cycle.png')

if __name__ == '__main__':
	main()
