"""Working file for the draft paper "A Simple Economic Model with Interactions"
by Gusev and Kroujiline on 07/04/2020"""

import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-15,15))

class SolowModel(object):
	def __init__(self,alpha,tau_y,epsilon,delta,lam):
		"""Numerical approach to the second order differential in eq. 5
		requires rewriting via x[0] = K, x[1] = K', thus converting the
		problem into a first-order differential equation

		Parameters
		----------
		@param alpha 		- 	float of the capital share of production
		@param tau_y 		- 	float of the characteristic timescale
		@param epsilon 		- 	float of the technology growth constant
		@param delta 		- 	float of the depreciation rate
		@param lam 			- 	float of the savings rate

		Returns
		----------
		@return SolowModel	- 	Instance for
		"""

		self.alpha = alpha
		self.tau_y = tau_y
		self.epsilon = epsilon
		self.delta = delta
		self.lam = lam
		self.args = (alpha,tau_y,epsilon,lam,delta)

	@staticmethod
	def classicApproximateSolution(alpha,tau_y,epsilon,lam,delta,a,t_points):
		"""Implementation of equation 6 - the theoretical approximate solution
		for the output path in the classic regime limiting case

		Parameters
		----------
		@param alpha 	- 	float of the capital share of production
		@param tau_y 	- 	float of the characteristic timescale
		@param epsilon 	- 	float of the technology growth constant
		@param lam 		- 	float of the savings rate
		@param delta 	- 	float of the depreciation rate
		@param a 		- 	float constant of integration
		@param t_points - 	(tx1) np.array of points in time to evaluate output

		Returns
		----------
		@return y_t 	- 	(tx1) np.array of output at each time in t_points
		"""

		constant = (lam/delta)**(alpha/(1-alpha))
		lt_growth_path = np.exp((epsilon/(1-alpha))*t_points)
		temp = a*np.exp(-1*((1-alpha)/tau_y)*t_points)
		relaxation_path = (temp+1)**(1/(1-alpha))
		return constant*(relaxation_path+lt_growth_path-1)

	@staticmethod
	def classicSecondOrder(X,t,alpha,tau_y,epsilon,lam,delta):
		"""Numerical approach to the second order differential in eq. 5
		requires rewriting via x[0] = K, x[1] = K', thus converting the
		problem into a first-order differential equation

		Parameters
		----------
		@param X 		- 	K and K' starting points
		@param t 		- 	float of the current time
		@param alpha 	- 	float of the capital share of production
		@param tau_y 	- 	float of the characteristic timescale
		@param epsilon 	- 	float of the technology growth constant
		@param lam 		- 	float of the savings rate
		@param delta 	- 	float of the depreciation rate

		Returns
		----------
		@return K 		- 	K and K' at time t
		"""
		y0 = X[1] # By definition of x[0] and x[1]
		# Rearrange equation 5
		first_deriv = -1*(1+tau_y*delta)*X[1]
		current_val = lam*np.exp(epsilon*t)*(X[0]**alpha)-delta*X[0]
		y1 = (1/tau_y)*(first_deriv + current_val)
		return [y0,y1]

	def classicModelAnalysis(self,t_0,t_end,count,save=None):
		"""Function to analyse and graph the trajectories und er the classic regime
		where capital supply equals demand

		Parameters
		----------
		None

		Returns
		----------
		None
		"""

		# Generate period
		t_points = np.linspace(t_0,t_end,num=count,endpoint=True)
		print(t_points.shape)

		# Generate approximate solution
		approx = self.classicApproximateSolution(
			self.alpha,
			self.tau_y,
			self.epsilon,
			self.lam,
			self.delta,
			1,
			t_points)
		print(approx.shape)

		# Generate the numerical solution
		origin = (5,0)
		numerical = odeint(self.classicSecondOrder,origin,t_points,self.args)
		print(numerical.shape)

		# Generate a graph of the solutions
		fig,ax = plt.subplots(1,1)
		fig.set_size_inches((6,6))
		ax.plot(t_points,numerical[:,0],label='Numerical',color='Blue')
		ax.plot(t_points,approx,label='Approximate',color='Orange')
		ax.set_xlabel('t')
		ax.set_ylabel('Y(t)')
		ax.set_xlim(t_0,t_end)
		ax.set_title("Classical Regime Path")
		plt.legend(loc='upper center',ncol=4)
		fig.tight_layout()
		if save is not None: plt.savefig(filename=save)
		plt.show()


	def cycleRegimeSystem(X,t,c1,c2,s_star,omega_y,epsilon,tau_s,beta1,beta2,tau_h,noise):
		"""Function of the dynamical system of equations 10a-c that represent

		Parameters
		----------
		@param X 		- 	3x1 vector of starting points (z,s,h)
		@param t 		- 	float of time t at which to evaluate
		@param c 		- 	float
		@param c1		-	float
		@param c2		-	float
		@param s_star	-	float
		@param omega_y	-	float equal to 1/tau_y, char. timescale of output
		@param epsilon	-	float of the
		@param tau_s	-	flaot characteristic timescale of sentiment
		@param beta1	- 	float
		@param beta2	- 	float
		@param tau_h	- 	float characteristic timescale of analysts
		@param noise	- 	vector of noise parameters, noise.shape[0]>t+1

		Returns
		----------
		@return Y 		- 	list of the values (z',s',h') at at time t
		"""

		constant = c*np.exp(X[0])-1
		s_prime = (np.tanh(beta1*X[1]+beta2*X[2])-X[1])/tau_s
		h_prime = (np.tanh((gamma*omega_y*constant)+noise[t]) - X[2])/tau_h
		z_prime = c1*s_prime + c2*(X[1]-s_star) - omega_y*constant + epsilon
		return [z_prime,s_prime,h_prime]


	def cycleRegimeSolution():
		pass

	def cycleRegimeVisualisation():
		pass




args = {
	'alpha':0.5,
	'tau_y':1e3,
	'epsilon':1e-5,
	'delta':0.5,
	'lam':0.5}

sm = SolowModel(**args)
sm.classicModelAnalysis(t_0=0,t_end=120000,count=1000)
