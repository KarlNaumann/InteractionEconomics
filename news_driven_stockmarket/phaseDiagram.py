import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from matplotlib import pyplot as plt

"""Functional setup for the main phase diagram"""

class sentimentProcess(object):
	def __init__(self,omega_s=0.04,beta1=1.1,beta2=0.55,omega_h=0.4,gamma=56.0,delta=0.03):
		"""Class object that contains the process for the sentiment s and h.
		Described in equations 13b-13c of M. Gusev et al. "Predictable Markets?..."

		Parameters
		------------
		omega_s -	float constant (eq. 13b, p. 23) - default 0.04 (fig. 13 p. 25)
		beta1 	-	float constant (eq. 13b, p. 23) - default 1.1 (fig. 13 p. 25)
		beta2	-	float constant (eq. 13b, p. 23) - default 0.55 (fig. 13 p. 25)
		omega_h -	float constant (eq. 13c, p. 23) - default 0.4 (fig. 13 p. 25)
		gamma 	-	float constant (eq. 13c, p. 23) - default 56.0 (fig. 13 p. 25)
		delta 	-	float constant (eq. 13c, p. 23) - default 0.03 (fig. 13 p. 25)

		Returns
		------------
		sentimenProcess Object
		"""

		# Save initiated values to be accessible in class
		self.omega_s = omega_s
		self.omega_h = omega_h
		self.beta1 = beta1
		self.beta2 = beta2
		self.gamma = gamma
		self.delta = delta
		self.args = (omega_s,omega_h,beta1,beta2,gamma,delta)

		# Determine the systems equilibria
		self.solutions = self.findSolutions()
		self.solutions = self.analyseSolution(self.solutions)

		self.saddles = [s for s in self.solutions if self.solutions[s]['type']=='saddle']

		self.trajectorySeparatrices,self.sep_origins,self.sep_gradients = self.separatrices()

		# Generate some perturbations near the non-saddle points to evaluate how stable the nodes/focii are
		distances = [-1e-6,1e-6]
		for point in self.solutions:
			if self.solutions[point]['type']!='saddle':
				self.sep_origins.extend([(point[0]+d,point[1]) for d in distances])
				self.sep_origins.extend([(point[0],point[1]+d) for d in distances])

	@staticmethod
	def velocity(X,t,omega_s,omega_h,beta1,beta2,gamma,delta):
		""""Static function to calculate the velocity vector at a given point
			X=(s,h) and constants defining the system

		Parameters
		------------
		X 	-	tuple (s,h) where s,h are floats
		t 	-	placeholder for the time period (necessary for odeint to solve the system)
		omega_s,omega_h,beta1,beta2,gamma,delta - float constants

		Returns
		------------
		vs 	-	float of velocity of s
		vh 	-	float of veloctity of h
		"""
		s,h = X
		vs = -omega_s*s + omega_s*np.tanh(beta1*s+beta2*h)
		vh = -omega_h*h + omega_h*np.tanh(gamma*vs + delta)
		return [vs,vh]

	def findSolutions(self):
		""" We check the conditions for the equilibria based on the visual method in appendix
		C - i.e. find the solutions s to equation C3

		Parameters
		----------
		None

		Returns
		----------
		solutions 	- 	(list) of tuples with solution coordinates"""

		# H solution is determined
		h_star = np.tanh(self.delta)

		# Define function f(beta1,s)
		f = lambda s: (0.5*np.log((1+s)/(1-s))-self.beta1*s)

		# Define intersect function to minimize on different domains and find intersect
		# INput is 2d array s.t. minimizer works, but second value is always 0, no effect on estimation
		intersect = lambda x: np.sqrt((self.beta2*h_star - f(x[0]))**2)

		if self.beta1<=1: # Paramagnetic case - 1 solution
			s_plus = minimize(intersect,x0=(0.5,0),bounds=((-1,1),(None,None))).x[0]
			solutions = [(s_plus,h_star)]

		else: # Ferromagnetic case - 1 or 3 solutions
			# Determine extremum
			s_ext = -np.sqrt((self.beta1-1)/self.beta1)
			f_ext = f(s_ext)

			if f_ext > (self.beta2*h_star): # Three solutions
				# Minimize the intersect function to determine solution s
				s_neg = minimize(intersect,x0=(s_ext-0.01,0),bounds=((-1,s_ext),(None,None))).x[0]
				s_0 = minimize(intersect,x0=(s_ext+0.01,0),bounds=((s_ext,0),(None,None))).x[0]
				s_plus = minimize(intersect,x0=(0.5,0),bounds=((0,1),(None,None))).x[0]
				solutions = [(s_neg,h_star),(s_0,h_star),(s_plus,h_star)]

			else: # One solution case
				s_plus = minimize(intersect,x0=(0.5,0),bounds=((0,1),(None,None))).x[0]
				solutions = [(s_plus,h_star)]

		return solutions

	def analyseSolution(self,solutions):
		"""Categorize the solutions by type and save relevant information such as the eigenvalues and
		eigenvectors of the jacobian

		Parameters
		----------
		solutions 	- 	(list) of tuples with solution coordinates

		Returns
		----------
		result 		- 	(dict) of. the jacobian, eigenvalues of the jacobian, and eigen vectors of each
						solution. It is indexed by the tuple of solution coordinates.

		"""
		result = {}

		# Relevant constants
		eta = self.omega_h/self.omega_s
		g_bar = self.omega_s*self.gamma

		for point in solutions:
			s = point[0]
			# Determine the Jacobian
			psi = 1-self.beta1*(1-s**2)
			chi = eta*g_bar*((1/np.cosh(self.delta))**2)
			phi = self.beta2*eta*g_bar*(1-s**2)*((1/np.cosh(self.delta))**2) - eta
			jacobian = np.array([[-psi,(phi+eta)/chi],[-chi*psi,phi]])

			# Find the eigenvalues and eigenvectors
			eig_val,eig_vec = np.linalg.eig(jacobian)
			discriminant = (phi-psi)**2 - 4*psi*eta

			# Save calculations first
			result[point] ={'jac':jacobian,'eig_val':eig_val,'eig_vec':eig_vec}

			#Categorize the points
			if self.beta1*(1-s**2)>1: # Saddle point
				result[point]['type'] = 'saddle'
				result[point]['stable'] = False
			else:
				if discriminant>0: # node
					result[point]['type'] = 'node'
					if phi-psi<0: result[point]['stable'] = True
					else:result[point]['stable'] = False
				else: # focus
					result[point]['type'] = 'focus'
					if phi-psi<0: result[point]['stable'] = True
					else:result[point]['stable'] = False

		return result

	def separatrices(self,t_end=1e4,freq=1e5,epsilon=1e-7,min_dist=1e-5,count=8):
		"""Determine, numercially, the hyperbola that represents the separatrix. In order to
		do so perturb the saddle point (x0,y0) via e=5e-6 i.e. (x0+e,y0) and (x0,y0+e).

		Parameters
		----------
		t_end 	-	endpoint of the timespan (2 x t_end for backward)
		freq 	- 	number of points at which to evaluate in the timespan
		epsilon -	perturbance to the saddle point to generate separatrix
		min_dist -	(float) minimum distance for the gradient evaluation
		count 	-	number of origin points on separatrix to generate for diagram

		Returns
		----------
		separatrices 	- 	list of (2xt_end) vectors of (s,h) for the separatrices
		"""

		#Parameters - larger timespan backward as gradient is slow to increase
		fwd_timespan = np.linspace(0,t_end,freq)
		bwd_timespan = -1 * np.linspace(0,2*t_end,2*freq)

		#Result dictionaries indexed by the saddle points
		paths = {}
		gradients = {}
		origins = []

		for saddle_point in self.saddles:
			###########################################################################################
			# Perturb origins slightly to generate non-stationary starting points
			perturbed_origins = [
				(saddle_point[0]+epsilon,saddle_point[1]),
				(saddle_point[0]-epsilon,saddle_point[1])]#,
				#(saddle_point[0],saddle_point[1]+epsilon),
				#(saddle_point[0],saddle_point[1]-epsilon)]

			# Estimate the separatrix path by integrating forward and backward
			saddle_paths=[]
			for point in perturbed_origins:
				for span in [fwd_timespan,bwd_timespan]:
					saddle_paths.append(odeint(self.velocity,point,span,self.args,hmax=0.5))

			# Save separatrices for this saddle
			paths[saddle_point] = saddle_paths

			###########################################################################################
			#Find the velocities of the separatrix at the saddle point by evaluating change across saddle
			saddle_gradients = []
			for d in [0,2]:
				# List of same direction paths by evaluating the signs of the velocities
				temp =[p for p in saddle_paths if np.abs(np.sum(np.sign(self.velocity(p[100,:],0,*self.args))))==d]
				# Find a point on each separatrix that is 1e-3 away from the saddle to generate the gradient (l2 norm)
				points = []
				for path in temp:
					for i in range(path.shape[0]):
						dist = np.linalg.norm(saddle_point-path[i,:])
						if dist>min_dist:
							points.append(path[i,:])
							break
				# Take the difference in points to estimate the gradient
				saddle_gradients.append(points[0]-points[1])

			gradients[saddle_point] = saddle_gradients


			###########################################################################################
			# Origin points on the different separatrix trajectories near the separatrix

			for path in paths[saddle_point]:
				indices = [10]
				for j in range(count):
					for i in range(indices[-1],path.shape[0]):
						delta_s = np.abs(path[indices[-1],0]-path[i,0])
						delta_h = np.abs(path[indices[-1],1]-path[i,1])
						if delta_s>1e-1 or delta_h>1e-1:
							# Once significant changes in the separatrix are detected, the point is saved
							indices.append(i)
							break
					if i == path.shape[0]-1:break

				for i in indices:
					origins.append((path[i,0],path[i,1]+1e-3))
					origins.append((path[i,0],path[i,1]-1e-3))

		return paths,origins,gradients

	def potential(self,s):
		"""Calculate the potential at point s via equation  C16.

		Parameters
		----------
		s 	-	(float) location of s

		Returns
		----------
		potential 	- 	(float) the potential of the system at point s
		"""
		pt1 = 0.5*(self.beta1-1)*(s**2)
		pt2 = 0.25*(self.beta1 - 2/3)*(s**4)
		pt3 = self.beta2*self.delta*s
		return - (self.omega_h/self.omega_s)*(pt1-pt2+pt3)

	def graphPotential(self,count,s_range=(-1,1),title="potential.png"):
		"""Graph the potential as in figure 13d

		Parameters
		----------
		count 	- 	(int) number of points at which to evaluate s
		s_range -	(tuple) of the minimum and maximum of s
		title 	- 	(str) title under which image will be saved in the working directory

		"""
		s_points = np.linspace(s_range[0],s_range[1],count)
		fig,ax = plt.subplots(1,1)
		ax.plot(s_points,self.potential(s_points))
		plt.savefig(title)

	def phasePortrait(self,t0=0,t_end=1000,freq=10,count=50,nullcline=False,title="phasePortrait.png"):
		""" Generate the phase portrait based on the given parameters i.e. the trajectories
		of (s,p) in the (s,p) plane that are solutions to the ODE system.

		Parameters
		----------
		t0,t_end 	-	(int) starting and ending times for the integration process (default: 0,1000)
		freq 		-	(int) frequency for integration per timeperiod
		count 		- 	(int) number of borderline starting points to generate
		nulllcline 	- 	(boolean) whether or not to plut the nullcline for s and h
		title 		-	(string) name under which to save figure in current working directory

		Returns
		----------
		None 		- will display plot and save to title in working directory
		"""

		# Generate the figure
		fig,ax = plt.subplots()

		###########################################################################################
		# Generate border case starting points
		h_high = [(s,1) for s in np.linspace(-1,1,count)]
		h_bottom = [(s,-1) for s in np.linspace(-1,1,count)]
		s_high = [(1,h) for h in np.linspace(-1,1,count)]
		s_low = [(-1,h) for h in np.linspace(-1,1,count)]
		border_points = h_high+h_bottom+s_high+s_low

		# Generate the path from each border origin
		timespan = np.linspace(0,t_end,freq*(t_end))
		for origin in border_points:
			path = odeint(self.velocity,origin,timespan,self.args)
			ax.plot(path[:,0],path[:,1],color='blue',linewidth=0.5)

		# Generate trajectories starting from separatrices to highlight their effects
		for origin in self.sep_origins:
			path = odeint(self.velocity,origin,np.linspace(0,10000,10000),self.args)
			ax.plot(path[:,0],path[:,1],color='blue',linewidth=0.5)

		###########################################################################################
		# Optionally add the nullcine in s and h to the graph
		if nullcline:
			S,H = np.meshgrid(np.linspace(-1,1,1000),np.linspace(-1,1,1000))
			# Initiate the velocities
			v_s = np.empty(S.shape)
			v_h = np.empty(H.shape)

			# Generate the matrices of s and h velocities
			for i in range(S.shape[0]):
				for j in range(S.shape[1]):
					v_s[i,j],v_h[i,j] = self.velocity([S[i,j],H[i,j]],0,*self.args)

			# Plot the nullcline
			ax.contour(S,H,v_s,levels=[0],linewdith=3,linestyle='--',colors='green',label='Nullincline S')
			ax.contour(S,H,v_h,levels=[0],linewdith=3,linestyle='--',colors='green',label='Nullincline H')

		###########################################################################################
		"""To add saddles - assert existence, add gradient arrows, then add separatrices"""
		if len(self.saddles)>=1:
			# Add gradient arrows
			arrow_arg = {'color':'black','headwidth':3,'width':0.004}

			for point in self.sep_gradients:
				for gradient in self.sep_gradients[point]:
					if np.sum(np.sign(gradient))==0:
						ax.quiver(point[0],point[1],gradient[0],gradient[1],pivot='tip',label='Gradient',**arrow_arg)
						ax.quiver(point[0],point[1],-gradient[0],-gradient[1],pivot='tip',**arrow_arg)
					else:
						ax.quiver(point[0],point[1],gradient[0],gradient[1],**arrow_arg)
						ax.quiver(point[0],point[1],-gradient[0],-gradient[1],**arrow_arg)

			# Add the separatrices to the graph
			for point in self.trajectorySeparatrices:
				for separatrix in self.trajectorySeparatrices[point]:
					ax.plot(separatrix[:,0],separatrix[:,1],color='red',linewidth=1.5,label='Separatrix')

		###########################################################################################
		# Plot the solutions by category
		for point in self.solutions:
			info = self.solutions[point]

			if info['type']=='node':
				if info['stable']:
					label="Stable Node"
					ax.plot(point[0],point[1],marker='o', markersize=5, color="green",label=label)
				else:
					label = "Unstable Node"
					ax.plot(point[0],point[1],marker='o', markersize=5, color="orange",label=label)
			elif info['type']=='focus':
				if info['stable']:
					label="Stable Focus"
					ax.plot(point[0],point[1],marker='o', markersize=5, color="green",label=label)
				else:
					label = "Unstable Focus"
					ax.plot(point[0],point[1],marker='o', markersize=5, color="orange",label=label)
			elif info['type']=='saddle':
				label = "Unstable Saddle"
				ax.plot(point[0],point[1],marker='o', markersize=5, color="red",label=label)

		###########################################################################################
		# Format the figure
		fig.set_size_inches(8,6)
		ax.set_xlabel('S')
		ax.set_ylabel('H')
		ax.set_title("Phase Diagram - "+title[:-4])
		ax.set_xlim(-1,1)
		ax.set_ylim(-1,1)
		ax.set_xticks([-1+(0.2*i) for i in range(11)])
		ax.set_yticks([-1+(0.2*i) for i in range(11)])
		plt.legend(loc='upper center',ncol=4)
		fig.tight_layout()
		#plt.show(block=False)
		plt.savefig(title)

if __name__=="__main__":

	# Main case - fig.13c
	param_mainCase13c = {'omega_s':0.04,'beta1':1.1,'beta2':0.55,'omega_h':0.4,'gamma':56,'delta':0.03,}
	graph_mainCase13c = {'t_end':1000,'freq':20,'count':50,'title':"MainCasePhasePortrait.png"}
	system = sentimentProcess(**param_mainCase13c)
	system.phasePortrait(**graph_mainCase13c)

	additional_figures=False
	if additional_figures:
		# Limit cycle - fig.13b
		param_limitCycle13a = {'omega_s':0.04,'beta1':1.1,'beta2':0.55,'omega_h':0.4,'gamma':67.7,'delta':0.03}
		graph_limitCycle13a = {'t_end':1000,'freq':20,'count':50,'title':"LimitCyclePhasePortrait.png"}
		system = sentimentProcess(**param_limitCycle13a)
		system.phasePortrait(**graph_limitCycle13a)

		# Fig C3
		param_c3 = {'omega_s':0.04,'beta1':1.1,'beta2':0.55,'omega_h':0.4,'gamma':1.6/0.04,'delta':0}
		gamma_c3 = [1.6/0.04,2.0/0.04,2.6/0.04,3.6/0.04]
		graph_c3 = {'t_end':1000,'freq':20,'count':50,'title':"c3.png"}
		titles = ["c3a.png","c3b.png","c3c.png","c3d.png"]
		for i in range(4):
			param_c3['gamma']=gamma_c3[i]
			graph_c3['title']=titles[i]
			system = sentimentProcess(**param_c3)
			system.phasePortrait(**graph_c3)

		# Fig C4
		param_c4 = {'omega_s':0.04,'beta1':1.15,'beta2':0.55,'omega_h':0.4,'gamma':2.4/0.04,'delta':0}
		delta_c4 = [0.0,0.06,0.08]
		graph_c4 = {'t_end':1000,'freq':20,'count':50,'title':"c3.png"}
		titles = ["c4a.png","c4b.png","c4c.png","c4d.png"]
		for i in range(3):
			param_c4['delta']=delta_c4[i]
			graph_c4['title']=titles[i]
			system = sentimentProcess(**param_c4)
			system.phasePortrait(**graph_c4)



