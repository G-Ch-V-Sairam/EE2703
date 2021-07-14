'''									ASSIGNMENT 7: LAPLACE TRANSFORM
									   G CH V SAIRAM , EE19B081						'''
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt

def plot_graphs(x,y,title,xlabel,ylabel):	#Plots the graphs
	plt.title(title)	#Sets the title
	plt.xlabel(xlabel)	#Sets the xlabel
	plt.ylabel(ylabel)	#Sets the ylabel
	plt.plot(x,y)
	
def transfer_func(frequency,decay):
	num=([1,-1*decay])			#Numerator polynomial of the given transfer function
	denom= np.polymul([1.0,0,2.25],[1,-2*decay,frequency*frequency + decay*decay])		#Denominator polynomial
	return sp.lti(num,denom)	#Returns the transfer function with given numerator and denominator

			#QUESTION 1
			
H1=transfer_func(1.5,-0.5)
t,x = sp.impulse(H1,None,np.linspace(0,50,5001))	#Impulse response of H1
plt.figure()
plot_graphs(t,x,'Damping oscillator with 0.5 decay','t','x')	#Plot the impulse response of H1
plt.savefig("assgn7_plot1.png",dpi=300)

			#QUESTION 2

H2=transfer_func(1.5,-0.05)
t,x = sp.impulse(H2,None,np.linspace(0,50,5001))		#Impulse response of H2
plt.figure()
plot_graphs(t,x,'Damping oscillator with 0.05 decay','t','x')	#Plot the impulse response of H2
plt.savefig("assgn7_plot2.png",dpi=300)

plt.show()

			#QUESTION 3

freqs=np.linspace(1.4,1.6,5)		#An array [1.4,1.45,1.5,1.55,1.6]

#Plots the impulse response for the various values of f in the array freqs
for f in freqs:
	transfer_func = sp.lti([1],[1,0,2.25])
	t = np.linspace(0,150,5001)
	time_domain_func = np.cos(f*t)*np.exp(-0.05*t)*(t>0)
	t,x,svec = sp.lsim(transfer_func,time_domain_func,t)
	plt.figure()
	plot_graphs(t,x,'Damping oscaillator with frequency='+str(f),'t','x')
	plt.savefig("assgn7_plot"+str(list(freqs).index(f)+3)+".png",dpi=300)
plt.show()

			#QUESTION 4

#After solving the given equations , we get the transfer functions X and Y as defined below
#Calculate their impulse responses and plot them
X = sp.lti([1,0,2],[1,0,3,0])
t,x = sp.impulse(X,None,np.linspace(0,50,5001))
plot_graphs(t,x,"Coupled Oscilations","t","x")
Y = sp.lti([2],[1,0,3,0])
t,y = sp.impulse(Y,None,np.linspace(0,50,5001))
plot_graphs(t,y,"Coupled Oscilations","t","y")
plt.savefig("assgn7_plot8.png",dpi=300)

plt.show()

			#QUESTION 5

R=100
L=1e-6
C=1e-6
H = sp.lti([1],[L*C,R*C,1])		#Steady state transfer function of the 2-port network
w,S,phi = H.bode()
fig,(ax1,ax2) = plt.subplots(2,1)
ax1.set_title("Magnitude response")
ax1.semilogx(w,S)		#Plot its magnitude response
ax2.set_title("Phase response")
ax2.semilogx(w,phi)		#Plot its phase response
plt.savefig("assgn7_plot9.png",dpi=300)
plt.show()

			#QUESTION 6

def func(t):
	return np.cos(1000*t) -np.cos(1e6*t)		#The input signal Vi(t)

time=np.linspace(0,30e-6,10000)
t,y,_ = sp.lsim(H,func(time),time)
plot_graphs(t,y,"Output of RLC for t<30us","t","x")		#Plots output signal for t<30us
time=np.linspace(0,30e-3,10000)
plt.savefig("assgn7_plot10.png",dpi=300)
plt.figure()
t,y,_ = sp.lsim(H,func(time),time)
plot_graphs(t,y,"Output of RLC for t<30ms","t","x")		#Plots output signal for t<30ms
plt.savefig("assgn7_plot11.png",dpi=300)
plt.show()
