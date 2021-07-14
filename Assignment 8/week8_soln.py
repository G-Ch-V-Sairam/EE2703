'''									ASSIGNMENT 8:ANALYSIS OF CIRCUITS USING LAPLACE TRANSFORMS
										G CH V SAIRAM , EE19B081			'''

#Importing the required modules
import scipy.signal as sp
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import sympy
import warnings

warnings.filterwarnings("ignore")

def plot_graph(x,y,title,xlabel,ylabel):		#This function helps plot the graphs
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(x,y)
	plt.grid(True)

def lowpass(R1,R2,C1,C2,G,Vi):				#low pass filter definition function
	s=  sympy.symbols("s")
	A = sympy.Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0],[0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
	b=  sympy.Matrix([0,0,0,-Vi/R1])
	V = A.inv()*b				#A*V=B implies V = inverse(A)*B
	return A,b,V

def highpass(R1,R3,C1,C2,G,Vi):				#high pass filter definition function
	s=  sympy.symbols("s")
	A=sympy.Matrix([[0,-1,0,1/G],[s*C2*R3/(s*C2*R3+1),0,-1,0],[0,G,-G,1],[-s*C2-1/R1-s*C1,0,s*C2,1/R1]])
	b=sympy.Matrix([0,0,0,-Vi*s*C1])
	V=A.inv()*b
	return A,b,V

def symToTransferFn(Y):				#this function converts a sympy function into a version that can be understood by scipy.signals module
	Y = sympy.simplify(Y)
	n,d = sympy.fraction(Y)
	n,d = sympy.Poly(n,s), sympy.Poly(d,s)		#Differentiates the numerator polynomial and denominator polynomial of expression Y
	num,den = n.all_coeffs(), d.all_coeffs()	#Gives coefficients of numerator and denominator polynomials respectively.
	num,den = [float(f) for f in num], [float(f) for f in den]		#Convert them into floats
	return num,den

def inputs(t):
	return np.sin(2000*np.pi*t)+np.cos(2e6*np.pi*t)		#The given Vi(t)

def inp_response(Y,inp=inputs,tlim=1e-2):		#This function calculates the response for any arbitrary function
	t = np.linspace(0,tlim,100000)
	Vi=inp(t)							#Define Vi(t)
	num,den = symToTransferFn(Y)
	H = sp.lti(num,den)				#Define the transfer function
	t,y,svec = sp.lsim(H,Vi,t)		#Apply convolution.
	return t,y

def damped1(t,decay=3e3,freq=1e7):		#High frequency damped sinusoid with frequency 1e7 , decay 5e3
	return np.cos(freq*t)*np.exp(-decay*t) * (t>0)

def damped2(t,decay=1e1,freq=1e3):		#Low frequency damped sinusoid with frequency 1e3 , decay 5e1
	return np.cos(freq*t)*np.exp(-decay*t) * (t>0)


s =  sympy.symbols("s")

#Define the low pass transfer function
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
H=V[3]
print("Impulse response of lowpass filter=",H,'\n')
w=np.logspace(0,8,801)
ss=1j*w
hf=sympy.lambdify(s,H,"numpy")
v=hf(ss)

#Plotting the Magnitude response of the low pass filter
plt.title("Low pass Magnitude response")
plt.xlabel("w")
plt.ylabel("Magnitude response")
plt.loglog(w,abs(v),lw=2)
plt.grid(True)
plt.savefig("assgn8_plot1.png",dpi=3000)
plt.show()

					#Question1- Find the step response
A1,b1,V1 = lowpass(10000,10000,1e-9,1e-9,1.586,1/s)
Vo1 = V1[3]
H1 = symToTransferFn(Vo1)
t,y1 = sp.impulse(H1,None,np.linspace(0,1e-2,10000,dtype=float))
plot_graph(t,y1,"Step response for low pass filter",'t','Vo(t)')
plt.savefig("assgn8_plot2.png",dpi=3000)
plt.show()

					#Question2 - Find the output voltage for the given input voltage i.e; sum of two sinusoids of different frequencies
t = np.linspace(0,1e-3,1000000)
t,y=inp_response(H)
plot_graph(t,y,"Response of Low pass Filter to sum of sinusoids","t","Vo")
plt.savefig("assgn8_plot3.png",dpi=3000)
plt.show()

					#Question3 - Analysing the given highpass filter circuit
A2,b2,V2=highpass(10000,10000,1e-9,1e-9,1.586,1)
H=V2[3]
print("Impulse response of highpass filter=",H,'\n')
					#High pass Magnitude response
w=np.logspace(0,8,801)
ss=1j*w
hf=sympy.lambdify(s,H,"numpy")
v=hf(ss)
plt.title("Magnitude response for high pass filter")
plt.xlabel("w")
plt.loglog(w,abs(v),lw=2)
plt.grid(True)
plt.savefig("assgn8_plot4.png",dpi=3000)
plt.show()


					#Question2.2 - Response of high pass filter to the sum of sinusoids 
t,y=inp_response(H,inputs,tlim= 1e-5)
plot_graph(t,y,"Response of High pass Filter to sum of sinusoids","t","Vo")
plt.savefig("assgn8_plot5.png",dpi=3000)
plt.show()

					#Question4 - Response to damped sinusoids
t,y=inp_response(H,damped1,tlim=1e-3)
plot_graph(t,y,'Response of high pass filter to Damped high frequency sinusoid','t','Vo')
plt.savefig("assgn8_plot6.png",dpi=3000)
plt.show()
t,y=inp_response(H,damped2,tlim=1e-3)
plot_graph(t,y,'Response of high pass filter to Damped low frequency sinusoid','t','Vo')
plt.savefig("assgn8_plot7.png",dpi=3000)
plt.show()

					#Question5 - Step response for highpass filter
A3,b3,V3 = highpass(10000,10000,1e-9,1e-9,1.586,1/s)
Vo3 = V3[3]
H3 = symToTransferFn(Vo3)
t,y3 = sp.impulse(H3,None,np.linspace(0,1e-3,10000,dtype=float))
plot_graph(t,y3,"Step response for high pass filter",'t','Vo(t)')
plt.savefig("assgn8_plot8.png",dpi=3000)
plt.show()

