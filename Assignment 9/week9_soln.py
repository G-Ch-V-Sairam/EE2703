'''						ASSIGNMENT 9: THE DIGITAL FOURIER TRANSFORM
								G CH V SAIRAM , EE19B081						'''


from pylab import *
import numpy as np
import matplotlib.pyplot as plt

#Example 1

x=np.random.rand(100)		#Creates an array of 100 random numbers
X=np.fft.fft(x)				#Calculates DFT of x
y=np.fft.ifft(X)			#Calculates inverse DFT of x
print ("Absolute Maximum Error = ",abs(x-y).max())

#Example 2

pi=np.pi
x=np.linspace(0,2*pi,128)
y=np.sin(5*x)
Y=np.fft.fft(y)		#Y=DFT of sin(5x)

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.abs(Y),lw=2)		#Plots magnitude of Y
plt.grid(True)
plt.ylabel(r"Magnitude of Y")
plt.title("Spectrum of sin(5t) without phase shift")
plt.subplot(2,1,2)
plt.plot(np.unwrap(np.angle(Y)),lw=2)		#Plots phase angle of Y
plt.ylabel(r"Phase of Y")
plt.xlabel(r"$\omega$")
plt.grid(True)
plt.savefig("assgn9_plot1.png")
plt.show()

#Example 3

x=np.linspace(0,2*pi,129);x=x[:-1]
y=np.sin(5*x)

'''Our position vector started at 0 and went to 2pi, which is correct.
The fft gave an answer in the same value. 
So we need to shift the pi to 2pi portion to the left as it represents negative frequency.
This can be done with a command called fftshift.'''

Y=np.fft.fftshift(np.fft.fft(y))/128.0
w=np.linspace(-64,63,128)

plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y),lw=2)
plt.xlim([-10,10])		#Set upper and lower limits to x values.
plt.ylabel(r"Magnitude of Y",size=16)
plt.title(r"Spectrum of sin(5t)")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
ii=np.where(abs(Y)>1e-3)			#Find where magnitude value is significant.
plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)	#Plot phase points only at those points.
plt.xlim([-10,10])
plt.ylabel(r"Phase of Y",size=16)
plt.xlabel(r"$k$",size=16)
plt.grid(True)
plt.savefig("assgn9_plot2.png")
plt.show()


#Example 4

y=(1+0.1*np.cos(x))*np.cos(10*x)
Y=np.fft.fftshift(np.fft.fft(y))/128

plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y),lw=2)
plt.xlim([-15,15])
plt.ylabel(r"Magnitude of Y",size=16)
plt.title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
plt.xlim([-15,15])
plt.ylabel(r"Phase of Y",size=16)
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig("assgn9_plot3.png")
plt.show()

#Example 5

t=np.linspace(-4*pi,4*pi,513);t=t[:-1]
y=(1+0.1*np.cos(t))*np.cos(10*t)
Y=np.fft.fftshift(np.fft.fft(y))/512
w=np.linspace(-64,64,513);w=w[:-1]

plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y),lw=2)
plt.xlim([-15,15])
plt.ylabel(r"Magnitude of Y",size=16)
plt.title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
plt.xlim([-15,15])
plt.ylabel(r"Phase of Y",size=16)
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig("assgn9_plot4.png")
plt.show()


def dft(x_start,x_end,f,xlim1,titl,ylabel1,ylabel2,xlabel1,savename,ro=True):
	steps=512
	x=np.linspace(x_start,x_end,steps+1)[:-1]		#Set x range
	y = f(x)
	Y=np.fft.fftshift(np.fft.fft(y))/float(steps)	#Find dft of various functions
	w=512*(np.linspace(-pi,pi,steps+1)[:-1])/(x_end-x_start)	#Set w range
	plt.figure()
	plt.subplot(2,1,1)
	plt.plot(w,abs(Y),lw=2)		#Plot magnitude response of DFT of various functions
	plt.xlim([-xlim1,xlim1])	#Set x limits
	plt.ylabel(ylabel1)		#Set y label
	plt.title(titl)		#Set title
	plt.grid(True)
	plt.subplot(2,1,2)
	if ro==True:
		plt.plot(w,np.angle(Y),'ro',lw=2)		#Plot phase points everywhere within the x limits
		plt.xlim([-xlim1,xlim1])
	ii=np.where(abs(Y)>1e-3)	#Find the points where magnitude is not negligible
	plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)	#Plot phase points only at those points
	plt.ylabel(ylabel2)		#Set y label to phase response plot.
	plt.xlabel(xlabel1)
	plt.grid(True)
	plt.savefig(savename)
	plt.show()

def f1(x):
    return (np.sin(x))**3
def f2(x):
    return (np.cos(x))**3
def f3(x):
    return np.cos(20*x + 5*np.cos(x))

#Problem 2
dft(-4*pi,4*pi,f1,15,r"Spectrum of $sin^3(t)$",r"Magnitude of Y",r"Phase of Y",r"$\omega$","assgn9_plot5.png")		#DFT of sin(x)**3
dft(-4*pi,4*pi,f2,15,r"Spectrum of $cos^3(t)$",r"Magnitude of Y",r"Phase of Y",r"$\omega$","assgn9_plot6.png")		#DFT of cos(x)**3

#Problem 3
dft(-4*pi,4*pi,f3,30,r"Spectrum of $cos(20t+5cos(t))$",r"$|Y|$",r"Phase of $Y$",r"$\omega$","assgn9_plot7.png",ro=False)		#DFT of cos(20t+5cos(t))

#Problem 4

def gauss(x):
    return np.exp(-0.5*x**2)		#Gaussian function

def expectedgauss(w):
    return np.sqrt(2*pi)*np.exp(-w**2/2)		#Gaussian in w

wlim=5
tolerance=1e-6
T = 8*pi
N = 128
Yold=0
err=1+tolerance
iters = 0
error=[]

#To find the optimum time range to get an accurate frequency domain.

while err>tolerance:  
	x = np.linspace(-T/2,T/2,N+1)[:-1]
	w = np.linspace(-N*pi/T,N*pi/T,N+1)[:-1]
	y = gauss(x)
	Y=np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y)))*T/N
	Y_exp=expectedgauss(w)
	err = np.max(np.abs(Y[::2]-Yold))
	error.append(np.max(np.abs(Y-Y_exp)))
	Yold=Y
	iters+=1
	T*=2
	N*=2
	
error = np.max(error)
print("True Error: ",error)
print("Samples, N = "+str(N)+"\n"+"Time Period,T = "+str(int(T/pi))+"pi")

#Estimate DFT of gaussian function. Plot its magnitude and phase.

mag = np.abs(Y)
phi = np.angle(Y)

plt.figure()
plt.subplot(2,1,1)
plt.plot(w,mag,lw=2)
plt.xlim([-wlim,wlim])
plt.ylabel('Magnitude',size=16)
plt.title("Estimate fft of gaussian")
plt.grid(True)
plt.subplot(2,1,2)
ii=np.where(mag>1e-3)
plt.plot(w[ii],phi[ii],'go',lw=2)
plt.xlim([-wlim,wlim])
plt.ylabel("Phase",size=16)
plt.xlabel("w",size=16)
plt.grid(True)
plt.savefig("assgn9_plot8.png")
plt.show()

#The actual DFT of Gaussian is a Gaussian in w. Hence, we plot the magnitude and phase of gaussian in w.

Y_exp = expectedgauss(w)   
mag_exp = np.abs(Y_exp)
phi_exp = np.angle(Y_exp)

plt.figure()
plt.subplot(2,1,1)
plt.plot(w,mag_exp,lw=2)
plt.xlim([-wlim,wlim])
plt.ylabel('Magnitude',size=16)
plt.title("True fft of gaussian")
plt.grid(True)
plt.subplot(2,1,2)
ii=np.where(mag_exp>1e-3)
plt.plot(w[ii],phi_exp[ii],'go',lw=2)
plt.xlim([-wlim,wlim])
plt.ylabel("Phase",size=16)
plt.xlabel("w",size=16)
plt.grid(True)
plt.savefig("assgn9_plot9.png")
plt.show()
