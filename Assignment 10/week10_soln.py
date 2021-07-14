'''					Assignment 10: Spectra of non-periodic signals
						G Ch V Sairam , EE19B081			'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.mplot3d.axes3d as p3			#Used to plot 3d graphs

pi=np.pi

#Example-Spectra of sin(sqrt(2)t)

t=np.linspace(-pi,pi,65)[:-1]
dt=t[1]-t[0]
fmax=1/dt
y=np.sin(np.sqrt(2)*t)
y[0]=0		# The sample corresponding to -tmax should be set to zero
y=np.fft.fftshift(y)
Y=np.fft.fftshift(np.fft.fft(y))/64.0
w=np.linspace(-pi*fmax,pi*fmax,65)[:-1]

plt.figure()
plt.subplot(2,1,1)
plt.plot(w,np.abs(Y),lw=2)
plt.xlim([-10,10])
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
plt.xlim([-10,10])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig("assgn10_plot1.png")
plt.show()


t1=np.linspace(-pi,pi,65)[:-1]
t2=np.linspace(-3*pi,-pi,65)[:-1]
t3=np.linspace(pi,3*pi,65)[:-1]

# Plot the time function over several time periods.
plt.figure(2)
plt.plot(t1,np.sin(np.sqrt(2)*t1),'b',lw=2)
plt.plot(t2,np.sin(np.sqrt(2)*t2),'r',lw=2)
plt.plot(t3,np.sin(np.sqrt(2)*t3),'r',lw=2)
plt.ylabel(r"$y$",size=16)
plt.xlabel(r"$t$",size=16)
plt.title(r"$\sin\left(\sqrt{2}t\right)$")
plt.grid(True)
plt.savefig("assgn10_plot2.png")
plt.show()


#To find the function the DFT is trying to fourier analyse, we replicate just the blue points.
y=np.sin(np.sqrt(2)*t1)

plt.figure(3)
plt.plot(t1,y,'bo',lw=2)
plt.plot(t2,y,'ro',lw=2)
plt.plot(t3,y,'ro',lw=2)
plt.ylabel(r"$y$",size=16)
plt.xlabel(r"$t$",size=16)
plt.title(r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$ ")
plt.grid(True)
plt.savefig("assgn10_plot3.png")
plt.show()


#The DFT is just like the fourier series, except that both time and frequency are samples. 
#So, if the time samples are like a ramp, the frequency samples will decay as 1/ω. Let us verify this for the ramp itself.
t=t1
dt=t[1]-t[0]
fmax=1/dt
y=t
y[0]=0
y=np.fft.fftshift(y)
Y=np.fft.fftshift(np.fft.fft(y))/64.0
w=np.linspace(-pi*fmax,pi*fmax,65)[:-1]
plt.figure()
plt.semilogx(np.abs(w),20*np.log10(np.abs(Y)),lw=2)
plt.xlim([1,10])
plt.ylim([-20,0])
plt.xticks([1,2,5,10],["1","2","5","10"],size=16)
plt.ylabel(r"$|Y|$ (dB)",size=16)
plt.title(r"Spectrum of a digital ramp")
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig("assgn10_plot4.png")
plt.show()


#We observe that the spikes occur at the end of a periodic interval.
#So, to damp the function at those points, we multiply our function with something called a 'HAMMING WINDOW'.
#This smeares the 2 spikes but also suppresses the jump at the edge of the window.
t1=np.linspace(-pi,pi,65)[:-1]
t2=np.linspace(-3*pi,-pi,65)[:-1]
t3=np.linspace(pi,3*pi,65)[:-1]
n=np.arange(64)
wnd=np.fft.fftshift(0.54+0.46*np.cos(2*pi*n/63))
y=np.sin(np.sqrt(2)*t1)*wnd
plt.figure(3)
plt.plot(t1,y,'bo',lw=2)
plt.plot(t2,y,'ro',lw=2)
plt.plot(t3,y,'ro',lw=2)
plt.ylabel(r"$y$",size=16)
plt.xlabel(r"$t$",size=16)
plt.title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
plt.grid(True)
plt.savefig("assgn10_plot5.png")
plt.show()


#The hamming window gave an extra -10dB of suppression.
#Plot the DFT of this sequence.
t=t1
dt=t[1]-t[0];
fmax=1/dt
wnd=np.fft.fftshift(0.54+0.46*np.cos(2*pi*n/63))
y=np.sin(np.sqrt(2)*t)*wnd
y[0]=0
y=np.fft.fftshift(y)
Y=np.fft.fftshift(np.fft.fft(y))/64.0

plt.figure()
plt.subplot(2,1,1)
plt.plot(w,np.abs(Y),lw=2)
plt.xlim([-8,8])
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
plt.xlim([-8,8])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig("assgn10_plot6.png")
plt.show()


#We should get better results if we increase the number of points .
t=np.linspace(-4*pi,4*pi,257)[:-1]
dt=t[1]-t[0];fmax=1/dt
n=np.arange(256)
wnd=np.fft.fftshift(0.54+0.46*np.cos(2*pi*n/256))
y=np.sin(np.sqrt(2)*t)
y=y*wnd
y[0]=0
y=np.fft.fftshift(y)
Y=np.fft.fftshift(np.fft.fft(y))/256.0
w=np.linspace(-pi*fmax,pi*fmax,257)[:-1]

plt.figure()
plt.subplot(2,1,1)
plt.plot(w,np.abs(Y),lw=2)
plt.xlim([-4,4])
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
plt.xlim([-4,4])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig("assgn10_plot7.png")
plt.show()


#Define a function to plot and return the DFT of any arbitrary function.
def spectrum(lim,n,f,xlim1,title1,t_=True,t1=0,windowing=True,xlabel1 = r"$\omega$",ylabel1= r"Magnitude of Y", ylabel2 = r"Phase of Y",savename = "abc.png"):
	if t_:
		t=np.linspace(-lim,lim,n+1)[:-1]
	else:
		t=t1
	dt=t[1]-t[0]
	fmax=1/dt
	y = f(t)
	if (windowing):
		m=np.arange(n)		#An array of values 1 to n-1 .
		wnd=np.fft.fftshift(0.54+0.46*np.cos(2*pi*m/n))		#Define the hamming window.
		y = y*wnd		#Multiply the function by the above-specified hamming window.
	y[0]=0		#Sample corresponding to -tmax =0
	y=np.fft.fftshift(y)
	Y=np.fft.fftshift(np.fft.fft(y))/float(n)
	w=np.linspace(-pi*fmax,pi*fmax,n+1)[:-1]
	
	#Plot the magnitude and phase of the function.
	mag = np.abs(Y)			#Magnitude of Y
	phi = np.angle(Y)		#Phase of Y
	plt.figure()
	plt.subplot(2,1,1)
	plt.plot(w,mag,lw=2)
	plt.xlim([-xlim1,xlim1])
	plt.ylabel(ylabel1,size=16)
	plt.title(title1)
	plt.grid(True)
	plt.subplot(2,1,2)
	phi[np.where(mag<3e-3)] = 0		#Set the phase of those points where magnitude is negligible to be 0 .
	plt.plot(w,phi,'ro',lw=2)
	plt.xlim([-xlim1,xlim1])
	plt.ylabel(ylabel2,size=16)
	plt.xlabel(xlabel1,size=16)
	plt.grid(True)
	plt.savefig(savename)
	plt.show()
	return w,Y

#Set any value in 0.5<w<1.5 fow w0 and any arbitrary value for delta.
#Here, I have chosen w0=0.9 and delta=1.2 . 
w0=0.9
delta=1.2
print("Actual omega=",w0)
print("Actual delta=",delta,"\n")

def cos3(t,w=0.86):			#Define the cos**3(t) function
	return (np.cos(w*t))**3
	
def cosine(t):			#Define the cosine function cos(wt+delta)
	return np.cos(w0*t + delta)

def noisycosine(t):
	return np.cos(w0*t + delta) + 0.1*np.random.randn(128)		#Add white gaussian noise to the cosine function. 

#Question 2

#FFT of cos^3 windowed and unwindowed
a,b = spectrum(4*pi,256,cos3,5,r"Spectrum of $cos^3(w_0t)$ without windowing",windowing=False,savename = 'assgn10_plot8.png')
a,b = spectrum(4*pi,256,cos3,5,r"Spectrum of $cos^3(w_0t)$ with windowing",savename = 'assgn10_plot9.png')

#Question 3

#FFT of cos(wt+delta) windowed.
w,Y = spectrum(pi,128,cosine,3,r"Spectrum of $cos(w_0t + \delta)$ for $w_0$="+str(w0)+",delta="+str(delta),savename = 'assgn10_plot10.png')

#A function to estimate w and delta from the digital spectra plot.
def est_omega_delta(w,Y):
	ii = np.where(w >= 0)
	sol_w = np.sum(w[ii][:5]*np.absolute(Y)[ii][:5])/np.sum(np.absolute(Y)[ii][:5])
	kk = np.argmax(np.absolute(Y[ii]))
	sol_delta = np.angle(Y[ii])[kk]
	print ("Estimated omega = ", sol_w)
	print ("Estimated delta=",sol_delta)#weighted average for first 2 points

est_omega_delta(w,Y)

#Question 4

print("\nAfter adding white gaussian noise:")
#FFT of cos(wt+delta) + noise , windowed
w,Y = spectrum(pi,128,noisycosine,3,r"Spectrum of $cos(w_0t + \delta)$+noise for $w_0$="+str(w0)+",delta="+str(delta),savename = 'assgn10_plot11.png')

est_omega_delta(w,Y)		#Estimate w and delta for noisy cosine function.

#Question 5
 
def chirp(t):
    return np.cos(16*(1.5 + t/(2*pi))*t) 

#FFT of the chirp function both unwindowed and windowed.
w,Y = spectrum(pi,1024,chirp,60,r"Spectrum of chirp function without windowing",windowing=False,savename = 'assgn10_plot12.png')
w,Y = spectrum(pi,1024,chirp,60,r"Spectrum of chirp function with windowing",savename = 'assgn10_plot13.png')

#question 6

t = np.linspace(-np.pi, np.pi, 1025)[:-1]
dt=t[1] - t[0]
fmax = 1/dt
Y1 = np.zeros((64,16), dtype=complex)		#Creates a 64 by 16 complex matrix 
Y1_wnd = np.zeros((64, 16), dtype=complex)

for i in range(16):
    t1 = t[64*i:(i+1)*64]		#Break the 1024 vector into pieces that are each 64 samples wide.
    w1 = np.linspace(-np.pi*fmax, np.pi*fmax, 65)[:-1]
    y1 = chirp(t1)		
    n = np.arange(64)
    wnd = np.fft.fftshift(0.54 + 0.46*np.cos(2*np.pi*n/63))
    y1_wnd = y1 * wnd		#Define the hamming window and multiply it with the chirp function.
    y1 = np.fft.fftshift(y1)
    Y1[:, i] = np.fft.fftshift(np.fft.fft(y1))/64.0		#DFT of the chirp function, unwindowed.
    y1_wnd = np.fft.fftshift(y1_wnd)
    Y1_wnd[:, i] = np.fft.fftshift(np.fft.fft(y1_wnd))/64.0		#DFT of the chirp function, windowed.


t1 = t[::64]
t1, w1 = np.meshgrid(t1, w1)

#Plot the time-frequency plot of the unwindowed chirp function
fig9 = plt.figure()
ax = p3.Axes3D(fig9)
ax.plot_surface(w1, t1, np.absolute(Y1), cmap=cm.jet)
plt.title('DFT Magnitude plot of the Chirped Signal without hamming window')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$t$')
plt.savefig("assgn10_plot14.png")
plt.show()

#Plot the time-frequency plot of the windowed chirp function 
fig10 = plt.figure()
ax = p3.Axes3D(fig10)
ax.plot_surface(w1, t1, np.absolute(Y1_wnd), cmap=cm.jet)
plt.title('DFT Magnitude plot of the Chirped Signal with hamming window')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$t$')
plt.savefig("assgn10_plot15.png")
plt.show()
