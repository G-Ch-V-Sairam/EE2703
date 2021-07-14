#							EE2703 ASSIGNMENT-4 SOLUTION


import math
import numpy as np
import scipy.integrate as intg
from matplotlib.pylab import *
import matplotlib.pyplot as plt

def e(x):
	return np.exp(x)

def cos_cos(x):
	y=np.cos(x)					
	return np.cos(y)	

X=np.linspace(-2*math.pi,4*math.pi,600)		#Define a linear space X from -2*pi to 4*pi with 600 points 

plt.figure(1)
semilogy(X,e(X))							#Plot e^x function on a semilog plot 
per_ext_e=e(X%(2*math.pi))
semilogy(X,per_ext_e,'--')							#Plots the periodic extension of e^x
title(r'Semilog plot of e^x')						#Adds a title
xlabel(r'x')								#Gives a label on the x-axis
ylabel(r'log(e^x)')								#Gives a label on the y-axis
grid()										#Adds grid to the figure
legend(['log(e^x)','Periodic extension of log(e^(x))'])
savefig("assn4_plot1.png",dpi=300)					#Saves this plot in a file named 'assn4_plot1.png'


plt.figure(2)
plot(X,cos_cos(X))						#Plot cos(cos(x)) function w.r.t x
per_ext_coscos=cos_cos(X%(2*math.pi))
semilogy(X,per_ext_coscos,'--')						#Plots the periodic extension of cos(cos(x))
title(r'Plot of cos(cos(x)) function')
xlabel(r'x')
ylabel(r'cos(cos(x))')
legend(['cos(cos(x))','Periodic extension of cos(cos(x))'])
grid()
savefig("assn4_plot2.png",dpi=300)

#	Finding the fourier series coefficients of the function e^x
a_e = np.zeros(51)
def fcos1(x,k):
    return e(x)*np.cos(k*x)/np.pi										#Returns the function to be integrated to get a_n values(except a_0)
def fsin1(x,k):
    return e(x)*np.sin(k*x)/np.pi										#Returns the function to be integrated to get b_n values
a_e[0] = intg.quad(e,0,2*np.pi)[0]/(2*np.pi)							#Finds value of a_0
for i in range(1,51):
    if(i%2==1):
        a_e[i] = intg.quad(fcos1,0,2*np.pi,args=(int(i/2)+1))[0]		#Finds values of a_n for e^x
    else:
   	    a_e[i] = intg.quad(fsin1,0,2*np.pi,args=(int(i/2)))[0]			#Finds values of b_n for e^x

#		Finding the fourier coefficients of cos(cos(x))

a_coscos = np.zeros(51)
def fcos2(x,k):
    return cos_cos(x)*np.cos(k*x)/np.pi									#Returns the function to be integrated to get a_n values for cos(cos(x))
def fsin2(x,k):
    return cos_cos(x)*np.sin(k*x)/np.pi									#Returns the function to be integrated to finf b_n values for cos(cos(x))
a_coscos[0] = intg.quad(cos_cos,0,2*np.pi)[0]/(2*np.pi)					#Finds a_0 value for cos(cos(x))
for i in range(1,51):
	if(i%2==1):
		a_coscos[i] = intg.quad(fcos2,0,2*np.pi,args=(int(i/2)+1))[0]		#Finds a_n values for cos(cos(x))
	else:
		a_coscos[i] = intg.quad(fsin2,0,2*np.pi,args=(int(i/2)))[0]			#Finds b_n values for cos(cos(x))

# 	The list a_e consists of the first 51 fourier series coefficients of e^x
#The list a_coscos consists of the first 51 fourier series coefficients of cos(cos(x))

n=range(51)
plt.figure(3)
semilogy(n,abs(a_e),'ro')									#Plot the fourier coefficients of e^x on semilog scale
title(r'Fourier coeffn.s of e^x on semilog scale')
xlabel(r'n')
ylabel(r'log(coeff)')
grid()
savefig("assn4_plot3.png",dpi=300)


plt.figure(4)
loglog(n,abs(a_e),'ro')										#Plot the fourier coefficients of e^x on loglog scale
title(r'Fourier coeffn.s of e^x on loglog scale')
xlabel(r'log(n)')
ylabel(r'log(coeff)')
grid()
savefig("assn4_plot4.png",dpi=300)


plt.figure(5)
semilogy(n,abs(a_coscos),'ro')								#Plot the fourier coefficients of cos(cos(x)) on semilog scale
title(r'Fourier coeffn.s of cos(cos(x)) on semilog scale')
xlabel(r'n')
ylabel(r'log(coeff)')
grid()
savefig("assn4_plot5.png",dpi=300)


plt.figure(6)
loglog(n,abs(a_coscos),'ro')								#Plot the fourier coefficients of cos(cos(x)) on loglog scale
title(r'Fourier coeffn.s of cos(cos(x)) on loglog scale')
xlabel(r'n')
ylabel(r'log(coeff)')
grid()
savefig("assn4_plot6.png",dpi=300)



x=linspace(0,2*pi,401)								#Create a linear space from 0 to 2*pi with 400 points
x=x[:-1] 											#We drop the last value since 0 is the same as 2*pi
A=np.zeros((400,51))

A[:,0]=1											#First column consists of all 1's
for k in range(1,26):
	A[:,2*k-1]=cos(k*x) 							#The odd columns consists of cos(kx)
	A[:,2*k]=sin(k*x)								#The even columns consist of sin(kx)


b1=e(x)
c1=lstsq(A,b1,rcond=None)[0]					#lstsq returns a list. We need only the first value in the list.It consists of a list of the predicted values for c vector for e^x function.
b2=cos_cos(x)
c2=lstsq(A,b2,rcond=None)[0]					#lstsq returns a list. We need only the first value in the list.It consists of a list of the predicted values for c vector for cos(cos(x)) function.
		
plt.figure(7)		
semilogy(n,abs(a_e),'ro')									#Plot original values of fourier coefficients of e^x with red dots on semilog scale
semilogy(n,abs(c1),'go',markersize=4)									#Plot predicted values with green dots on the same plot
title(r'Fourier coeffn.s of e^x on semilog scale')
xlabel(r'n')
ylabel(r'log(coeff)')
legend(['By integration','By least squares method'])
grid()
savefig("assn4_plot7.png",dpi=300)

	
plt.figure(8)
loglog(n,abs(a_e),'ro')										#Plot original values of fourier coefficients of e^x with red dots on loglog scale
loglog(n,abs(c1),'go',markersize=4)										#Plot predicted values with green dots
title(r'Fourier coeffn.s of e^x on loglog scale')
xlabel(r'log(n)')
ylabel(r'log(coeff)')
legend(['By integration','By least squares method'])
grid()
savefig("assn4_plot8.png",dpi=300)


plt.figure(9)
semilogy(n,abs(a_coscos),'ro')								#Plot original values of fourier coefficients of cos(cos(x)) with red dots on semilog scale
semilogy(n,abs(c2),'go',markersize=4)									#Plot predicted values with green dots
title(r'Fourier coeffn.s of cos(cos(x)) on semilog scale')
xlabel(r'n')
ylabel(r'log(coeff)')
legend(['By integration','By least squares method'])
grid()
savefig("assn4_plot9.png",dpi=300)


plt.figure(10)
loglog(n,abs(a_coscos),'ro')								#Plot original values of fourier coefficients of cos(cos(x)) with red dots on loglog scale
loglog(n,abs(c2),'go',markersize=4)										#Plot predicted values with green dots
title(r'Fourier coeffn.s of cos(cos(x)) on loglog scale')
xlabel(r'n')
ylabel(r'log(coeff)')
legend(['By integration','By least squares method'])
grid()
savefig("assn4_plot10.png",dpi=300)


error_e=np.amax(np.abs(a_e-c1))					#Calculating error in coefficients of e^x
error_coscos=np.amax(np.abs(a_coscos-c2))		#Calculating error in coefficients of cos(cos(x))
print("The error in the coefficients of e^x=",error_e)
print("The error in the coefficients of cos(cos(x))",error_coscos)

plt.figure(11)
result_e=c_[np.dot(A,c1)]									#Calculate the matrix obtained by multiplying A , the matrix consisting predicted coefficient valuees of e^x
semilogy(x,result_e,'go')									#Plot the expected plot of e^x
semilogy(x,e(x),'r-')											#Plot e^x on a semilog plot
title(r'Exponential function e^x')
xlabel(r'x')
ylabel(r'log(e^x)')
legend(['Using lstsq method','actual function'])
grid()
savefig("assn4_plot11.png",dpi=300)

plt.figure(12)
result_coscos=c_[np.dot(A,c2)]								#Calculate the matrix obtained by multiplying A , the matrix consisting predicted coefficient valuees of cos(cos(x))
plot(x,result_coscos,'go')									#Plot the expected plot of cos(cos(x))
plot(x,cos_cos(x),'r-')											#Plot the original function cos(cos(x))
title(r'cos(cos(x))')
xlabel(r'x')
ylabel(r'cos(cos(x))')
legend(['Using lstsq method','actual function'])
grid()
savefig("assn4_plot12.png",dpi=300)

plt.show()
