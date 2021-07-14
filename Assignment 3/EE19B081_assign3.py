'''
		EE2703 Assignment-3 Solution 	'''




'''Note:
	1.The commandline argument to run this code is :
							python3 EE19B081_assign3.py
	
	2.These plots vary if the file generate_data.py is run multiple times.		''' 


import numpy as np
from matplotlib.pylab import *
import scipy.special as sp
import matplotlib.pyplot as plt
from statistics import stdev									#Importing various modules and functions
from random import random
from scipy.linalg import lstsq

				#Q2)Load the file "fitting.dat" and extract the data
data=np.loadtxt("fitting.dat")								#Load the file 'fitting.dat'
x=data[:,0]											#Extract the 1st column from the file , i.e; time assigned to the variable x
y=data[:,1:]											#Extract the remaining columns, which contain f(t)+noise for various values of sigma assigned to the variable y.

def g_list(A,B):										#Define the function g for any co-efficients A,B
	g_val= [A*sp.jn(2,t)+B*t for t in x] 
	return g_val

			#Q3,4)Plot the function values without noise , and with noise for different sigma values
plt.figure(0)

plot(x,y)											#Plotting the columns of given data ,i.e; f(t) +noise for various sigma values
plot(x,g_list(1.05,-0.105),color='black')							#Plotting original value of the function without any noise
title(r'Q4: Data to be fitted to theory')
xlabel(r'$time$',size=14)									
ylabel(r'$f(t)+noise$',size=14)
scl=np.logspace(-1,-3,9)
legend(scl) 
grid()  
savefig('plot0.png',dpi=300)
				#Q5)Plot the function and the error bars for sigma=0.10
plt.figure(1)

noise=[]
for t in range(len(y[:,0])):
	noise.append(y[:,0][t]-g_list(1.05,-0.105)[t])					#Calculating noise for sigma=0.10
std_dev=stdev(noise)										#Calculate standard devation of the noise through numpy
plot(x,g_list(1.05,-0.105),color='black')							#Plot original function
errorbar(x[::5],y[:,0][::5],std_dev,fmt='ro')							#Plot error bars
title('Q5. Datapoints for sigma =0.10 along with exact function')
xlabel(r'$t$',size=14)
legend(["f(t)","Error bars"])
grid()
savefig('plot1.png',dpi=300)

	#Q6)Set up the matrix M and check whether the matrix multiplication of M,column vector [A0,B0] is equal to the given function for random values of A0,B0

Jn_func_values=[sp.jn(2,t) for t in x]							#	= Bessel function of 2nd order
M=c_[Jn_func_values , x]									#Setting up the matrix M


A0=random();B0=random()									#Create some random values of A,B to check whether g(t,A0,B0) is equal to M * column vector [A0,B0]
A_B_matrix=[A0,B0]
A_B_matrix=c_[A_B_matrix]

g_matrix=np.dot(M,A_B_matrix)

for n in range(len(x)):
	if g_matrix[n][0]!=g_list(A0,B0)[n]:
		flag=False
		break		
if(flag):
	print("The vector obtained by multiplying matrix M with column matrix [A0  B0]  is equal to g(t,A0,B0).")
	
else:
	print("The vector obtained by multiplying matrix M with column matrix [A0  B0]  is not equal to g(t,A0,B0).")
	
				#Q7)Calculate the error values
				
A=[i/10 for i in range(21)]									#Set up the lists A,B as specified in the question
B=[j/100 for j in range(-20,1)]
E=[]
for i in range(len(A)):
	E.append([])
	for j in range(len(B)):
		E[i].append(0)
		E[i][j]=((y[:,0]-g_list(A[i],B[j]))**2).mean()				#Calculating the mean squared error
			

				#Q8)Plot the contour plot of E[i][j]

plt.figure(2)
		
CS=contour(A,B,E,levels=20)		#Plotting the contour plot of the mean squared error
plot(1.05,-0.105,marker='o',color='r',label='$Exact location$')
title('Contour Plot of Error Eij')
xlabel(r'$A$',size=10)
ylabel(r'$B$',size=10)
clabel(CS,CS.levels[:4], inline=1, fontsize=10)
grid()
savefig('plot2.png',dpi=300)
					#Q9)Obtain the best estimate of A,B using the lstsq function
estimate=[]
for i in range(len(y[0])):
	p,resid,rank,sig=lstsq(M,y[:,i])
	estimate.append(p)									#Calculating the estimated values of A,B using lstsq function
	
A_error=[abs(estimate[i][0]-1.05) for i in range(len(estimate))]				#A_error = absolute value of (estimate(A) - actual value of A)
B_error=[abs(estimate[i][1]+0.105) for i in range(len(estimate))]				#B_error = absolute value of (estimate(B) - actual value of B)

					#Q10)Plot the error in approximation of A,B for different data files versus noise sigma
plt.figure(3)

plot(scl,A_error,'o',linestyle=':')									#Plotting A_error versus sigma
plot(scl,B_error,'o',linestyle=':')									#Plotting B_error versus sigma
title("Variation Of error with Noise")
xlabel(r'$Noise standard deviation$',size=10)
ylabel(r'MS Error',size=10)									
legend(["Aerr","Berr"])
grid()
savefig('plot3.png',dpi=300)
					#Q11)Repeat the above plot in log-log scale
plt.figure(4)

loglog(scl,A_error,'o',linestyle=':')									#Plotting A_error vs. sigma in log-log scale
loglog(scl,B_error,'o',linestyle=':')									#Plotting B_error vs. sigma in log-log scale
legend(["Aerr","Berr"])
title("Variation Of error with Noise on loglog scale")
xlabel(r'$\sigma_n$',size=10)
ylabel(r'MS Error',size=10)
grid()
savefig('plot4.png',dpi=300)
plt.show()

