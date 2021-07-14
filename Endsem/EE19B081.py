'''				APL ENDSEM 
			G CH V SAIRAM , EE19B081		'''



#		 QUESTION 1 - PSEUDO CODE
'''
This program plots the magnetic field vector along the z-axis from a loop antenna and fit the data into an exponential.
Define a meshgrid of size 3 by 3 by 1000
k=1/radius , lambda = 2pi/k
Break the loop into 100 sections and find out the phi values of the centres of the sections
Calculate I=4picos(phi)/mu_0 at the above points and plot them.
r' = radius*c_[cos(phi), sin(phi), zeros_like(phi)]
dl' = c_[-sin(phi), cos(phi), zeros_like(phi)]*lambda/100
Function calc(l):
			R[i,j,k,l]=|r_ijk-r'_l|
A_(x,y)[i,j,k] = sum_l(cos(phi[l])*exp(-1j*k*R[i,j,k,l])*dl'_(x,y)[l])
Calculate B using the given equation
plot B vs z
Fit B as c*(z^b)
i.e; log(B)=log(c)+b*log(z)
fit [1 log(z)]*[log(c) b] = log(B)

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

pi=np.pi
mu_0= 4*pi*1e-7


#		QUESTION 2
x=np.linspace(0,2,3)
y=np.linspace(0,2,3)
z=np.linspace(1,1000,1000)
X,Y,Z=np.meshgrid(x,y,z)		# 3 by 3 by 1000 meshgrid


#		QUESTION 3
radius=10
k=1/radius
sections=100
phi=np.linspace(0,2*pi,sections+1)		#phi values of the centre points of the 100 sections of the loop.
phi=phi[:-1]
x=radius*np.cos(phi)
y=radius*np.sin(phi)

current_x =-np.sin(phi)*np.cos(phi)*4*pi/mu_0	#I_x
current_y = np.cos(phi)*np.cos(phi)*4*pi/mu_0	#I_y

plt.quiver(x,y,current_x,current_y)		#Plot the current elements
plt.title("Current elements")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("endsem_plot1.png")
plt.show()

# r vector
r = np.zeros((3,3,1000,3))
r[:,:,:,0]=X
r[:,:,:,1]=Y
r[:,:,:,2]=Z


#		QUESTION 4
r_prime = np.array([x,y, np.zeros(len(phi))]).T	# r' vector
dl_prime = np.array([-2*pi*y/sections,2*pi*x/sections, np.zeros(len(phi))]).T	# dl' vector


#		QUESTIONS 5 and 6
def calc(l):
	R = r - r_prime[l] 	# Vector R = |r_ijk - r'_l|
	R = np.sqrt(np.sum(R**2, axis=-1))
	temp = np.cos(phi[l])*np.exp(-1j*k*R)/R
	temp = np.expand_dims(temp, axis=-1) 	#Add another dimension to the array so that the dl' vector can be broadcasted
	term = temp*dl_prime[l]
	return term


#		QUESTION 7

#Calculating Vector potential
A = np.zeros_like(r, dtype=np.complex128) 	#Initializing A
for l in range(dl_prime.shape[0]): # for loop is used since l is a vector 
    A += calc(l)


# 		QUESTION 8
B_z = (A[2,1,:,1]-A[1,2,:,0]-A[0,1,:,1]+A[1,0,:,0])/2 	#Calculating B using the given equation


#		QUESTION 9
plt.loglog(z,abs(B_z))	# log-log plot of magnetic field
plt.title("Loglog plot of Magnetic field along z-axis")
plt.xlabel("Distance along z axis")
plt.ylabel("B")
plt.grid(True)
plt.savefig("endsem_plot2.png")
plt.show()


#		QUESTION 10
M = c_[np.ones(990), np.log(z[10:])]
B_matrix = np.log(np.abs(B_z)[10:])
ans, residue, rank, s = lstsq(M, B_matrix,rcond=-1)	# lstsq(M,y) returns the answer for the matrix equation Mx=y
c = np.exp(ans[0])	# Since we get log(c) by the lstsq method.
b = ans[1]
print('c=',c, '\nb=',b)


#		QUESTION 11
#Comparing Magnetic field and the lstsq fit
plt.loglog(z,abs(B_z),'r-')
plt.loglog(z,c*(z**b),'b-')
plt.title("Comparing magnetic field with its fit")
plt.xlabel("z")
plt.ylabel("Magnetic field")
plt.legend(["Original Magnetic field","Least squares fit"])
plt.grid()
plt.savefig("endsem_plot3.png")
plt.show()
