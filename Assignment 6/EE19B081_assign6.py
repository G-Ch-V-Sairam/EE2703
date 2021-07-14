'''						ASSIGNMENT-6 : 1-D SIMULATION OF A TUBELIGHT
							G Ch V Sairam , EE19B081	'''	

import numpy as np
import matplotlib.pyplot as plt			#Importing various modules required
from tabulate import tabulate
from sys import argv

if len(argv)==7:
	n=int(argv[1])
	M=int(argv[2])						#Taking all the inputs from the commandline.
	nk=int(argv[3])
	u0=int(argv[4])
	p=float(argv[5])
	Msig=float(argv[6])
	
else:
	n=100	#Spatial grid size
	M=5		#Number of electrons injected per turn
	nk=500	#Number of turns to simulate
	u0=7	#Threshold velocity
	p=0.5	#Probability that ionization will occur
	Msig=2	#Standard deviation
	print("Default values are being used. If you want to input different values , type all the input values in the order n,M,nk,uo,p,Msig .")


xx=np.zeros(n*M)
u=np.zeros(n*M)					#First, let the position, velocity, displacement vectors be zero matrices. We will update them gradually.
dx=np.zeros(n*M)

I=[]
X=[]							#Similarly , the light intensity, electron postion, electron velocity be empty lists. We will add values to the lists using .extend .
V=[]

ii=np.where(xx>0)				#We find all those electrons whose position is greater than zero. 

for k in range(nk): 	
	dx[ii] = u[ii] + 0.5	#Displacement increases due to electric field.
	xx[ii] += dx[ii]	#Advance the electron position and velocity for the turn.
	u[ii]  += 1
	
	jj = np.where(xx >= n)	#Determine the particles which have already hit the anode.
	xx[jj] = 0	#Set their position and velocity to zero.		
	u[jj]  = 0
	
	#This block finds the electrons which have ionised.
	kk = np.where(u >= u0)[0]
	ll = np.where(np.random.rand(len(kk))<=p)
	kl = kk[ll]
	
	u[kl] = 0	#They suffered an inelastic collision.Set their velocities to zero.
	xx[kl] -= dx[kl]*np.random.rand(len(kl))	#Try to determine the actual point of collision using random number.
	
	I.extend(xx[kl].tolist())	#These photons resulted an emission from that point. So add this to the I list.
	
	m  = int(np.random.randn()*Msig+M)	#Inject M new electrons. And determine the actual number of electrons.
	mm = np.where(xx == 0)	#Find the elctrons whose positions are zero.
	
	minimum = min(len(mm[0]),m)	#Find the minimum of , number of electrons to be added and the number of slots available.
	xx[mm[0][:minimum]] = 1 	#Set their position to 1 and velocity to zero.
	u[mm[0][:minimum]]  = 0 
	
	ii = np.where(xx > 0)
	X.extend(xx[ii].tolist())	#Add their positions to the X and V vectors.
	V.extend(u[ii].tolist())		
			
plt.figure(0)
plt.hist(X,bins=n,cumulative=False,edgecolor='black')		#Plot a histogram specifying the electron density
plt.title("Electron density")		#Set the title
plt.xlabel("$x$")					#set the x label
plt.ylabel("Number of electrons")	#Set the y label
plt.savefig("assn6_plot0.png")


plt.figure(1)
count,bins,trash = plt.hist(I,bins=n,edgecolor='black')		#Plot a histogram of the emission intensity of light
plt.title("Emission Intensity")
plt.xlabel("$x$")
plt.ylabel("I")
plt.savefig("assn6_plot1.png")

plt.figure(2)
plt.scatter(X,V,marker='x')				#Plot the elctron phase space.
plt.title("Electron Phase Space")
plt.xlabel("Position")
plt.ylabel("Velocity")
plt.savefig("assn6_plot2.png")

plt.show()

xpos=0.5*(bins[0:-1]+bins[1:])		#Converting into midpoint values
print("Intensity data:")
print("xpos\tcount")
for k in range(len(count)):
	print(str(float("{0:.2f}". format(xpos[k])))+'\t'+str(int(count[k])))		#Print the table-like data showing population counts and the bin position

