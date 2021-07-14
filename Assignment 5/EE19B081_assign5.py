from matplotlib.pylab import *
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sys import argv

#If the user provides any inputs , we must use them , else we must use the given default values.
if len(argv)==5:
    Nx=int(argv[1])
    Ny=int(argv[2])
    radius=int(argv[3])  
    Niter=int(argv[4])
else:
    Nx=25
    Ny=25
    radius=8
    Niter=1500
    print("Default values specified are being used here. If you want to use your own parameters , specify all the 4 parameters.")

phi=np.zeros((Nx,Ny),dtype = float)
x = np.linspace(-0.5,0.5,Ny)
y = np.linspace(-0.5,0.5,Nx)
Y,X = np.meshgrid(y,x)
ii=np.where(X**2+Y**2<=((radius/Nx)**2))		#Points inside the central lead
phi[ii]=1.0										#Set the potential inside the central lead to 1.0

plt.figure(1)
contourf(X,Y,phi)								#A contour plot is a graphical technique for representing a 3-dimensional surface by plotting constant z slices.
plot(x[ii[0]],y[ii[1]],'ro')					#Indicate the points inside central lead by red dots.
title("Contour plot of Potential")
xlabel("X")
ylabel("Y")
colorbar()
savefig("Assgn5_plot1.png")

errors=[]
for k in range(Niter):
	oldphi=phi.copy()
	phi[1:-1,1:-1]=0.25*(phi[1:-1,0:-2]+ phi[1:-1,2:]+ phi[0:-2,1:-1] + phi[2:,1:-1])	#Updating phi
	phi[1:-1,0]=phi[1:-1,1]
	phi[1:-1,Nx-1]=phi[1:-1,Nx-2]														#Asserting the boundaries
	phi[0,1:-1]=phi[1,1:-1]
	phi[ii]=1.0
	errors.append(np.max(np.abs(phi-oldphi)))
	if errors[k]==0:
		print("The steady state has been reached at ",k)
		break

'''logy=logA+Bx
	To find A,B , we create 2 matrices such that logy=[logA B]*(transpose of [1 x])
	We use least squaeres method to solve this matrix equation.
	
'''
	
def fit_error(x,y):
	logy=np.log(y)
	mat_x=np.zeros((len(x),2))
	mat_x[:,0]=x
	mat_x[:,1]=1
	B,logA=lstsq(mat_x, np.transpose(logy),rcond=None)[0]
	return (np.exp(logA),B)
   
def net_error(a,b,N):
	return np.abs(a/b*np.exp(b*(N+0.5)))			#Find cummulative error

def fit_exp(x,a,b):
	return A*np.exp(B*x)
	
A,B=fit_error(range(Niter),errors)
A_500,B_500=fit_error(range(Niter)[500:],errors[500:])

plt.figure(2)
semilogy(range(Niter),errors)														#Semilog plot of iteratons vs. errors
semilogy(range(Niter)[::50],errors[::50],'ro')										#Fit1
semilogy(range(Niter)[::50],fit_exp(range(Niter)[::50],A,B),'go',markersize=4)		#Fit2 
title("Semilog plot of errors")
xlabel("Number of iterations")
ylabel("log(Error)")
legend(['Original','fit1','fit2'])
grid()
savefig("Assgn5_plot2.png")

plt.figure(3)
loglog(range(Niter),errors)
loglog(range(Niter)[::50],errors[::50],'ro')
loglog(range(Niter)[::50],fit_exp(range(Niter)[::50],A,B),'go',markersize=4)
title("Loglog plot of errors")
xlabel("log(Number of iterations)")
ylabel("log(Error)")
legend(['Original','fit1','fit2'])
grid()
savefig("Assgn5_plot3.png")

plt.figure(4)
title(r'Plot of Cumulative Error values On a loglog scale')
loglog(range(Niter)[::50],np.abs(net_error(A_500,B_500,np.arange(Niter)[::50])),'ro')
xlabel("iterations")
ylabel("Net  maximum error")
grid()
legend(['Cumulative error values'])
savefig("Assgn5_plot4.png")

plt.figure(5)
#plotting 2d contour of final potential
title("2D Contour plot of final potential")
xlabel("X")
ylabel("Y")
plot(ii[0]/Nx -0.5,ii[1]/Ny -0.5,'ro')
contourf(Y,X[::-1],phi)										#Updated contour plot of phi
colorbar()
savefig("Assgn5_plot5.png")

#plotting 3d contour of final potential
fig1=plt.figure(6)     
ax=plt.axes(projection='3d')
plt.title('The 3-D surface plot of the potential')
surf = ax.plot_surface(Y, X[::-1], phi, rstride=1, cstride=1, cmap=plt.cm.jet)
savefig("Assgn5_plot6.png")

Jx = np.zeros((Ny,Nx))
Jy = Jx.copy()
#Creating the current vectors
Jx[1:-1,1:-1] = 0.5*(phi[1:-1,0:-2] - phi[1:-1, 2:])
Jy[1:-1,1:-1] = 0.5*(phi[2:,1:-1] - phi[0:-2, 1:-1])

plt.figure(7)
plot(ii[0]/Nx -0.5,ii[1]/Ny -0.5,'ro')
xlabel('X')
ylabel('Y')
title('Vector Plot of Currents')
quiver(Y,X[::-1],Jx,Jy, scale=5)	#Plot the current vectors
savefig("Assgn5_plot7.png")


plt.show()
