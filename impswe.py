import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt


def smooth_ic(x):
    """ Initial Condition for Shallow water equations """
    nx = len(x)
    zeros = np.zeros(nx)
    ones = np.ones(nx)
    ic = np.where(np.logical_or(x <= 0.25, x >= 0.75), zeros,
                  0.5*(1+np.cos(4*np.pi*(x-0.5)))*ones)
    return ic


# In the code below we solve the linear shallow water equations
# on a staggered grid using an implicit methods

phiref = 1.0  # The mean geopotential

# Grid points
nx = 40
dx = 1./float(nx)

# Grid for phi values
xphi = np.arange(nx)*dx

# Grid for u values
xu = (np.arange(nx)+.5)*dx

# Set initial condition
phi = smooth_ic(xphi)
u = smooth_ic(xu)*np.sqrt(phiref)

# Set initial variables
lfirst = True
phim = np.copy(phi)
um = np.copy(u)
phip = np.copy(phi)
up = np.copy(u)

rhs = np.zeros(nx)

dt = .01
nu = dt*dt*phiref/(4*dx*dx)

# Set up the problem to use scipy's sparse solver the tridiagonal solve
dSuper = 
dMain  = 
dSub   = 
SolverDiags = 
SolverData =  
AMatrix = sparse.spdiags(SolverData, SolverDiags, nx, nx, format = 'csc')

# Set up plots
plt.plot(xphi,phi,'g-')
plt.plot(xu,u,'b-')

# number of steps to take
nstep = 100

for istep in range(nstep):
    # Set up the RHS
    rhs = 

    # Solve Helmholtz Problem
    phip = np.copy(spsolve(AMatrix,rhs))

    # Use new phi to compute new u
    up = 
    u = np.copy(up)
    phi = np.copy(phip)
        
plt.plot(xphi,phi,'go')
plt.plot(xu,u,'bo')
plt.show()




