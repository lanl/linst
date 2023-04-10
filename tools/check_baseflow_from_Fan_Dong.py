
def trapezoid_integration(fct, y):

    nn = len(y)
    wt = np.zeros(nn-1, dp)
    
    integral = 0.0 

    for i in range(0, nn-1):
       wt[i]  = ( y[i+1]-y[i] )/2.0 
       integral = integral + wt[i]*( fct[i] + fct[i+1] )

    return integral

def trapezoid_integration_cum(fct, y):

    nn = len(y)
    wt = np.zeros(nn-1, dp)
    
    integral = np.zeros(nn, dp)

    for i in range(0, nn-1):
       wt[i]       = ( y[i+1]-y[i] )/2.0
       integral[i+1] = integral[i] + wt[i]*( fct[i] + fct[i+1] )

    return integral


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def major_formatter(x, pos):
    return f'{x:.3f}'



#####################################################################################
#####################################################################################

import sys
import math
import time
import warnings
import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import ticker

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '16'
plt.rc('font', family='serif')

dp  = np.dtype('d')  # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

#####################################################
############# REQUIRED INPUTS #######################
#####################################################


# From paper:
# Multiple eigenmodes of the Rayleigh-Taylor instability observed for a fluid interface with smoothly
# varying density. III. Excitation and nonlinear evolution

A = 0.8 # Atwood number

nx   = 1001
xmin = -10
xmax = 10

x = np.linspace(xmin, xmax, nx)

delta  = A/( 2.0*( 1.0-np.sqrt(1-A**2.) ) )
pres_c = 50
x_c = xmin

dens_bsfl = (1.0-A*np.tanh(delta*x))/(1.0+A)
drhodx    = -A/(1.0+A)*delta/np.cosh(delta*x)**2.

pres_bsfl_num = trapezoid_integration_cum(dens_bsfl, x)
pres_bsfl_num = pres_bsfl_num + pres_c

pres_bsfl = pres_c + 1.0/(1.0+A)*( x-x_c + A/delta*np.log( np.cosh(delta*x_c)/np.cosh(delta*x) ) )

pres_bsfl_old = pres_bsfl

# print("delta*x = ", delta*x)
# print("np.cosh(delta*x) = ", np.cosh(delta*x))
# print("np.cosh(delta*x_c) = ", np.cosh(delta*x_c))
# print("np.cosh(delta*x_c)/np.cosh(delta*x) = ", np.cosh(delta*x_c)/np.cosh(delta*x))
# print("np.log( np.cosh(delta*x_c)/np.cosh(delta*x) ) = ", np.log( np.cosh(delta*x_c)/np.cosh(delta*x) ))

# Because in the paper they have x-axis to the right and gravity to the right, I need to re-arrange my pressure and density arrays

# print("pres_bsfl = ", pres_bsfl)

# dens_bsfl = dens_bsfl[nx::-1]
# pres_bsfl = pres_bsfl[nx::-1]
# pres_bsfl_num = pres_bsfl_num[nx::-1]

# print("pres_bsfl = ", pres_bsfl)

# for i in range(0, nx):
#     if ( abs( pres_bsfl_old[i]-pres_bsfl[nx-1-i] ) > 1.0e-14 ):
#         print("i = ", i)
#         print("Something wrong here")

# Take derivative of pressure and plot (see if you recover density)
dpdx = np.zeros(nx, dp)
drhodx_num = np.zeros(nx, dp)

for i in range(1, nx-1):
    dx = x[i+1]-x[i-1]
    dpdx[i] = ( pres_bsfl[i+1] - pres_bsfl[i-1] )/dx

    drhodx_num[i] = ( dens_bsfl[i+1] - dens_bsfl[i-1] )/dx
        
ptn = plt.gcf().number + 1

f = plt.figure(ptn)
plt.plot(x, dens_bsfl, 'k', linewidth=1.5, label="Baseflow density")
plt.plot(x[1:nx-2], dpdx[1:nx-2], 'r-.', linewidth=1.5, label="Numerical derivative of pressure")
plt.xlabel("x", fontsize=14)
plt.ylabel('density', fontsize=16)
plt.gcf().subplots_adjust(left=0.16)
plt.gcf().subplots_adjust(bottom=0.15)
#plt.ylim([xmin, xmax])
plt.legend(loc="upper right")
f.show()    

ptn = ptn + 1

f = plt.figure(ptn)
plt.plot(x, pres_bsfl, 'k', linewidth=1.5, label="Analytic integration")
plt.plot(x, pres_bsfl_num, 'r-.', linewidth=1.5, label="Numerical integration")
plt.xlabel(r"x", fontsize=14)
plt.ylabel('pressure', fontsize=16)
plt.gcf().subplots_adjust(left=0.16)
plt.gcf().subplots_adjust(bottom=0.15)
#plt.ylim([xmin, xmax])
plt.legend(loc="upper right")
f.show()

ptn = ptn + 1

f = plt.figure(ptn)
plt.plot(x, drhodx, 'k', linewidth=1.5, label="Analytical density derivative")
plt.plot(x[1:nx-2], drhodx_num[1:nx-2], 'r-.', linewidth=1.5, label="Numerical density derivative")
plt.xlabel("x", fontsize=14)
plt.ylabel('density derivative', fontsize=16)
plt.gcf().subplots_adjust(left=0.16)
plt.gcf().subplots_adjust(bottom=0.15)
#plt.ylim([xmin, xmax])
plt.legend(loc="upper right")
f.show()    


input("Plot")

