

def compute_inner_prod(f, g):

    inner_prod = 0.5*( np.multiply(np.conj(f), g) + np.multiply(f, np.conj(g)) )

    return inner_prod

def compute_der(y, var):

    ny = len(y)
    print("ny = ", ny)
    
    dvardz = np.zeros(ny, dpc)
    dvardz[0] = (var[1]-var[0])/(y[1]-y[0])
    dvardz[-1] = (var[-1]-var[-2])/(y[-1]-y[-2])

    for i in range(1, ny-1):

        dvardz[i] = (var[i+1]-var[i-1])/(y[i+1]-y[i-1])

    return dvardz

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

import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import ticker

import module_post_proc as module_pp

matplotlib.use('TkAgg') #----> Specify the backend

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

# Streamwise wavenumber
alpha = 100.
beta = 0.0

# File 1
ny = 351
filename1 = "/Users/aph/Desktop/Flow2/Eigenfunctions_Global_Solution_Sandoval.txt"
filename2 = "/Users/aph/Desktop/Flow2/Eigenfunctions_Global_Solution_Boussinesq.txt" 

y, u1, v1, w1, p1, r1 = module_pp.ReadEigenfunction(filename1, ny)
y, u2, v2, w2, p2, r2 = module_pp.ReadEigenfunction(filename2, ny)

# Compute derivative
dp1dz = compute_der(y, p1)
dp2dz = compute_der(y, p2)

w_dpdz1 = -compute_inner_prod(w1, dp1dz)
w_dpdz1 = -compute_inner_prod(w2, dp2dz)

ui_dpdxi1 = -compute_inner_prod(u1, 1j*alpha*p1) - compute_inner_prod(v1, 1j*beta*p1) - compute_inner_prod(w1, dp1dz)
ui_dpdxi2 = -compute_inner_prod(u2, 1j*alpha*p2) - compute_inner_prod(v2, 1j*beta*p2) - compute_inner_prod(w2, dp2dz)

##############################################
# PLOTS
##############################################

# Plot the eigenfunctions
ptn = plt.gcf().number + 1

f  = plt.figure(ptn)
ax = plt.subplot(111)
ax.plot(np.abs(u1), y, 'k', linewidth=1.5, label=r"$u$ (Sandoval=S)")
ax.plot(np.abs(w1), y, 'r', linewidth=1.5, label=r"$w$ (S)")
ax.plot(np.abs(p1), y, 'b', linewidth=1.5, label=r"$p$ (S)")
ax.plot(np.abs(r1), y, 'g', linewidth=1.5, label=r"$r$ (S)")

ax.plot(np.abs(u2), y, 'k--', linewidth=1.5, label=r"$u$ (Boussinesq=B)")
ax.plot(np.abs(w2), y, 'r--', linewidth=1.5, label=r"$w$ (B)")
ax.plot(np.abs(p2), y, 'b--', linewidth=1.5, label=r"$p$ (B)")
ax.plot(np.abs(r2), y, 'g--', linewidth=1.5, label=r"$r$ (B")

plt.xlabel("eigenfunctions amplitude", fontsize=14)
plt.ylabel("z", fontsize=14)
plt.gcf().subplots_adjust(left=0.18)
plt.gcf().subplots_adjust(bottom=0.15)
plt.ylim([-0.1, 0.1])

plt.legend(loc="upper right")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
          ncol=4, fancybox=True, shadow=True, fontsize=10)

f.show()



# Velocity-pressure correlation plot
ptn = plt.gcf().number + 1

f = plt.figure(ptn)
plt.plot(np.real(ui_dpdxi1), y, 'r', linewidth=1.5, label=r"$-u_i\dfrac{\partial p}{\partial x_i}$ (Sandoval)")
plt.plot(np.real(ui_dpdxi2), y, 'k--', linewidth=1.5, label=r"$-u_i\dfrac{\partial p}{\partial x_i}$ (Boussinesq)")

plt.xlabel(r"$-u'_idp'/dx_i$", fontsize=18)
plt.ylabel("z", fontsize=18)
#plt.title('Velocity-pressure correlation', fontsize=12)

plt.gcf().subplots_adjust(left=0.19)
plt.gcf().subplots_adjust(bottom=0.15)

ymin = -0.01
ymax = -ymin
plt.ylim([ymin, ymax])

f.show()

plt.legend(bbox_to_anchor=(1.15, 1.15), loc='upper right', fontsize=12)


input("End of prgram")




