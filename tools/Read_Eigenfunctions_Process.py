

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

def ReadEigenfunction(filename, ny):
    
    # Loading file data into numpy array and storing it in variable called data_collected
    data_collected = np.loadtxt(filename, skiprows=0, dtype=float)

    y = data_collected[:,0]
    
    # amplitude
    ua = data_collected[:,1]
    va = data_collected[:,2]
    wa = data_collected[:,3]
    pa = data_collected[:,4]
    ra = data_collected[:,5]

    # real parts
    ur = data_collected[:,1+5]
    vr = data_collected[:,2+5]
    wr = data_collected[:,3+5]
    pr = data_collected[:,4+5]
    rr = data_collected[:,5+5]

    # imag parts
    ui = data_collected[:,1+10]
    vi = data_collected[:,2+10]
    wi = data_collected[:,3+10]
    pi = data_collected[:,4+10]
    ri = data_collected[:,5+10]

    u = ur+1j*ui
    v = vr+1j*vi
    w = wr+1j*wi
    p = pr+1j*pi
    r = rr+1j*ri

    return y, u, v, w, p, r


#####################################################################################
#####################################################################################

import sys
import math
import time
import warnings
import numpy as np

import matplotlib.pyplot as plt

import module_utilities as mod_util
import incomp_ns as mod_incomp

import class_main_arrays as marray

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

# Streamwise wavenumber
alpha = 100.
beta = 0.0

# File 1
ny = 351
filename1 = "/Users/aph/Desktop/Flow2/Eigenfunctions_Global_Solution_Sandoval.txt"
filename2 = "/Users/aph/Desktop/Flow2/Eigenfunctions_Global_Solution_Boussinesq.txt" 

y, u1, v1, w1, p1, r1 = ReadEigenfunction(filename1, ny)
y, u2, v2, w2, p2, r2 = ReadEigenfunction(filename2, ny)

# Compute derivative
dp1dz = compute_der(y, p1)
dp2dz = compute_der(y, p2)

w_dpdz1 = -compute_inner_prod(w1, dp1dz)
w_dpdz1 = -compute_inner_prod(w2, dp2dz)

ui_dpdxi1 = -compute_inner_prod(u1, 1j*alpha*p1) - compute_inner_prod(v1, 1j*beta*p1) - compute_inner_prod(w1, dp1dz)
ui_dpdxi2 = -compute_inner_prod(u2, 1j*alpha*p2) - compute_inner_prod(v2, 1j*beta*p2) - compute_inner_prod(w2, dp2dz)


ptn = plt.gcf().number + 1

f = plt.figure(ptn)
#plt.plot(np.real(w_dpdz1), y, 'k', linewidth=1.5, label=r"-w_dpdz")
plt.plot(np.real(ui_dpdxi1), y, 'r', linewidth=1.5, label=r"$-u_i\dfrac{\partial p}{\partial x_i}$ (Sandoval)")
plt.plot(np.real(ui_dpdxi2), y, 'k--', linewidth=1.5, label=r"$-u_i\dfrac{\partial p}{\partial x_i}$ (Boussinesq)")
#plt.plot(np.real(RHS), y, 'r--', linewidth=1.5, label=r"RHS")

plt.xlabel("u'*dp'/dz", fontsize=18)
plt.ylabel("z", fontsize=18)

plt.gcf().subplots_adjust(left=0.17)
plt.gcf().subplots_adjust(bottom=0.15)

#ymin = -0.005
#ymax = -ymin
#plt.ylim([ymin, ymax])

f.show()

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper right', fontsize=8)
    



input("End of prgram")




