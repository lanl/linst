

def ReadTecplotFormattedDatta(filename):
    
    # Loading file data into numpy array and storing it in variable called data_collected
    data_collected = np.loadtxt(filename, skiprows=0, dtype=float)

    omegar = data_collected[:,0]
    omegai = data_collected[:,1]
        
    return omegar, omegai




#####################################################################################
#####################################################################################

import pylab
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

# Number of files to read
nfiles = 1

# File 1
filename1 = "eigenvalues_ny201.txt" #"eigenvalues_ny201_hyptan_alpha_0pt4446_Re5.txt" #"eigenvalues_ny201_poiseuille_alpha1_re2000.txt"
wr1, wi1  = ReadTecplotFormattedDatta(filename1)

if (nfiles == 2):
    # File 2
    filename2 = "eigenvalues_ny201_hyptan_alpha_0pt4446_Re500.txt"#"eigenvalues_ny201_poiseuille_alpha1_re10000.txt"

    wr2, wi2 = ReadTecplotFormattedDatta(filename2)

#####################################################
############# END REQUIRED INPUTS ###################
#####################################################

nl = 11
xl = np.linspace(-1000, 1000, nl)
yl = np.zeros(nl, dp)


#f = plt.figure(1)
fig, ppool = plt.subplots()

plt.plot(wr1, wi1, 'ks', markerfacecolor='none', label=r"$Re=5$")
if (nfiles == 2):
    plt.plot(wr2, wi2, 'bs', markerfacecolor='none', label=r"$Re=500$")
plt.plot(xl, yl, 'r')
plt.xlabel(r'$\omega_r$', fontsize=18)
plt.ylabel(r'$\omega_i$', fontsize=18)
plt.gcf().subplots_adjust(left=0.18)
plt.gcf().subplots_adjust(bottom=0.13)
#plt.title('Eigenvalue spectrum')
#plt.xlim([0, 1])
#plt.ylim([-0.6, 0.25])

plt.xlim([-0.1, 0.1])
plt.ylim([-1, 1])

if (nfiles == 2):
    plt.legend(loc="lower left", fontsize=15)


# opt = {'head_width': 0.01, 'head_length': 0.01, 'width': 0.2,'length_includes_head': True}

# for i in range(1, 360, 20):
#     x = math.radians(i)*math.cos(math.radians(i))
#     y = math.radians(i)*math.sin(math.radians(i))

# pylab.arrow(0.29, 0, 0, 0.3, alpha=0.01, **opt)

#plt.arrow(0.29, 0, 0, 0.3, width = 0.001, head_width = 0.01, head_length=0.05, facecolor='red', edgecolor='red')
plt.arrow(0.09, 0, 0, 0.2, width = 0.0005, head_width = 0.005, head_length=0.05, facecolor='red', edgecolor='red')

ppool.annotate('Unstable', xy =(0.04, 0.01), xytext =(0.04, 0.01), color='red')

fig.show()

input("End of program")
