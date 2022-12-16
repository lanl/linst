import sys
import math
import time
import warnings
import numpy as np
#from scipy import linalg
import class_gauss_lobatto as mgl
import class_baseflow as mbf
import class_build_matrices as mbm
import class_solve_gevp as msg
import class_mapping as mma
import matplotlib.pyplot as plt

import module_utilities as mod_util
import incomp_ns as mod_incomp

import class_main_arrays as marray

from matplotlib import cm
from matplotlib import ticker

# Grab Currrent Time Before Running the Code
start = time.time()

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '16'
plt.rc('font', family='serif')

dp  = np.dtype('d')  # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

###################################
#   Flags and reference strings   #
###################################

# Gravity
grav = 9.81

# rt_flag == True: "Rayleigh-Taylor" stability equations (from Chandrasekhar, see pp. 429 -> ...);
# rt_flag == False: Incompressible stability equations (Shear-layer and Poiseuille baseflows)
# rt_flag   = False

# only used if rt_flag is True.
# prim_form = 1: RT problem with primitive variable formulation,
# prim_form = 0: inviscid RT with w-equation only (see Chandrasekhar page 433) 
prim_form = 1

# Currently not working: keep to False
#Local     = True

# Plotting flags
plot_grid_bsfl = 0 # set to 1 to plot grid distribution and baseflow profiles
plot_eigvcts = 0 # set to 1 to plot eigenvectors ==> will be set to 1 if only one location and one alpha
plot_eigvals = 0 # set to 1 to plot eigenvalues

str_vars = np.array(['u-velocity', 'v-velocity', 'w-velocity', 'pressure'])

###################################
#   Reference data from Michalke  #
###################################

alp_mich = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.4446, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
ome_mich = np.array([0.0, 0.04184, 0.06975, 0.08654, 0.09410, 0.09485, 0.09376, 0.08650, 0.07305, 0.05388, 0.02942, 0.0], dtype=float)

#print("alp_mich, ome_mich = ", alp_mich, ome_mich)

###################################
#         Read input file         #
###################################

inFile = sys.argv[1]

rt_flag, SolverT, baseflowT, ny, Re_min, Re_max, npts_re, alp_min, alp_max, npts_alp, alpha, beta, yinf, lmap, target1 = mod_util.read_input_file(inFile)

Re_range = np.linspace(Re_min, Re_max, npts_re)

Local, plot_eigvcts = mod_util.set_local_flag_display_sizes(npts_alp, npts_re, plot_eigvcts)

# Create instance for main array omega
iarr = marray.MainArrays(npts_re, npts_alp)

iarr.re_array = Re_range

for i in range(0, npts_re):

    if (i > 0):
        target1 = iarr.omega_array[i-1,0]
        print("setting target to target = ",target1)
        
    Re = Re_range[i]
    mod_incomp.incomp_ns_fct(prim_form, Local, plot_grid_bsfl, plot_eigvcts, plot_eigvals, rt_flag, SolverT, \
                             baseflowT, ny, Re, alp_min, alp_max, npts_alp, alpha, beta, yinf, lmap, target1, alp_mich, ome_mich, npts_re, iarr, i, grav)


if ( Local and npts_alp > 1 and npts_re > 1 ):
    # Contour plot of stability banana
    [X, Y] = np.meshgrid(Re_range, alpha)
    
    #fig = plt.figure(figsize=(6,5))
    fig = plt.figure()
    #left, bottom, width, height = 0.16, 0.16, 0.8, 0.8
    #ax = fig.add_axes([left, bottom, width, height])
    
    Max_gr = np.amax(np.imag(iarr.omega_array))
    print("Maximum temporal growth rate = ", Max_gr)
    
    levels = np.linspace(0, Max_gr, 50)
    
    cp = plt.contourf(X, Y, np.transpose(np.imag(iarr.omega_array)), levels=levels, cmap=cm.jet) # ok: cm.jet, cm.magma, cm.inferno, cm.viridis, cm.hot, cm.afmhot, cm.Spectral cm.nipy_spectral ; ugly: cm.rainbow, cm.gist_rainbow, cm.plasma

    plt.xlabel("Reynolds number (Re)", fontsize=18)
    plt.ylabel(r"Wavenumber ($\alpha$)", fontsize=18)
    plt.title("Stability diagram")
    
    #ax.set_title('Stability diagram')
    #ax.set_xlabel('Reynolds number (Re)', fontsize=18)
    #ax.set_ylabel(r'$\alpha$', fontsize=18)
    fig.subplots_adjust(left=0.15)
    fig.subplots_adjust(bottom=0.16)
    plt.show()


if ( Local and npts_alp == 1 and npts_re > 1 ):

    ptn = plt.gcf().number + 1
    
    # f = plt.figure(ptn)
    # plt.plot(Re_range, np.imag(iarr.omega_array), 'k', markerfacecolor='none')
    # plt.xlabel(r'Reynolds number (Re)', fontsize=18)
    # plt.ylabel(r'$\omega_i$', fontsize=18)
    # plt.gcf().subplots_adjust(left=0.17)
    # plt.gcf().subplots_adjust(bottom=0.16)
    # #plt.xlim([-10, 10])
    # #plt.ylim([-1, 1])
    # f.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Re_range, np.imag(iarr.omega_array), 'k', markerfacecolor='none')
    
    #ax.set_title("scientific notation")
    #ax.set_yticks([0, 50, 100, 150])
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel(r'Reynolds number (Re)', fontsize=18)
    ax.set_ylabel(r'$\omega_i$', fontsize=18)
    # plt.xlabel(r'Reynolds number (Re)', fontsize=18)
    # plt.ylabel(r'$\omega_i$', fontsize=18)
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(bottom=0.16)
    #plt.xlim([0, 500])
    #plt.ylim([-0.05, 0.1])
    fig.show()

    input("DDDDDDD")


if ( Local and npts_alp > 1 and npts_re == 1 ):

    ptn = plt.gcf().number + 1
    
    # f = plt.figure(ptn)
    # plt.plot(Re_range, np.imag(iarr.omega_array), 'k', markerfacecolor='none')
    # plt.xlabel(r'Reynolds number (Re)', fontsize=18)
    # plt.ylabel(r'$\omega_i$', fontsize=18)
    # plt.gcf().subplots_adjust(left=0.17)
    # plt.gcf().subplots_adjust(bottom=0.16)
    # #plt.xlim([-10, 10])
    # #plt.ylim([-1, 1])
    # f.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(alpha, np.imag(iarr.omega_array[0,:]), 'k', markerfacecolor='none')
    
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel(r'Wavenumber ($\alpha$)', fontsize=18)
    ax.set_ylabel(r'$\omega_i$', fontsize=18)
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(bottom=0.16)
    #plt.xlim([0, 500])
    #plt.ylim([-0.05, 0.1])
    fig.show()

    input("DDDDDDD")

    
# Write stability banana
filename      = "Banana_tec_bottom.dat"
mod_util.write_stability_banana(npts_alp, npts_re, iarr, alpha, Re_range, filename)

# Grab Currrent Time After Running the Code
end = time.time()

#Subtract Start Time from The End Time
total_time = end - start
print("Total runtime: %21.11f [s]" % total_time)
