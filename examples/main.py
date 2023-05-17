import sys
import math
import time
import warnings
import numpy as np
#from scipy import linalg
import class_gauss_lobatto as mgl
import class_baseflow as mbf
import class_build_matrices as mbm
import class_mapping as mma

import matplotlib
matplotlib.use('TkAgg') #----> Specify the backend
import matplotlib.pyplot as plt


import module_utilities as mod_util

import class_main_arrays as marray

from matplotlib import cm
from matplotlib import ticker

# Grab Currrent Time Before Running the Code
start = time.time()

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '16'
plt.rc('font', family='serif')

# Create instance for class GaussLobatto
cheb = mgl.GaussLobatto(size=251)
map = mma.MapShearLayer(sinf=10, cheb=cheb, l=0.2)
bsfl = mbf.RTSimple(y=map.y, At=0.4)

solver = mbm.Boussinesq(
    map=map,
    Re=1000,
    Fr=0.101936799,
    Sc=1.e50,
    bsfl=bsfl, 
    )

solver.solve(alpha=4.672, beta=0.0, omega_guess=1.e-12+0.816j)
solver.plot_eigvals()
plt.show()

###################################
#     Plot and write out data     #
###################################

# if ( Local and riist.npts_alp > 1 and bsfl_ref.npts_re > 1 ):
#     # Contour plot of stability banana
#     [X, Y] = np.meshgrid(Re_range, riist.alpha)
    
#     #fig = plt.figure(figsize=(6,5))
#     fig = plt.figure()
#     #left, bottom, width, height = 0.16, 0.16, 0.8, 0.8
#     #ax = fig.add_axes([left, bottom, width, height])
    
#     Max_gr = np.amax(np.imag(iarr.omega_array))
#     print("Maximum temporal growth rate = ", Max_gr)
    
#     levels = np.linspace(0, Max_gr, 50)
    
#     cp = plt.contourf(X, Y, np.transpose(np.imag(iarr.omega_array)), levels=levels, cmap=cm.jet) # ok: cm.jet, cm.magma, cm.inferno, cm.viridis, cm.hot, cm.afmhot, cm.Spectral cm.nipy_spectral ; ugly: cm.rainbow, cm.gist_rainbow, cm.plasma

#     plt.xlabel("Reynolds number (Re)", fontsize=18)
#     plt.ylabel(r"Wavenumber ($\alpha$)", fontsize=18)
#     plt.title("Stability diagram")
    
#     #ax.set_title('Stability diagram')
#     #ax.set_xlabel('Reynolds number (Re)', fontsize=18)
#     #ax.set_ylabel(r'$\alpha$', fontsize=18)
#     fig.subplots_adjust(left=0.15)
#     fig.subplots_adjust(bottom=0.16)
#     plt.show()


# if ( Local and riist.npts_alp == 1 and bsfl_ref.npts_re > 1 ):

#     ptn = plt.gcf().number + 1
    
#     # f = plt.figure(ptn)
#     # plt.plot(Re_range, np.imag(iarr.omega_array), 'k', markerfacecolor='none')
#     # plt.xlabel(r'Reynolds number (Re)', fontsize=18)
#     # plt.ylabel(r'$\omega_i$', fontsize=18)
#     # plt.gcf().subplots_adjust(left=0.17)
#     # plt.gcf().subplots_adjust(bottom=0.16)
#     # #plt.xlim([-10, 10])
#     # #plt.ylim([-1, 1])
#     # f.show()

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(Re_range, np.imag(iarr.omega_array), 'k', markerfacecolor='none')
    
#     #ax.set_title("scientific notation")
#     #ax.set_yticks([0, 50, 100, 150])
#     formatter = ticker.ScalarFormatter(useMathText=True)
#     formatter.set_scientific(True)
#     formatter.set_powerlimits((-1,1))
#     ax.yaxis.set_major_formatter(formatter)

#     ax.set_xlabel(r'Reynolds number (Re)', fontsize=18)
#     ax.set_ylabel(r'$\omega_i$', fontsize=18)
#     # plt.xlabel(r'Reynolds number (Re)', fontsize=18)
#     # plt.ylabel(r'$\omega_i$', fontsize=18)
#     plt.gcf().subplots_adjust(left=0.15)
#     plt.gcf().subplots_adjust(bottom=0.16)
#     #plt.xlim([0, 500])
#     #plt.ylim([-0.05, 0.1])
#     fig.show()

#     input("DDDDDDD")


# if ( Local and riist.npts_alp > 1 and bsfl_ref.npts_re == 1 ):

#     ptn = plt.gcf().number + 1
    
#     # f = plt.figure(ptn)
#     # plt.plot(Re_range, np.imag(iarr.omega_array), 'k', markerfacecolor='none')
#     # plt.xlabel(r'Reynolds number (Re)', fontsize=18)
#     # plt.ylabel(r'$\omega_i$', fontsize=18)
#     # plt.gcf().subplots_adjust(left=0.17)
#     # plt.gcf().subplots_adjust(bottom=0.16)
#     # #plt.xlim([-10, 10])
#     # #plt.ylim([-1, 1])
#     # f.show()

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(riist.alpha, np.imag(iarr.omega_array[0,:]), 'k', markerfacecolor='none')
    
#     formatter = ticker.ScalarFormatter(useMathText=True)
#     formatter.set_scientific(True)
#     formatter.set_powerlimits((-1,1))
#     ax.yaxis.set_major_formatter(formatter)

#     ax.set_xlabel(r'Wavenumber ($\alpha$)', fontsize=18)
#     ax.set_ylabel(r'$\omega_i$', fontsize=18)
#     plt.gcf().subplots_adjust(left=0.15)
#     plt.gcf().subplots_adjust(bottom=0.16)
#     #plt.xlim([0, 500])
#     #plt.ylim([-0.05, 0.1])
#     fig.show()

#     input("DDDDDDD")

    
# # Write stability banana
# filename      = "Banana_tec_bottom.dat"
# mod_util.write_stability_banana(riist.npts_alp, bsfl_ref.npts_re, iarr, riist.alpha, Re_range, filename)

# # Grab Currrent Time After Running the Code
# end = time.time()

# #Subtract Start Time from The End Time
# total_time = end - start
# print("Total runtime: %21.11f [s]" % total_time)




























