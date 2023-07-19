import sys
import math
import time
import warnings
import numpy as np
#from scipy import linalg
import gauss_lobatto as mgl
import baseflow as mbf
import build_matrices as mbm
import mapping as mma

import matplotlib
matplotlib.use('TkAgg') #----> Specify the backend
import matplotlib.pyplot as plt

import module_utilities as mod_util

from matplotlib import cm
from matplotlib import ticker

# Grab Currrent Time Before Running the Code
start = time.time()

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '16'
plt.rc('font', family='serif')

# Create instance for class GaussLobatto
cheb = mgl.GaussLobatto(size=401)
#map = mma.MapShearLayer(sinf=500, cheb=cheb, l=10.0)
map = mma.MapShearLayer(sinf=12., cheb=cheb, l=3.0)
bsfl = mbf.RTPsDNS(y=map.y, At=0.05, map=map)

solver = mbm.Boussinesq(
    map=map,
    #Re=400,
    Re=1000,
    Fr=1.0,
    Sc=1.0e50,
    #Sc=1.0e50,
    bsfl=bsfl,
    )

#print(str(solver))

#solver.solve(alpha=np.linspace(0.9,0.9,1), beta=0.9, omega_guess=0.0+0.056263*1j)
#solver.solve(alpha=np.linspace(0.9,0.9,1), beta=0.0, omega_guess=0.0+0.1471*1j)
#solver.solve(alpha=np.linspace(0.5,0.5,1), beta=0.0, omega_guess=0.0+0.12753*1j)
#solver.solve(alpha=np.linspace(0.1,0.1,1), beta=-0.1, omega_guess=0.0+0.12753*1j)
#solver.solve(alpha=np.linspace(2.4,2.4,1), beta=2.4, omega_guess=0.0+0.20142*1j)
solver.solve(alpha=np.linspace(2.4,2.4,1), beta=2.4, omega_guess=0.0+0.194639*1j)
solver.write_stab_banana()
#solver.plot_eigvals()
#plt.show()
#solver.write_eigvals()

#q_eigvect = solver.identify_eigenvector_from_target()
#solver.get_normalized_eigvects(q_eigvect, bsfl.rt_flag)
#solver.plot_eigvects(bsfl.rt_flag)
#solver.write_eigvects_out_new()

#solver.write_baseflow_out(solver)

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




























