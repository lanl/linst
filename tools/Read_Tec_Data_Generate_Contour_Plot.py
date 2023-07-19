

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def major_formatter(x, pos):
    return f'{x:.3f}'

def ReadTecplotFormattedDatta(filename, nx, ny):
    
    # Loading file data into numpy array and storing it in variable called data_collected
    data_collected = np.loadtxt(filename, skiprows=3, dtype=float)

    # Printing data stored
    #print(data_collected)
    
    #print(data_collected.shape)
    
    #print("data_collected.dtype = ",data_collected.dtype)
    
    var1_tmp = data_collected[:,0]
    var2_tmp = data_collected[:,1]
    
    var3_tmp = data_collected[:,2]
    var4_tmp = data_collected[:,3]
    
    
    var1 = np.zeros((nx,ny), dp)
    var2 = np.zeros((nx,ny), dp)
    
    var3 = np.zeros((nx,ny), dp)
    var4 = np.zeros((nx,ny), dp)
    
    #print(var1.shape)
    
    count = 0
    
    for j in range(0,ny):
        
        for i in range(0,nx):
            
            var1[i,j] = var1_tmp[count]
            var2[i,j] = var2_tmp[count]
            
            var3[i,j] = var3_tmp[count]
            var4[i,j] = var4_tmp[count]
            
            count = count + 1
        
    return var1, var2, var3, var4


def ReadTecplotFormattedEigenfunction_2D(filename, nx, ny):
    
    # Loading file data into numpy array and storing it in variable called data_collected
    data_collected = np.loadtxt(filename, skiprows=3, dtype=float)

    # Printing data stored
    #print(data_collected)
    
    #print(data_collected.shape)
    
    #print("data_collected.dtype = ",data_collected.dtype)
    
    xx = data_collected[:,0]
    zz = data_collected[:,1]
    
    rho_d = data_collected[:,2]
    
    x = np.zeros((nx,ny), dp)
    z = np.zeros((nx,ny), dp)
    
    rho_dist = np.zeros((nx,ny), dp)
    
    #print(renum.shape)
    
    count = 0
    
    for j in range(0,ny):
        
        for i in range(0,nx):
            
            x[i,j] = xx[count]
            z[i,j] = zz[count]
            
            rho_dist[i,j] = rho_d[count]
            
            count = count + 1
        
    return x, z, rho_dist


#####################################################################################
#####################################################################################

import sys
import math
import time
import warnings
import numpy as np

import matplotlib.pyplot as plt

#import module_utilities as mod_util
#import incomp_ns as mod_incomp

#import class_main_arrays as marray

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

# Type of plot
tplot  = 2 # 1 -> line plot, 2 -> contour plot, 3 -> line plot with alpha as x-axis, 4 -> 2d (reconstructed) eigenfunctions

# Number of files to read
nfiles = 1

# File names and dimensions

# File 1
nx1 = 30
ny1 = 30
filename1 = "/Users/aph/Refactor/linst/linst/Stability_banana_rt_2d.dat" #"Eigenfunction_2d_rho_ny351.dat" #"Banana_tec_bottom_rt_atw0pt5_atwmu0pt5_ny201.dat" #"Banana_tec_poiseuille_top.dat" #"Banana_tec_alpha_1.dat"  "Banana_tec_bottom_rt_atw1over3_atwmu1over3_ny181.dat" #

if (tplot == 4):
    x, z, rho_dist = ReadTecplotFormattedEigenfunction_2D(filename1, nx1, ny1)
else:
    var1, var2, var3, var4 = ReadTecplotFormattedDatta(filename1, nx1, ny1)

if (nfiles == 2 and tplot != 4):
    # File 2
    nx2 = 1
    ny2 = 101
    filename2 = "Banana_tec_bottom_rt_atw1over3_atwmu1over3_ny201.dat" #"Banana_tec_bottom_rt_atw0pt5_atwmu0pt5_ny231.dat" #"Banana_tec_poiseuille_bottom.dat"

    re2, alp2, wr2, wi2 = ReadTecplotFormattedDatta(filename2, nx2, ny2)

#####################################################
############# END REQUIRED INPUTS ###################
#####################################################

if   (tplot==1): # line plot

    if (ny1 != 1):
        sys.exit("For a line plot, ny1 has to be equal to 1!!!!!")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(re1, wi1, 'k', markerfacecolor='none')
    
    #ax.set_title("scientific notation")
    #ax.set_yticks([0, 50, 100, 150])
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel(r'Reynolds number (Re)', fontsize=18)
    ax.set_ylabel(r'$\omega_i$', fontsize=18)
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(bottom=0.16)
    plt.xlim([5000, 50000])
    #plt.ylim([-0.05, 0.1])
    fig.show()

elif (tplot==2): # contour plot

    # Find max growth rate
    print("var4.shape = ", var4.shape)

    ijdx = np.unravel_index(np.argmax(var4), var4.shape)
    idx = ijdx[0]
    jdx = ijdx[1]
    
    print("idx, jdx = ", idx, jdx)
    
    fig = plt.figure()
    
    Max_gr1 = np.amax(var4)
    Min_gr1 = np.amin(var4)
    #Max_gr2 = np.amax(wi2)
    print("Maximum temporal growth rate = ", Max_gr1)
    print("Mininum temporal growth rate = ", Min_gr1)
    print("alpha, beta with max. growth rate: ", var2[idx,jdx], var1[idx,jdx])

    print("growth rate var4[idx,jdx]: ", var4[idx,jdx])
    print("growth rate var4[12,12]: ", var4[12,12])
    print("growth rate var4[13,13]: ", var4[13,13])
    #print("Maximum temporal growth rate = ", Max_gr2)
    
    #Max_gr = np.maximum(Max_gr1, Max_gr2)
    
    #levels = np.linspace(0, Max_gr, 50)
    levels = np.linspace(Min_gr1, Max_gr1, 50)

    # ok: cm.jet, cm.magma, cm.inferno, cm.viridis, cm.hot, cm.afmhot, cm.Spectral cm.nipy_spectral ; ugly: cm.rainbow, cm.gist_rainbow, cm.plasma
    cp = plt.contourf(var2, var1, var4, levels=levels, cmap=cm.jet) 
    #cp = plt.contourf(re2, alp2, wi2, levels=levels, cmap=cm.jet) 
    
    plt.xlabel("alpha", fontsize=18)
    plt.ylabel(r"beta", fontsize=18)
    #plt.title("Stability diagram")
    
    #ax.set_title('Stability diagram')
    #ax.set_xlabel('Reynolds number (Re)', fontsize=18)
    #ax.set_ylabel(r'$\alpha$', fontsize=18)
    fig.subplots_adjust(left=0.15)
    fig.subplots_adjust(bottom=0.16)
    #fig.subplots_adjust(right=0.88)

    #plt.colorbar()
    #plt.colorbar(format=ticker.FuncFormatter(fmt))
    plt.colorbar(format=ticker.FuncFormatter(major_formatter))
    
    plt.show()


elif   (tplot==3): # line plot

    if (nx1 != 1):
        sys.exit("For a line plot with alpha as x-axis nx1 should be 1")

    print("wi1=",wi1)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(alp1[0,:], wi1[0,:], 'k', markerfacecolor='none', label="$Atw=1/2$")
    ax.plot(alp2[0,:], wi2[0,:], 'r', markerfacecolor='none', label="$Atw=1/3$")
    #ax.set_title("scientific notation")
    #ax.set_yticks([0, 50, 100, 150])
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel(r'Dimensional Wavenumber (k)', fontsize=18)
    ax.set_ylabel(r'Dimensional growth rate ($\omega_i$)', fontsize=18)
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(bottom=0.16)
    #plt.xlim([5000, 50000])
    #plt.ylim([-0.05, 0.1])
    plt.legend(loc="upper right")
    fig.show()

elif (tplot==4): # contour plot ==> 2D eigenfunctions
    
    fig = plt.figure()
    
    Max_ct1 = np.amax(np.abs(rho_dist))
    print("Maximum = ", Max_ct1)
    
    levels = np.linspace(-Max_ct1, Max_ct1, 50)
    
    cp = plt.contourf(x, z, rho_dist, levels=levels, cmap=cm.jet)
    
    plt.xlabel(r"x", fontsize=18)
    plt.ylabel(r"z", fontsize=18)
    #plt.title("Stability diagram")
    
    #ax.set_title('Stability diagram')
    #ax.set_xlabel('Reynolds number (Re)', fontsize=18)
    #ax.set_ylabel(r'$\alpha$', fontsize=18)
    fig.subplots_adjust(left=0.15)
    fig.subplots_adjust(bottom=0.16)
    #fig.subplots_adjust(right=0.88)

    #plt.colorbar()
    #plt.colorbar(format=ticker.FuncFormatter(fmt))
    plt.colorbar(format=ticker.FuncFormatter(major_formatter))

    #plt.xlim([5000, 50000])
    plt.ylim([-0.00025, 0.00025])

    plt.show()

    
else:
    print("Not a proper value for flag 'tplot'")
    
input("End of prgram")




