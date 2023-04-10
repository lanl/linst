import os
import sys
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use('TkAgg') #----> Specify the backend
from scipy import linalg
import matplotlib.pyplot as plt

import module_post_proc as module_pp

sys.path.append('../src')
import class_gauss_lobatto as mgl
import class_readinput as rinput
import class_baseflow as mbf
import class_mapping as mma

from collections import Counter

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '15'
plt.rc('font', family='serif')

#matplotlib.rcParams['text.usetex'] = True

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

#####################################################
############# REQUIRED INPUTS #######################
#####################################################

# Directory where eigenfunctions and input files are:
file_dir = "/Users/aph/Documents/MATLAB/Two_eigenfunctions_alpha1_beta1_alpha1_betaminus1/"

# Name of input file
file_input = "input_rayleigh_taylor_re500000_alpha1_beta1.dat"

# Name of eigenfunction files
f1 = "Eigenfunctions_Global_Solution_nondimensional_351_pts_alpha1_beta1.txt"
f2 = "Eigenfunctions_Global_Solution_nondimensional_351_pts_alpha1_betaminus1.txt"

# Solver type
boussinesq = 1

# X, Y dimensions
nx = 101
ny = 101
# nz is determined by eigenfunction file size and input file

# Number of disturbance wavelength for x and y grids
nwav = 4

#####################################################
############# EXECUTABLE STATEMENTS #################
#####################################################

#
# Read input file
#
riist = module_pp.ReadInputFile(file_dir, file_input)

#
# Set reference quantities
#

# Set reference quantities from non-dimensional parameters and Uref (ref. velocity) and gref (ref. acceleration)
bsfl_ref = mbf.Baseflow()
bsfl_ref.set_reference_quantities(riist)

#
# Get z-coordinate and differentiation matrix in z-direction
#

map = module_pp.GetDifferentiationMatrix(riist.ny, riist, bsfl_ref, boussinesq)

#
# Read eigenfunction data
#

# Generate full file name
f1_full = os.path.join(file_dir, f1)
f2_full = os.path.join(file_dir, f2)

# Read data
z1, ef1 = module_pp.ReadEigenfunction(f1_full, riist.ny)
z2, ef2 = module_pp.ReadEigenfunction(f2_full, riist.ny)

#
# Generate x/y grids
#
xgrid, ygrid = module_pp.GenerateGridsXY(nx, ny, nwav, riist.alpha[0], riist.beta)

#
# Compute total disturbance field
#

print("alpha = ", riist.alpha[0])
print("beta = ", riist.beta)

dist_3d = module_pp.ComputeDistField_2waves(riist.ny, ef1, ef2, riist.alpha[0], riist.beta, xgrid, ygrid)

#
# Write out tecplot file for 3d disturbance field of 2 wave superposition
#
zgrid = map.y
module_pp.Write3dDistFlow(xgrid, ygrid, zgrid, dist_3d, file_dir)


#print("map.y = ", map.y)
#print("map.D1 = ", map.D1)
