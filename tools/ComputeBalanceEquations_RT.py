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
#import module_utilities as mod_util
import class_build_matrices as mbm
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
#file_dir = "/Users/aph/Documents/MATLAB/Two_eigenfunctions_alpha1_beta1_alpha1_betaminus1/"

#file_dir = "/Users/aph/incompressible-stability-equations/src/Re_50_Reh_0pt6_TwoWaves"
#file_dir = "/Users/aph/incompressible-stability-equations/src/Re_500_Reh_6_TwoWaves"
file_dir = "/Users/aph/incompressible-stability-equations/src/Re_5000_Reh_60_TwoWaves"

# Name of input file
#file_input = "input_rayleigh_taylor_re500000_alpha1_beta1.dat"

#file_input = "input_rayleigh_taylor_re50_k5pt85_alpha_beta.dat"
#file_input = "input_rayleigh_taylor_re500_k26pt3_alpha_beta.dat"
file_input = "input_rayleigh_taylor_re5000_k95pt5_alpha_beta.dat"



# Name of eigenfunction files
f1 = "Eigenfunctions_Global_Solution_nondimensional_451_pts_alpha1_beta1.txt"
f2 = "Eigenfunctions_Global_Solution_nondimensional_451_pts_alpha1_betaminus1.txt"

# Solver type
boussinesq = 1

# X, Y dimensions
nx = 101
ny = 101
# nz is determined by eigenfunction file size and input file

# Number of disturbance wavelength for x and y grids
nwav = 4

# Growth rate
#omega_i = 3.5380691664356796
#omega_i = 7.46559899943202
omega_i = 13.502375227052253

#omega_i = 21.178342746597068

#####################################################
############# EXECUTABLE STATEMENTS #################
#####################################################

#
# Read input file
#
riist = module_pp.ReadInputFile(file_dir, file_input)

nz = riist.ny

print("riist.target.imag = ", riist.target.imag)
print("omega_i =", omega_i)

if (riist.target.imag != omega_i):
    sys.exit("Growth rate provided not consistent with input file")
    
#
# Set reference quantities
#

# Set reference quantities from non-dimensional parameters and Uref (ref. velocity) and gref (ref. acceleration)
bsfl_ref = mbf.Baseflow()
bsfl_ref.set_reference_quantities(riist)

#
# Get z-coordinate and differentiation matrix in z-direction
#
map = module_pp.GetDifferentiationMatrix(nz, riist, bsfl_ref, boussinesq)

zgrid = map.y

# Deal with dimensional vs. nondimensional solvers
mtmp = mbm.BuildMatrices(nz, riist.rt_flag, prim_form=1)

# Get the baseflow
bsfl = mbf.RayleighTaylorBaseflow(nz, zgrid, map, bsfl_ref, mtmp)

#
# Read eigenfunction data
#

# Generate full file name
f1_full = os.path.join(file_dir, f1)
f2_full = os.path.join(file_dir, f2)

# Read data
z1, ef1 = module_pp.ReadEigenfunction(f1_full, nz)
z2, ef2 = module_pp.ReadEigenfunction(f2_full, nz)

#
# Generate x/y grids
#
xgrid, ygrid = module_pp.GenerateGridsXY(nx, ny, nwav, riist.alpha[0], riist.beta)

#
# Compute total disturbance field
#

print("alpha = ", riist.alpha[0])
print("beta = ", riist.beta)

dist_3d = module_pp.ComputeDistField_2waves(ef1, ef2, riist.alpha[0], riist.beta, xgrid, ygrid, omega_i)

#
# Write out tecplot file for 3d disturbance field of 2 wave superposition
#
FlagWrite = False

if FlagWrite:
    module_pp.Write3dDistFlow(xgrid, ygrid, zgrid, dist_3d, file_dir)

#
# Compute Reynolds stresses and plot them
#
UseCstVal = 0 # 0 -> does not use the multiplicative constant, 1 -> does use the multiplicative constant (from integration and summation of two waves)

module_pp.compute_reynolds_stresses_two_wave_case(zgrid, riist.alpha[0], riist.beta, ef1, omega_i, UseCstVal)

#
# Compute Reynolds Stresses Blance Equations
#

module_pp.reynolds_stress_balance_eqs_two_wave_case(zgrid, riist.alpha[0], riist.beta, ef1, omega_i, map, mtmp, bsfl, bsfl_ref, UseCstVal)

#
# Compare unclosed term from a-equation in terms of exact and models
#

module_pp.compare_exact_model_terms_a_equation_lst(zgrid, riist.alpha[0], riist.beta, ef1, omega_i, map, mtmp, bsfl, bsfl_ref)
