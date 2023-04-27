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
import class_readinput as rinput

import matplotlib
matplotlib.use('TkAgg') #----> Specify the backend
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

prim_form = 1

#################################################
#  Create Chebyshev matrices and map if needed  #
#################################################

# Get input file name
inFile  = sys.argv[1]

# Instance for class_readinput
riist   = rinput.ReadInput()

# Read input file
riist.read_input_file(inFile)
target1 = riist.target

ny = riist.ny

# Set reference quantities from non-dimensional parameters and Uref (ref. velocity) and gref (ref. acceleration)
bsfl_ref = mbf.Baseflow()
bsfl_ref.set_reference_quantities(riist)

# Deal with dimensional vs. nondimensional solvers
mtmp = mbm.BuildMatrices(ny, riist.rt_flag, prim_form)

# Create instance for class GaussLobatto
cheb = mgl.GaussLobatto(ny)

# Get Gauss-Lobatto points
cheb.cheby(ny-1)

# Get spectral differentiation matrices on [1,-1]
cheb.spectral_diff_matrices(ny-1)

yi = cheb.xc

# Create instance for class Mapping
map = mma.Mapping(ny)

print("")
print("Rayleigh-Taylor Stability Analysis")
print("==================================")
print("")
# In main.py I make yinf and lmap dimensional for solver boussinesq == -2
map.map_shear_layer(riist.yinf, yi, riist.lmap, cheb.DM, bsfl_ref, mtmp.boussinesq)
        
mod_util.calc_min_dy(map.y, mtmp.boussinesq, bsfl_ref)
        
bsfl = mbf.RayleighTaylorBaseflow(ny, map.y, map, bsfl_ref, mtmp)
        
#################################################
#            Solve stability problem            # 
#################################################

Re_range = np.linspace(bsfl_ref.Re_min, bsfl_ref.Re_max, bsfl_ref.npts_re)
print("Re_range =", Re_range)

rt_flag = riist.rt_flag
alpha = riist.alpha
beta = riist.beta
lmap = riist.lmap
Re = Re_range[0]
ire = 0

Local = False
Tracking = False
mid_idx = mod_util.get_mid_idx(ny)

baseflowT = riist.baseflowT

print("rt_flag = ", rt_flag)

# Create instance for main array omega
iarr = marray.MainArrays(bsfl_ref.npts_re, riist.npts_alp, riist.ny)

# Create instance for Class SolveGeneralizedEVP
solve = msg.SolveGeneralizedEVP(ny, rt_flag, prim_form, mtmp.boussinesq)

# Build matrices and solve global/local problem
omega_all,eigvals_filtered,mob,q_eigvect = solve.solve_stability_problem(map, alpha, beta, target1, Re, ny, Tracking, mid_idx, bsfl, bsfl_ref, Local, rt_flag, prim_form, baseflowT, iarr, ire, lmap)





