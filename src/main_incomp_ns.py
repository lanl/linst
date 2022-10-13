import sys
import math
import numpy as np
#from scipy import linalg
import class_gauss_lobatto as mgl
import class_baseflow as mbf
import class_build_matrices as mbm
import class_solve_gevp as msg
import class_mapping as mma
import matplotlib.pyplot as plt

import module_utilities as mod_util

dp  = np.dtype('d')  # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

str_vars = np.array(['u-velocity', 'v-velocity', 'w-velocity', 'pressure'])

###################################
#    REQUIRED INPUT PARAMETERS    #
###################################

# Number of discretization points
ny = 200

# Reynolds number
Re = 1e50

# Longitudinal and spanwise wavenumbers: alpha and beta
alpha = 0.4
beta  = 0.

# Target eigenvalue to find corresponding eigenvector
target1 = 2.00000000000e-01 +1j*9.40901411190e-02

# Plotting flags
plot_grid_bsfl = 0 # set to 1 to plot grid distribution and baseflow profiles
plot_eigvals = 1 # set to 1 to plot eigenvalues
plot_eigvcts = 1 # set to 1 to plot eigenvectors

#################################
#     EXECUTABLE STATEMENTS     #
#################################

abs_target1 = np.abs(target1)

print("Eigenfunction will be extracted for mode with eigenvalue omega = ", target1)
print("abs_target=",abs_target1)

target2 = np.conj(target1)
abs_target2 = np.abs(target2)

#if (ny % 2) == 0:
    #sys.exit("ny needs to be odd")

# Create instance for class GaussLobatto
cheb = mgl.GaussLobatto(ny)

# Get Gauss-Lobatto points
cheb.cheby(ny-1)

# Get spectral differentiation matrices on [1,-1]
cheb.spectral_diff_matrices(ny-1)

yi = cheb.xc

# Create instance for class Mapping
map = mma.Mapping(ny)

yinf = 80.
lmap = 2 # cannot be = 0; close to zero: extreme stretching
map.map_shear_layer(yinf, yi, lmap, cheb.DM)

# Create instance for Class Baseflow
#bsfl = mbf.Baseflow(ny)
bsfl = mbf.HypTan(ny, map.y)

if plot_grid_bsfl == 1:
    mod_util.plot_cheb_baseflow(ny, map.y, yi, bsfl.U, bsfl.Up)
    
# Create instance for Class BuildMatrices
mob = mbm.BuildMatrices(4*ny) # System matrices are 4*ny by 4*ny

# Build main stability matrices
mob.set_matrices(ny, Re, alpha, beta, bsfl, map)

# Set boundary conditions
mob.set_bc_shear_layer(mob.mat_lhs, mob.mat_rhs, ny, map)

# Create instance for Class SolveGeneralizedEVP
solve = msg.SolveGeneralizedEVP(4*ny*4*ny) # System matrices are 4*ny by 4*ny
eigvals, eigvcts = solve.solve_eigenvalue_problem(mob.mat_lhs, mob.mat_rhs)

# Plot and Write out eigenvalues
if plot_eigvals == 1:
    mod_util.plot_eigvals(eigvals)
    
mod_util.write_out_eigenvalues(eigvals, ny)

# Find index of target eigenvalue to extract eigenvector
idx_tar1, found1 = mod_util.get_idx_of_closest_eigenvalue(eigvals, abs_target1, target1)
idx_tar2, found2 = mod_util.get_idx_of_closest_eigenvalue(eigvals, abs_target2, target2)

if ( found1 == True and found2 == True ):
    found = True
    print("Both target eigenvalues have been found")

# Plot and Write out eigenvalues
if plot_eigvcts == 1:
    mod_util.plot_eigvcts(ny, eigvcts, target1, idx_tar1, idx_tar2, alpha, map, bsfl)

