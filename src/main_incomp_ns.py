import sys
import math
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

dp  = np.dtype('d')  # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

str_vars = np.array(['u-velocity', 'v-velocity', 'w-velocity', 'pressure'])

###################################
#    REQUIRED INPUT PARAMETERS    #
###################################

# Number of discretization points
ny = 201

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

yinf = 80
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
solve = msg.SolveGeneralizedEVP(4*ny) # System matrices are 4*ny by 4*ny
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

# Plot eigenvectors
if plot_eigvcts == 1:
    ueig, veig, peig = mod_util.plot_eigvcts(ny, eigvcts, target1, idx_tar1, idx_tar2, alpha, map, bsfl)

phase_u = np.arctan2(ueig.imag, ueig.real)
phase_v = np.arctan2(veig.imag, veig.real)
phase_p = np.arctan2(peig.imag, peig.real)

amp_u   = np.abs(ueig)
amp_v   = np.abs(veig)
amp_p   = np.abs(peig)

phase_u_uwrap = np.unwrap(phase_u)
phase_v_uwrap = np.unwrap(phase_v)
phase_p_uwrap = np.unwrap(phase_p)

if (ny % 2) != 0:
    mid_idx = int( (ny-1)/2 )
else:
    warnings.warn("When ny is even, there is no mid-index")
    mid_idx = int ( ny/2 )
    
print("mid_idx = ", mid_idx)

# When I take phase_ref as phase_v_uwrap ==> I get u and v symmetric/anti-symmetric
# When I take phase_ref as phase_u_uwrap ==> v is not symmetric/anti-symmetric
phase_ref     = phase_u_uwrap[mid_idx]

phase_u_uwrap = phase_u_uwrap - phase_ref
phase_v_uwrap = phase_v_uwrap - phase_ref
phase_p_uwrap = phase_p_uwrap - phase_ref

#print("exp(1j*phase) = ", np.exp(1j*phase_u))
#print("exp(1j*phase_unwrapped) = ", np.exp(1j*phase_u_uwrap))

print("np.max(np.abs( np.exp(1j*phase_u) - np.exp(1j*phase_u_uwrap ))) = ", np.max(np.abs( np.exp(1j*phase_u) - np.exp(1j*phase_u_uwrap ))))

# print("phase_u = ", phase_u)
# print("")
# print("phase_v = ", phase_v)
# print("")
# print("phase_p = ", phase_p)

ueig_ps = amp_u*np.exp(1j*phase_u_uwrap)
veig_ps = amp_v*np.exp(1j*phase_v_uwrap)
peig_ps = amp_p*np.exp(1j*phase_p_uwrap)

amp_u_ps = np.abs(ueig_ps)
amp_v_ps = np.abs(veig_ps)
amp_p_ps = np.abs(peig_ps)

ueig_from_continuity = -np.matmul(map.D1, veig_ps)/(1j*alpha)

#print("np.abs(ueig*veig)=",np.abs(ueig*veig))
#print("np.abs(ueig_ps*veig_ps)=",np.abs(ueig_ps*veig_ps))

fa = plt.figure(1001)
plt.plot(phase_u_uwrap, map.y, 'ks', label="Phase")
plt.xlabel('Phase')
plt.ylabel('y')
plt.title('Phase')
plt.legend(loc="upper left")
fa.show()

fa = plt.figure(1002)
plt.plot(ueig_ps.real, map.y, 'k', label="real(u)")
plt.plot(ueig_from_continuity.real, map.y, 'g--', label="real(u) from continuity")
plt.xlabel('real(u)')
plt.ylabel('y')
#plt.title('Phase')
plt.legend(loc="upper right")
fa.show()

fa = plt.figure(1003)
plt.plot(ueig_ps.imag, map.y, 'k', label="imag(u)")
plt.plot(ueig_from_continuity.imag, map.y, 'g--', label="imag(u) from continuity")
plt.xlabel('imag(u)')
plt.ylabel('y')
#plt.title('Phase')
plt.legend(loc="upper right")
fa.show()

# fa = plt.figure(1004)
# plt.plot(amp_u_ps, map.y, 'k', label="amp(u) (phase shifted)")
# plt.plot(amp_u, map.y, 'g--', label="amp(u)")
# plt.xlabel('amp(u)')
# plt.ylabel('y')
# #plt.title('Phase')
# plt.legend(loc="upper right")
# fa.show()

fa = plt.figure(1005)
plt.plot(veig_ps.real, map.y, 'k', label="real(v)")
plt.xlabel('real(v)')
plt.ylabel('y')
#plt.title('Phase')
plt.legend(loc="upper right")
fa.show()

fa = plt.figure(1006)
plt.plot(veig_ps.imag, map.y, 'k', label="imag(v)")
plt.xlabel('imag(v)')
plt.ylabel('y')
#plt.title('Phase')
plt.legend(loc="upper right")
fa.show()



input("Press any key to continue.........")
