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

#################################
#     EXECUTABLE STATEMENTS     #
#################################

def incomp_ns_fct(prim_form, Local, plot_grid_bsfl, plot_eigvcts, plot_eigvals, rt_flag, SolverT, \
                  baseflowT, ny, Re, alp_min, alp_max, npts_alp, alpha, beta, yinf, lmap, target1, alp_mich, ome_mich, npts_re, iarr, ire, grav):

    if (rt_flag==False):
        prim_form = 1

    found1 = False
    found2 = False

    mid_idx = mod_util.get_mid_idx(ny)

    if (npts_alp > 1 or npts_re > 1):
        Tracking = True
    else:
        Tracking = False

    abs_target1 = np.abs(target1)

    # Create instance for class GaussLobatto
    cheb = mgl.GaussLobatto(ny)

    # Get Gauss-Lobatto points
    cheb.cheby(ny-1)

    # Get spectral differentiation matrices on [1,-1]
    cheb.spectral_diff_matrices(ny-1)

    yi = cheb.xc

    # Create instance for class Mapping
    map = mma.Mapping(ny)

    # Create instance for Class Baseflow
    #bsfl = mbf.Baseflow(ny)

    if (rt_flag):
        k = np.sqrt(alpha[0]**2. + beta**2.)
        if (ire==0):
            print("")
            print("Rayleigh-Taylor Stability Analysis")
            print("==================================")
            print("")
        map.map_shear_layer(yinf, yi, lmap, cheb.DM)
        bsfl = mbf.RayleighTaylorBaseflow(ny, map.y, map, Re, grav, k)

    else:
        if ( baseflowT == 1 ):
            if (ire==0):
                print("")
                print("Mixing-Layer Stability Analysis")
                print("===============================")
                print("")
            map.map_shear_layer(yinf, yi, lmap, cheb.DM)
            bsfl = mbf.HypTan(ny, map.y)

            flag_reset = mod_util.check_lmap_val_reset_if_needed(bsfl.Up, map, lmap, ny)

            # If the mapping parameter has been reset, we need to re-map!
            if ( flag_reset ):
                map.map_shear_layer(yinf, yi, lmap, cheb.DM)
                bsfl = mbf.HypTan(ny, map.y)
                
        elif ( baseflowT == 2 ):
            if (ire==0):
                print("")
                print("Plane Poiseuille Stability Analysis")
                print("===================================")
                print("")
            map.map_void(yi, cheb.DM)
            bsfl = mbf.PlanePoiseuille(ny, map.y)
        else:
            sys.exit("Not a proper value for flag baseflowT")

    print("Solving for Reynolds number Re = ", Re)
    print

    if plot_grid_bsfl == 1:
        mod_util.plot_baseflow(ny, map.y, yi, bsfl.U, bsfl.Up, map.D1)
    
    # Create instance for Class BuildMatrices
    if (rt_flag == True):
        if (prim_form==1):
            mob = mbm.BuildMatrices(5*ny) # System matrices for RT are 5*ny by 5*ny (variables: u, v, w , p, rho)
        else:
            mob = mbm.BuildMatrices(ny) # Inviscid system matrix for RT based on w-equation only (see Chandrasekhar page 433)
    else:
        mob = mbm.BuildMatrices(4*ny) # System matrices are 4*ny by 4*ny (variables: u, v, w , p)

    # Build main stability matrices
    if (rt_flag == True):
        if (prim_form==1):
            mob.set_matrices_rayleigh_taylor(ny, bsfl, map)
        else:
            mob.set_matrices_rayleigh_taylor_inviscid_w_equation(ny, bsfl, map)
    else:
        mob.set_matrices(ny, Re, bsfl, map)

    # Create instance for Class SolveGeneralizedEVP
    if (rt_flag == True):
        if (prim_form==1):
            solve = msg.SolveGeneralizedEVP(5*ny)
        else:
            solve = msg.SolveGeneralizedEVP(ny)
    else:
        solve = msg.SolveGeneralizedEVP(4*ny)


    omega_all, eigvals_filtered = solve.solve_stability_problem(mob, map, alpha, beta, target1, Re, ny, Tracking, mid_idx, bsfl, Local, rt_flag, prim_form, baseflowT, iarr, ire, lmap)

    #if (npts_alp > 1):
    #    mod_util.plot_imag_omega_vs_alpha(omega_all, "omega", alpha, alp_mich, ome_mich)

    # Plot and Write out eigenvalues
    if plot_eigvals == 1:
        mod_util.plot_eigvals(eigvals_filtered)

    if (not Local):
        mod_util.write_out_eigenvalues(solve.EigVal, ny)

    # Find index of target eigenvalue to extract eigenvector
    idx_tar1, found1 = mod_util.get_idx_of_closest_eigenvalue(solve.EigVal, abs_target1, target1)
    #idx_tar2, found2 = mod_util.get_idx_of_closest_eigenvalue(solve.EigVal, abs_target2, target2)

    if ( found1 == True and found2 == True ):
        found = True
        print("Both target eigenvalues have been found")
        print("")

    if (plot_eigvcts==1):
        
        # Get and Plot eigenvectors
        ueig, veig, weig, peig, reig = mod_util.get_plot_eigvcts(ny, solve.EigVec, target1, idx_tar1, alpha, map, bsfl, plot_eigvcts, rt_flag)
        
        phase_u = np.zeros(ny, dp)
        phase_v = np.zeros(ny, dp)
        phase_w = np.zeros(ny, dp)
        phase_p = np.zeros(ny, dp)
        phase_r = np.zeros(ny, dp)
        
        phase_u = np.arctan2(ueig.imag, ueig.real)
        phase_v = np.arctan2(veig.imag, veig.real)
        phase_w = np.arctan2(weig.imag, weig.real)
        phase_p = np.arctan2(peig.imag, peig.real)
        phase_r = np.arctan2(reig.imag, reig.real)
        
        amp_u   = np.abs(ueig)
        amp_v   = np.abs(veig)
        amp_w   = np.abs(weig)
        amp_p   = np.abs(peig)
        amp_r   = np.abs(reig)
        
        phase_u_uwrap = np.unwrap(phase_u)
        phase_v_uwrap = np.unwrap(phase_v)
        phase_w_uwrap = np.unwrap(phase_w)
        phase_p_uwrap = np.unwrap(phase_p)
        phase_r_uwrap = np.unwrap(phase_r)
        
        # When I take phase_ref as phase_v_uwrap ==> I get u and v symmetric/anti-symmetric
        # When I take phase_ref as phase_u_uwrap ==> v is not symmetric/anti-symmetric
        phase_ref     = phase_v_uwrap[mid_idx]
        
        phase_u_uwrap = phase_u_uwrap - phase_ref
        phase_v_uwrap = phase_v_uwrap - phase_ref
        phase_w_uwrap = phase_w_uwrap - phase_ref
        phase_p_uwrap = phase_p_uwrap - phase_ref
        phase_r_uwrap = phase_r_uwrap - phase_ref
        
        # Plot some results
        mod_util.plot_phase(phase_u, "phase_u", map.y)
        
        #print("exp(1j*phase) = ", np.exp(1j*phase_u))
        #print("exp(1j*phase_unwrapped) = ", np.exp(1j*phase_u_uwrap))
        
        print("np.max( np.abs( np.exp(1j*phase_u) - np.exp(1j*phase_u_uwrap ))) = ", np.max(np.abs( np.exp(1j*phase_u) - np.exp(1j*phase_u_uwrap ))))
        print("np.max( np.abs( np.exp(1j*phase_u)) - np.abs( np.exp(1j*phase_u_uwrap )) ) = ", np.max( np.abs( np.exp(1j*phase_u) ) - np.abs( np.exp(1j*phase_u_uwrap ) ) ) )
        
        Shift = 1
        
        if Shift == 1:
            ueig_ps = amp_u*np.exp(1j*phase_u_uwrap)
            veig_ps = amp_v*np.exp(1j*phase_v_uwrap)
            weig_ps = amp_w*np.exp(1j*phase_w_uwrap)
            peig_ps = amp_p*np.exp(1j*phase_p_uwrap)
            reig_ps = amp_r*np.exp(1j*phase_r_uwrap)
        else:
            ueig_ps = ueig
            veig_ps = veig
            weig_ps = weig
            peig_ps = peig
            reig_ps = reig
        
        print("")
        print("np.max(np.real(ueig_ps)) = ", np.max(np.real(ueig_ps)))
        print("np.max(np.abs(ueig_ps)) = ", np.max(np.abs(ueig_ps)))
        print("")
        print("")
        print("np.max(np.real(weig_ps)) = ", np.max(np.real(weig_ps)))
        print("np.max(np.abs(weig_ps)) = ", np.max(np.abs(weig_ps)))
        print("")
        print("np.max(np.real(peig_ps)) = ", np.max(np.real(peig_ps)))
        print("np.max(np.abs(peig_ps)) = ", np.max(np.abs(peig_ps)))
        print("")
        print("np.max(np.real(reig_ps)) = ", np.max(np.real(reig_ps)))
        print("np.max(np.abs(reig_ps)) = ", np.max(np.abs(reig_ps)))
        print("")
        
        amp_u_ps = np.abs(ueig_ps)
        amp_v_ps = np.abs(veig_ps)
        amp_w_ps = np.abs(weig_ps)
        amp_p_ps = np.abs(peig_ps)
        amp_r_ps = np.abs(reig_ps)
        
        ueig_from_continuity = -np.matmul(map.D1, veig_ps)/(1j*alpha)
        
        mod_util.plot_real_imag_part(ueig_ps, "u", map.y)
        #mod_util.plot_real_imag_part(veig_ps, "v", map.y)
        mod_util.plot_real_imag_part(weig_ps, "w", map.y)
        mod_util.plot_real_imag_part(peig_ps, "p", map.y)
        mod_util.plot_real_imag_part(reig_ps, "r", map.y)
        
        input("Check real imag parts")
        
        mod_util.plot_five_vars_amplitude(ueig_ps, veig_ps, weig_ps, peig_ps, reig_ps, "u", "v", "w", "p", "r", map.y)

        input("Waiting after plotting eigenvectors")
        
    ### CRASHES THE COMPUTER plt.close('all')

    # # Verify that continuity equation is satisfied
    # dupdx   = 1j*alpha*ueig_ps
    # dvpdy   = np.matmul(map.D1, veig_ps)

    # print("Continuity check:", np.max(np.abs(dupdx+dvpdy)))

    # # Compute stream function check
    # Phi_eig_cc = mod_util.compute_stream_function_check(ueig_ps, veig_ps, alpha[0], map.D1, map.y, ny)

    # # Read eigenfunction data from Thomas (1953): "The stability of plane Poiseuille flow"
    # mod_util.read_thomas_data(Phi_eig_cc, map.y, map.D1)
    
    # # Write eigenvectors from global analysis out
    # q_glob = np.zeros(4*ny, dpc)
    
    # q_glob[0:ny]      = ueig_ps
    # q_glob[ny:2*ny]   = veig_ps
    # q_glob[2*ny:3*ny] = weig_ps
    # q_glob[3*ny:4*ny] = peig_ps
    
    # mod_util.write_eigvects_out(q_glob, map.y, -666, ny)
    
    # # Energy balance computations
    # #mod_util.compute_growth_rate_from_energy_balance(ueig, veig, peig, bsfl.U, bsfl.Up, map.D1, map.y, alpha, Re)
    
    # input("Press any key to continue.........")







    
#for i in range(1, 10):
#    plt.close(i)

# # Number of discretization points in wall-normal direction
# ny = 201

# # Reynolds number
# Re = 10000 #1e50

# # Longitudinal and spanwise wavenumbers: alpha and beta
# alp_min = 1.0 #0.4446 #0.0001
# alp_max = 1.0 #0.4446 #0.9999
# npts_alp = 51
    
# alpha = np.linspace(alp_min, alp_max, npts_alp)
# beta  = 0.

# # Mapping parameters: yinf defines the domain extent in y and lmap the point clustering
# yinf = 1.0 #80
# lmap = 2 # cannot be = 0; close to zero: extreme stretching

# # Target eigenvalue to find corresponding eigenvector
# target1 = 2.00000000000e-01 +1j*9.40901411190e-02
# target1 = 5.00000000089e-04 +1j*4.91847937625e-04 # for alpha = 0.001
# target1 = 2.22300000000e-01 +1j*9.48510128415e-02 # for 0.4446

# target1 = 2.37526421071e-01 +1j*3.73971254580e-03 # for plane Poiseuille flow

    # for i in range(0, ny):
    #     phase_u[i] = np.arctan2(ueig[i].imag, ueig[i].real)
    #     phase_v[i] = np.arctan2(veig[i].imag, veig[i].real)
    #     phase_w[i] = np.arctan2(weig[i].imag, weig[i].real)
    #     phase_p[i] = np.arctan2(peig[i].imag, peig[i].real)
    #     phase_r[i] = np.arctan2(reig[i].imag, reig[i].real)
    
#print("np.max(np.real(veig_ps)) = ", np.max(np.real(veig_ps)))
#print("np.max(np.abs(veig_ps)) = ", np.max(np.abs(veig_ps)))

#print("np.abs(ueig*veig)=",np.abs(ueig*veig))
#print("np.abs(ueig_ps*veig_ps)=",np.abs(ueig_ps*veig_ps))

#mod_util.plot_amplitude(ueig_ps, "u", map.y)
#mod_util.plot_amplitude(veig_ps, "v", map.y)
#mod_util.plot_amplitude(peig_ps, "p", map.y)
#mod_util.plot_two_vars_amplitude(peig_ps, ueig_ps, "p", "u", map.y)

#mod_util.plot_four_vars_amplitude(ueig_ps, veig_ps, weig_ps, peig_ps, "u", "v", "w", "p", map.y)
