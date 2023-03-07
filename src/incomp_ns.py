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
import matplotlib
matplotlib.use('TkAgg') #----> Specify the backend
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
                  baseflowT, ny, Re, npts_alp, alpha, beta, mtmp, \
                  yinf, lmap, target1, alp_mich, ome_mich, npts_re, iarr, ire, bsfl_ref, riist):

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

    #################################################
    #  Create Chebyshev matrices and map if needed  #
    #################################################

    # Create instance for class GaussLobatto
    cheb = mgl.GaussLobatto(ny)

    # Get Gauss-Lobatto points
    cheb.cheby(ny-1)

    # Get spectral differentiation matrices on [1,-1]
    cheb.spectral_diff_matrices(ny-1)

    yi = cheb.xc

    # Create instance for class Mapping
    map = mma.Mapping(ny)

    # Create mapping and generate baseflow
    if (rt_flag):
        #k = np.sqrt(alpha[0]**2. + beta**2.)
        if (ire==0):
            print("")
            print("Rayleigh-Taylor Stability Analysis")
            print("==================================")
            print("")
            # In main.py I make yinf and lmap dimensional for solver boussinesq == -2
            map.map_shear_layer(yinf, yi, lmap, cheb.DM, bsfl_ref)

        bsfl = mbf.RayleighTaylorBaseflow(ny, map.y, map, bsfl_ref, mtmp)

    else:
        if ( baseflowT == 1 ):
            if (ire==0):
                print("")
                print("Mixing-Layer Stability Analysis")
                print("===============================")
                print("")
            map.map_shear_layer(yinf, yi, lmap, cheb.DM, bsfl_ref)
            bsfl = mbf.HypTan(ny, map.y)

            flag_reset = mod_util.check_lmap_val_reset_if_needed(bsfl.Up, map, lmap, ny)

            # If the mapping parameter has been reset, we need to re-map!
            if ( flag_reset ):
                map.map_shear_layer(yinf, yi, lmap, cheb.DM, bsfl_ref)
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

    if plot_grid_bsfl == 1:
        mod_util.plot_baseflow(ny, map.y, yi, bsfl.U, bsfl.Up, map.D1)


    # Check if baseflow is divergence free
    mod_util.check_baseflow_divergence(bsfl, bsfl_ref, mtmp, map)
        
    #################################################
    #            Solve stability problem            # 
    #################################################
        
    # Create instance for Class SolveGeneralizedEVP
    solve = msg.SolveGeneralizedEVP(ny, rt_flag, prim_form, mtmp.boussinesq)

    print("")
    print("Deletting instance mtmp")
    del mtmp

    
    # Build matrices and solve global/local problem
    omega_all,eigvals_filtered,mob,q_eigvect = solve.solve_stability_problem(map, alpha, beta, target1, Re, ny, Tracking, mid_idx, bsfl, bsfl_ref, Local, rt_flag, prim_form, baseflowT, iarr, ire, lmap)


    #################################################
    #           Plots, Energy balance, etc.         # 
    #################################################

    # Plot and write out eigenvalue spectrum (Global solver only)        
    if (not Local):
        if plot_eigvals == 1:
            mod_util.plot_eigvals(eigvals_filtered)
            
        mod_util.write_out_eigenvalues(solve.EigVal, ny)
        
        # Find index of target eigenvalue to extract eigenvector
        idx_tar1, found1 = mod_util.get_idx_of_closest_eigenvalue(solve.EigVal, abs_target1, target1)

        print("")
        print("Eigenvector will be extracted for eigenvalue omega = ", solve.EigVal[idx_tar1])
        q_eigvect = solve.EigVec[:,idx_tar1]
    else:
        idx_tar1 = -999

    #print("q_eigvect.shape = ", q_eigvect.shape)

    # Get eigenvectors
    norm_s, ueig, veig, weig, peig, reig = mod_util.get_normalize_eigvcts(ny, target1, idx_tar1, alpha, map, mob, bsfl, bsfl_ref, plot_eigvcts, rt_flag, q_eigvect, Local)

    if (not Local): solve.EigVec = solve.EigVec/norm_s

    #mod_util.plot_real_imag_part(ueig, "u", map.y, rt_flag, mob)
    #mod_util.plot_real_imag_part(veig, "v", map.y, rt_flag, mob)
    #mod_util.plot_real_imag_part(weig, "w", map.y, rt_flag, mob)
    #mod_util.plot_real_imag_part(-1j*veig, "-1j*v", map.y, rt_flag, mob)
    #mod_util.plot_real_imag_part(reig, "r", map.y, rt_flag, mob)
    #mod_util.plot_real_imag_part(peig, "p", map.y, rt_flag, mob)

    #####
    #idx_tar2, found2 = mod_util.get_idx_of_closest_eigenvalue(solve.EigVal, np.abs(0.0+1j*0.578232), 0.0+1j*0.578232)
    #idx_tar3, found3 = mod_util.get_idx_of_closest_eigenvalue(solve.EigVal, np.abs(0.0+1j*0.448152), 0.0+1j*0.448152)
    #idx_tar4, found4 = mod_util.get_idx_of_closest_eigenvalue(solve.EigVal, np.abs(0.0+1j*0.366181), 0.0+1j*0.366181)

    #mod_util.plot_real_imag_part(solve.EigVec[0:1*ny,idx_tar1], "u (mode 1)", map.y, rt_flag, mob)
    #mod_util.plot_real_imag_part(solve.EigVec[0:1*ny,idx_tar2], "u (mode 2)", map.y, rt_flag, mob)
    #mod_util.plot_real_imag_part(solve.EigVec[0:1*ny,idx_tar3], "u (mode 3)", map.y, rt_flag, mob)
    #mod_util.plot_real_imag_part(solve.EigVec[0:1*ny,idx_tar4], "u (mode 4)", map.y, rt_flag, mob)

    mod_util.write_eigvects_out_new(map.y, ueig, veig, weig, peig, reig, Local)
    
    input("Debug plot")
    
    # Plot eigenvectors if needed
    #mod_util.get_eigvcts(ny, solve.EigVec, target1, idx_tar1, alpha, map, mob, bsfl, bsfl_ref, plot_eigvcts, rt_flag, q_eigvect, Local)

    # Unwrap and shift phase
    Shift = 0
    ueig_ps, veig_ps, weig_ps, peig_ps, reig_ps = mod_util.unwrap_shift_phase(ny, ueig, veig, weig, peig, reig, Shift, map, mob, rt_flag)


    
    ptn = plt.gcf().number
    for i in range(1, ptn-1):
        plt.close(i)

    # Energy balance for Rayleigh-Taylor + Check that equations fullfilled by eigenfunctions
    if (rt_flag):
        if (Local):
            alpha_cur = alpha[-1]
            beta_cur  = beta
            omega_cur = omega_all[-1]
            #if   ( mob.boussinesq == -2 ): # non-dimensionalize since I have non-dimensionalized the eigenfunctions
                #alpha_cur = alpha[-1]*bsfl_ref.Lref
                #beta_cur  = beta*bsfl_ref.Lref
                #omega_cur = omega_all[-1]*bsfl_ref.Lref/bsfl_ref.Uref
        else:
            alpha_cur = alpha
            beta_cur  = beta
            omega_cur = omega_all[0]
                    
        mod_util.check_mass_continuity_satisfied_rayleigh_taylor(ueig_ps, veig_ps, weig_ps, peig_ps, reig_ps, \
        map.D1, map.D2, map.y, alpha_cur, beta_cur, np.imag(omega_cur), bsfl, bsfl_ref, mob, rt_flag)

        if   ( mob.boussinesq == 1 or mob.boussinesq == -3 or mob.boussinesq == -4 ):
            scale_fac = bsfl_ref.Lref
        else:
            scale_fac = 1.0
            
        print("")
        print("Range of dimensional wavenumbers and wavelengths:")
        print("-------------------------------------------------")
        alpha_min = min(alpha[0], alpha[-1])
        alpha_max = max(alpha[0], alpha[-1])
        print("alpha_min (dimensional)  = ", alpha_min/scale_fac)
        print("alpha_max (dimensional)  = ", alpha_max/scale_fac)
        lambda_min = 2*math.pi/alpha_max
        lambda_max = 2*math.pi/alpha_min
        print("lambda_min (dimensional) = ", lambda_min*scale_fac)
        print("lambda_max (dimensional) = ", lambda_max*scale_fac)

        
    # For R-T computations: Compute non-dimensional growth rate and wave-number using Chandrasekhar time and length scales
    if (rt_flag):

        print("")
        print("Non-dimensional growth rate and wavenumber using Chandrasekhar time ans length scales")
        print("=====================================================================================")
    
        if   ( mob.boussinesq == 1 or mob.boussinesq == -3 or mob.boussinesq == -4 ): # non-dimensional solvers

            if   ( mob.boussinesq == 1 ):  # Boussinesq solver
                print("Boussinesq R-T solver")

            elif ( mob.boussinesq == -3 ): # Sandoval solver
                print("Sandoval R-T solver")

            elif ( mob.boussinesq == -4 ): # Modified Sandoval solver
                print("Modified Sandoval R-T solver")
    
            print("Nondimensional growth rate (Chandrasekhar time scale)   n = ", iarr.omega_dim*bsfl_ref.Tscale_Chandra)
            # For these solvers the wavenumber is non-dimensional (non-dimensionalized by Lref)
            print("Nondimensional wave-number (Chandrasekhar length scale) k = ",riist.alpha/bsfl_ref.Lref*bsfl_ref.Lscale_Chandra)

            ueig_nd = ueig
            veig_nd = veig
            weig_nd = weig

            peig_nd = peig
            reig_nd = reig
            
            alpha_nd = alpha_cur
            beta_nd = beta_cur

            mod_util.compute_terms_rayleigh_taylor(ueig_nd,veig_nd,weig_nd,peig_nd,reig_nd,map,mob,bsfl,map.D1,map.D2,map.y,alpha_nd,beta_nd,Re,np.imag(iarr.omega_nondim),bsfl_ref,rt_flag,riist,Local)

            #mod_util.IntegrateDistDensity_CompareToPressure(peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, bsfl_ref)

            # Check energy balance for specific-volume-divergence correlation: 2*rho_bsfl*v'*du'_i/dx_i
            mod_util.compute_ener_bal_specific_volume_divergence_correlation(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, \
                                                                         alpha_nd, beta_nd, np.imag(iarr.omega_nondim), bsfl, bsfl_ref, mob, rt_flag)

            # Check if disturbance field is divergence free
            mod_util.check_disturbance_flow_divergence(ueig_nd, veig_nd, weig_nd, alpha_nd, beta_nd, map.D1)
        
            mod_util.build_total_flow_field(reig_nd, bsfl.Rho_nd, alpha_nd, beta_nd, iarr.omega_nondim, map.y)
            mod_util.write_2d_eigenfunctions(weig_nd, reig_nd, alpha_nd, beta_nd, iarr.omega_nondim, map.y, bsfl, mob)

            KE_dist,epsilon_dist,epsil_tild_min_2tild_dist = mod_util.compute_disturbance_dissipation(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.y, alpha_nd, beta_nd, map.D1, bsfl, bsfl_ref, mob)

            dudx = 0.0*ueig
            dudy = 0.0*ueig
            dudz = 0.0*ueig
            #
            dvdx = 0.0*ueig
            dvdy = 0.0*ueig
            dvdz = 0.0*ueig
            #
            dwdx = 0.0*ueig
            dwdy = 0.0*ueig
            dwdz = 0.0*ueig
            
            mod_util.compute_baseflow_dissipation(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, map.y, bsfl_ref, mob)

            # Sandoval equations balances
            if ( mob.boussinesq == -3 ): # Sandoval solver
                print("a-equation energy balance")
                print("=========================")
                print("")
                mod_util.compute_a_eq_ener_bal_sandoval(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, beta_nd, np.imag(iarr.omega_nondim), bsfl, bsfl_ref, mob, rt_flag)
                print("b-equation energy balance")
                print("=========================")
                mod_util.compute_b_eq_bal_sandoval(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, \
                                                   alpha_nd, beta_nd, np.imag(iarr.omega_nondim), bsfl, bsfl_ref, mob, rt_flag, map, KE_dist, epsilon_dist)
                print("density-self-correlation-equation energy balance")
                print("================================================")                
                mod_util.compute_dsc_eq_bal_sandoval(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, beta_nd, np.imag(iarr.omega_nondim), bsfl, bsfl_ref, mob, rt_flag, map)
            goThere = False
            if (goThere):
                mod_util.compute_a_equation_energy_balance(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, beta_nd, np.imag(iarr.omega_nondim), bsfl, bsfl_ref, mob, rt_flag)
                mod_util.compute_b_equation_energy_balance(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, beta_nd, np.imag(iarr.omega_nondim), bsfl, bsfl_ref, mob, rt_flag)

            # Boussinesq equations balances
            if   ( mob.boussinesq == 1 ):  # Boussinesq solver
                print("a-equation energy balance")
                print("=========================")
                mod_util.compute_a_eq_ener_bal_boussi(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, beta_nd, np.imag(iarr.omega_nondim), bsfl, bsfl_ref, mob, rt_flag)
                mod_util.compute_a_eq_ener_bal_boussi_Sc_1(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, \
                                                           alpha_nd, beta_nd, np.imag(iarr.omega_nondim), bsfl, bsfl_ref, mob, rt_flag, KE_dist, epsilon_dist, epsil_tild_min_2tild_dist)
                print("")
                print("b-equation energy balance")
                print("=========================")
                
                mod_util.compute_b_eq_ener_bal_boussi(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, beta_nd, np.imag(iarr.omega_nondim), bsfl, bsfl_ref, mob, rt_flag)
                print("")
                print("density-self-correlation-equation energy balance")
                print("================================================")                
                mod_util.compute_dsc_eq_bal_boussi(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y,alpha_nd, beta_nd, np.imag(iarr.omega_nondim), bsfl, bsfl_ref, mob, rt_flag, map)
                print("")
                
        elif ( mob.boussinesq == -2):                         # dimensional solver
            print("Chandrasekhar R-T solver")
            print("Nondimensional growth rate (Chandrasekhar time scale)   n = ", iarr.omega_dim*bsfl_ref.Tscale_Chandra)
            # Note: For solver -2, the wavenumber is made dimensional in main.py
            print("Nondimensional wave-number (Chandrasekhar length scale) k = ",riist.alpha*bsfl_ref.Lscale_Chandra)

            reig_nd = reig/bsfl_ref.rhoref
            weig_nd = weig/bsfl_ref.Uref
            alpha_nd = alpha_cur*bsfl_ref.Lref
            beta_nd = beta_cur*bsfl_ref.Lref
            mod_util.build_total_flow_field(reig_nd, bsfl.Rho_nd, alpha_nd, beta_nd, iarr.omega_nondim, map.y/bsfl_ref.Lref)
            mod_util.write_2d_eigenfunctions(weig_nd, reig_nd, alpha_nd, beta_nd, iarr.omega_nondim, map.y/bsfl_ref.Lref, bsfl, mob)


























        
    ### CRASHES THE COMPUTER plt.close('all')
    
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
    #mod_util.compute_growth_rate_from_energy_balance(ueig, veig, peig, bsfl.U, bsfl.Up, map.D1, map.y, alpha, Re)
    
    # input("Press any key to continue.........")







    

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









    # if (rt_flag == True):
    #     if (prim_form==1):
    #         mob = mbm.BuildMatrices(5*ny) # System matrices for RT are 5*ny by 5*ny (variables: u, v, w , p, rho)
    #     else:
    #         sys.exit("Just used for inviscid debugging 123456789")
    #         mob = mbm.BuildMatrices(ny) # Inviscid system matrix for RT based on w-equation only (see Chandrasekhar page 433)
    # else:
    #     mob = mbm.BuildMatrices(4*ny) # System matrices are 4*ny by 4*ny (variables: u, v, w , p)

    # if (rt_flag == True):
    #     if (prim_form==1):
    #         mob.set_matrices_rayleigh_taylor(ny, bsfl, map)
    #     else:
    #         mob.set_matrices_rayleigh_taylor_inviscid_w_equation(ny, bsfl, map)
    # else:
    #     mob.set_matrices(ny, Re, bsfl, map)

        
    # if (rt_flag == True):
    #     if (prim_form==1):
    #         solve = msg.SolveGeneralizedEVP(5*ny)
    #     else:
    #         solve = msg.SolveGeneralizedEVP(ny)
    # else:
    #     solve = msg.SolveGeneralizedEVP(4*ny)



    # Create instance for Class BuildMatrices
    #mob = mbm.BuildMatrices(ny, rt_flag, prim_form)

    # Build main stability matrices
    #mob.call_to_build_matrices(rt_flag, prim_form, ny, bsfl, Re, map)

   


    
