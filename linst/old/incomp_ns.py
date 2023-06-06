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



    # #####
    # #idx_tar2, found2 = mod_util.get_idx_of_closest_eigenvalue(solve.EigVal, np.abs(0.0+1j*0.578232), 0.0+1j*0.578232)
    # #idx_tar3, found3 = mod_util.get_idx_of_closest_eigenvalue(solve.EigVal, np.abs(0.0+1j*0.448152), 0.0+1j*0.448152)
    # #idx_tar4, found4 = mod_util.get_idx_of_closest_eigenvalue(solve.EigVal, np.abs(0.0+1j*0.366181), 0.0+1j*0.366181)

    # #mod_util.plot_real_imag_part(solve.EigVec[0:1*ny,idx_tar1], "u (mode 1)", map.y, rt_flag, mob)
    # #mod_util.plot_real_imag_part(solve.EigVec[0:1*ny,idx_tar2], "u (mode 2)", map.y, rt_flag, mob)
    # #mod_util.plot_real_imag_part(solve.EigVec[0:1*ny,idx_tar3], "u (mode 3)", map.y, rt_flag, mob)
    # #mod_util.plot_real_imag_part(solve.EigVec[0:1*ny,idx_tar4], "u (mode 4)", map.y, rt_flag, mob)

    # mod_util.write_eigvects_out_new(map.y, ueig, veig, weig, peig, reig, Local, bsfl_ref)

    # input("Pausing for eigenfunction output")

    # if   ( mob.boussinesq == -2 or mob.boussinesq == 10 ):
    #     sys.exit("Dimensional solver =====> check this!")
    
    # # Plot eigenvectors if needed
    # #mod_util.get_eigvcts(ny, solve.EigVec, target1, idx_tar1, alpha, map, mob, bsfl, bsfl_ref, plot_eigvcts, rt_flag, q_eigvect, Local)

    # # Unwrap and shift phase
    # Shift = 0
    # ueig_ps, veig_ps, weig_ps, peig_ps, reig_ps = mod_util.unwrap_shift_phase(ny, ueig, veig, weig, peig, reig, Shift, map, mob, rt_flag)

    # # Close previous plots
    # mod_util.close_previous_plots()

    # # Energy balance for Rayleigh-Taylor + Check that equations fullfilled by eigenfunctions
    # if (rt_flag):
    #     if (Local):
    #         alpha_cur = alpha[-1]
    #         beta_cur  = beta
    #         omega_cur = omega_all[-1]
    #         #if   ( mob.boussinesq == -2 ): # non-dimensionalize since I have non-dimensionalized the eigenfunctions
    #             #alpha_cur = alpha[-1]*bsfl_ref.Lref
    #             #beta_cur  = beta*bsfl_ref.Lref
    #             #omega_cur = omega_all[-1]*bsfl_ref.Lref/bsfl_ref.Uref
    #     else:
    #         alpha_cur = alpha
    #         beta_cur  = beta
    #         omega_cur = omega_all[0]
                    
    #     mod_util.check_mass_continuity_satisfied_rayleigh_taylor(ueig_ps, veig_ps, weig_ps, peig_ps, reig_ps, \
    #     map.D1, map.D2, map.y, alpha_cur, beta_cur, np.imag(omega_cur), bsfl, bsfl_ref, mob, rt_flag)

    #     if   ( mob.boussinesq == 1 or mob.boussinesq == -3 or mob.boussinesq == -4 ):
    #         scale_fac = bsfl_ref.Lref
    #     else:
    #         scale_fac = 1.0
            
    #     print("")
    #     print("Range of dimensional wavenumbers and wavelengths:")
    #     print("-------------------------------------------------")
    #     alpha_min = min(alpha[0], alpha[-1])
    #     alpha_max = max(alpha[0], alpha[-1])
    #     print("alpha_min (dimensional)  = ", alpha_min/scale_fac)
    #     print("alpha_max (dimensional)  = ", alpha_max/scale_fac)
    #     lambda_min = 2*math.pi/alpha_max
    #     lambda_max = 2*math.pi/alpha_min
    #     print("lambda_min (dimensional) = ", lambda_min*scale_fac)
    #     print("lambda_max (dimensional) = ", lambda_max*scale_fac)

        
    # # For R-T computations: Compute non-dimensional growth rate and wave-number using Chandrasekhar time and length scales
    # if (rt_flag):

    #     print("")
    #     print("Non-dimensional growth rate and wavenumber using Chandrasekhar time ans length scales")
    #     print("=====================================================================================")
    
    #     if   ( mob.boussinesq == 1 or mob.boussinesq == -3 or mob.boussinesq == -4 ): # non-dimensional solvers

    #         if   ( mob.boussinesq == 1 ):  # Boussinesq solver
    #             print("Boussinesq R-T solver")

    #         elif ( mob.boussinesq == -3 ): # Sandoval solver
    #             print("Sandoval R-T solver")

    #         elif ( mob.boussinesq == -4 ): # Modified Sandoval solver
    #             print("Modified Sandoval R-T solver")
    
    #         print("Nondimensional growth rate (Chandrasekhar time scale)   n = ", omega_sol_dim*bsfl_ref.Tscale_Chandra)
    #         # For these solvers the wavenumber is non-dimensional (non-dimensionalized by Lref)
    #         print("Nondimensional wave-number (Chandrasekhar length scale) k = ",alpha/bsfl_ref.Lref*bsfl_ref.Lscale_Chandra)

    #         ueig_nd = ueig
    #         veig_nd = veig
    #         weig_nd = weig

    #         peig_nd = peig
    #         reig_nd = reig
            
    #         alpha_nd = alpha_cur
    #         beta_nd = beta_cur

    #         #mod_util.compute_terms_rayleigh_taylor(ueig_nd,veig_nd,weig_nd,peig_nd,reig_nd,map,mob,bsfl,map.D1,map.D2,map.y,alpha_nd,beta_nd,Re,np.imag(omega_sol_nondim),bsfl_ref,rt_flag,riist,Local)

    #         # Close previous plots
    #         mod_util.close_previous_plots()
            
    #         #mod_util.comp_ke_bal_rt_boussinesq(ueig_nd,veig_nd,weig_nd,peig_nd,reig_nd,map,mob,bsfl,map.D1,map.D2,map.y,alpha_nd,beta_nd,Re,np.imag(omega_sol_nondim),bsfl_ref,rt_flag,riist,Local)
            
    #         #mod_util.IntegrateDistDensity_CompareToPressure(peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, bsfl_ref)

    #         # Check energy balance for specific-volume-divergence correlation: 2*rho_bsfl*v'*du'_i/dx_i
    #         mod_util.compute_ener_bal_specific_volume_divergence_correlation(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, \
    #                                                                      alpha_nd, beta_nd, np.imag(omega_sol_nondim), bsfl, bsfl_ref, mob, rt_flag)

            
    #         # Check if disturbance field is divergence free
    #         mod_util.check_baseflow_disturbance_flow_divergence_sandoval(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, alpha_nd, beta_nd, bsfl, bsfl_ref, map, mob)

        
    #         #mod_util.build_total_flow_field(reig_nd, bsfl.Rho_nd, alpha_nd, beta_nd, omega_sol_nondim, map.y)
    #         mod_util.write_2d_eigenfunctions(weig_nd, reig_nd, alpha_nd, beta_nd, omega_sol_nondim, map.y, bsfl, mob)

    #         ke_dim, eps_dim = mod_util.compute_disturbance_dissipation_dimensional(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.y, alpha_nd, beta_nd, map.D1, bsfl, bsfl_ref, mob, iarr)

    #         # Close previous plots
    #         mod_util.close_previous_plots()
            
    #         # Compute and plot Reynolds stresses 
    #         mod_util.compute_reynolds_stresses(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.y, bsfl_ref, rt_flag, map.D1)
            
    #         KE_dist,epsilon_dist,epsil_tild_min_2tild_dist = mod_util.compute_disturbance_dissipation(ueig_nd,veig_nd,weig_nd,peig_nd,reig_nd,map.y,alpha_nd,beta_nd,map.D1,map.D2,bsfl,bsfl_ref,mob)

    #         # Cmpute Taylor and integral length scales
    #         mod_util.compute_taylor_and_integral_scales(ke_dim, eps_dim, KE_dist, epsilon_dist, map.y, iarr, bsfl_ref)

    #         # Reynolds stress balance equations ==> THIS NEEDS TO BE AFTER CALL TO compute_taylor_and_integral_scales
    #         mod_util.reynolds_stresses_balance_equations(ueig_nd,veig_nd,weig_nd,peig_nd,reig_nd,map.y,mob,bsfl,bsfl_ref,rt_flag,map.D1,map.D2,alpha_nd,beta_nd,np.imag(omega_sol_nondim),iarr)
            
    #         input('Check Reynolds stresses equations balances...')

    #         dudx = 0.0*ueig
    #         dudy = 0.0*ueig
    #         dudz = 0.0*ueig
    #         #
    #         dvdx = 0.0*ueig
    #         dvdy = 0.0*ueig
    #         dvdz = 0.0*ueig
    #         #
    #         dwdx = 0.0*ueig
    #         dwdy = 0.0*ueig
    #         dwdz = 0.0*ueig
            
    #         mod_util.compute_baseflow_dissipation(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, map.y, bsfl_ref, mob)

    #         # Sandoval equations balances
    #         if ( mob.boussinesq == -3 ): # Sandoval solver
    #             print("a-equation energy balance")
    #             print("=========================")
    #             print("")
    #             mod_util.compute_a_eq_ener_bal_sandoval(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, beta_nd, np.imag(omega_sol_nondim), bsfl, bsfl_ref, mob, rt_flag)
    #             print("b-equation energy balance")
    #             print("=========================")
    #             mod_util.compute_b_eq_bal_sandoval(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, \
    #                                                alpha_nd, beta_nd, np.imag(omega_sol_nondim), bsfl, bsfl_ref, mob, rt_flag, map, KE_dist, epsilon_dist)
    #             print("density-self-correlation-equation energy balance")
    #             print("================================================")                
    #             mod_util.compute_dsc_eq_bal_sandoval(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, beta_nd, np.imag(omega_sol_nondim), bsfl, bsfl_ref, mob, rt_flag, map)
    #         goThere = False
    #         if (goThere):
    #             mod_util.compute_a_equation_energy_balance(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, beta_nd, np.imag(omega_sol_nondim), bsfl, bsfl_ref, mob, rt_flag)
    #             mod_util.compute_b_equation_energy_balance(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, beta_nd, np.imag(omega_sol_nondim), bsfl, bsfl_ref, mob, rt_flag)

    #         # Boussinesq equations balances
    #         if   ( mob.boussinesq == 1 ):  # Boussinesq solver
    #             print("a-equation energy balance")
    #             print("=========================")
    #             #mod_util.compute_a_eq_ener_bal_boussi(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, beta_nd, np.imag(omega_sol_nondim), bsfl, bsfl_ref, mob, rt_flag)
    #             #mod_util.compute_a_eq_ener_bal_boussi_Sc_1(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, \
    #             #                                           alpha_nd, beta_nd, np.imag(omega_sol_nondim), bsfl, bsfl_ref, mob, rt_flag, KE_dist, epsilon_dist, epsil_tild_min_2tild_dist, iarr)

                
    #             mod_util.compute_a_eq_ener_bal_boussi_Sc_1_NEW(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, \
    #                                                        alpha_nd, beta_nd, np.imag(omega_sol_nondim), bsfl, bsfl_ref, mob, rt_flag, KE_dist, epsilon_dist, epsil_tild_min_2tild_dist, iarr)

    #             print("")
    #             print("b-equation energy balance")
    #             print("=========================")
                
    #             #mod_util.compute_b_eq_ener_bal_boussi(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y, alpha_nd, beta_nd, np.imag(omega_sol_nondim), bsfl, bsfl_ref, mob, rt_flag)
    #             print("")
    #             print("density-self-correlation-equation energy balance")
    #             print("================================================")                
    #             #mod_util.compute_dsc_eq_bal_boussi(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, map.D1, map.D2, map.y,alpha_nd, beta_nd, np.imag(omega_sol_nondim), bsfl, bsfl_ref, mob, rt_flag, map)
    #             print("")
                
    #     elif ( mob.boussinesq == -2):                         # dimensional solver
    #         print("Chandrasekhar R-T solver")
    #         print("Nondimensional growth rate (Chandrasekhar time scale)   n = ", omega_sol_dim*bsfl_ref.Tscale_Chandra)
    #         # Note: For solver -2, the wavenumber is made dimensional in main.py
    #         print("Nondimensional wave-number (Chandrasekhar length scale) k = ",alpha*bsfl_ref.Lscale_Chandra)

    #         reig_nd = reig/bsfl_ref.rhoref
    #         weig_nd = weig/bsfl_ref.Uref
    #         alpha_nd = alpha_cur*bsfl_ref.Lref
    #         beta_nd = beta_cur*bsfl_ref.Lref
    #         #mod_util.build_total_flow_field(reig_nd, bsfl.Rho_nd, alpha_nd, beta_nd, omega_sol_nondim, map.y/bsfl_ref.Lref)
    #         mod_util.write_2d_eigenfunctions(weig_nd, reig_nd, alpha_nd, beta_nd, omega_sol_nondim, map.y/bsfl_ref.Lref, bsfl, mob)
