import sys
import warnings
import numpy as np
from scipy import linalg
import module_utilities as mod_util
import class_build_matrices as mbm
#import scipy as sp

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

class SolveGeneralizedEVP:
    """
    This class define a solution object and contain the main function that provides the solution 
    to the generalized eigenvalue problem (GEVP)
    """
    def __init__(self, ny, rt_flag, prim_form):
        """
        Constructor of class SolveGeneralizedEVP
        """
        if (rt_flag == True):
            if (prim_form==1):
                size = 5*ny
            else:
                size = ny
        else:
            size = 4*ny

        self.size = size
        self.EigVal = np.zeros(size*size, dpc)
        self.EigVec = np.zeros((size, size), dpc)

    def solve_stability_problem(self, map, alpha, beta, omega, Re, ny, Tracking, mid_idx, bsfl, bsfl_ref, Local, rt_flag, prim_form, baseflowT, iarr, ire, lmap):
        """
        Function...
        """        
        npts = len(alpha)

        omega_all = np.zeros(npts, dpc)
        q_eigvect = np.zeros(self.size, dpc)
        
        
        for i in range(0, npts):

            print("")
            print("Solving for wavenumber alpha = %10.5e (%i/%i)" % (alpha[i], i+1, npts) )
            print("------------------------------------------------")
            print("")

            if Tracking and Local:
                # Extrapolate omega
                if (i > 1):
                    pass
                    # Extrapolate in alpha space
                    omega = mod_util.extrapolate_in_alpha(iarr, alpha, i, ire)
                elif (ire > 1):
                    pass
                    # Extrapolate in Reynolds space
                    omega = mod_util.extrapolate_in_reynolds(iarr, i, ire)

                # Create instance for Class BuildMatrices
                mob = mbm.BuildMatrices(ny, rt_flag, prim_form)
                
                omega, q_eigvect = self.solve_stability_secant(ny, mid_idx, omega, omega + omega*1e-5, alpha[i], beta, \
                                                                  Re, map, mob, Tracking, bsfl, bsfl_ref, Local, baseflowT, rt_flag, prim_form)

                iarr.omega_array[ire, i] = omega

                # Only tracking a single mode here!!!!!
                eigvals_filtered = 0.0

                #mod_util.write_eigvects_out(q_eigenvects, map.y, i, ny)
            else:

                # Create instance for Class BuildMatrices
                mob = mbm.BuildMatrices(ny, rt_flag, prim_form)
                
                # Build main stability matrices
                mob.call_to_build_matrices(rt_flag, prim_form, ny, bsfl, bsfl_ref, Re, map)

                # Assemble matrices
                if (prim_form==1):
                    mob.assemble_mat_lhs(alpha[i], beta, omega, Tracking , Local, bsfl_ref)
                else:
                    sys.exit("Just used for inviscid debugging")
                    mob.assemble_mat_lhs_rt_inviscid(alpha[i], beta, omega, Tracking , Local)

                mob.call_to_set_bc(rt_flag, prim_form, ny, map)

                #mob.mat_lhs = np.conj(mob.mat_lhs)
                #mob.mat_rhs = np.conj(mob.mat_rhs)
                
                self.solve_eigenvalue_problem(mob.mat_lhs, mob.mat_rhs)

                eigvals_filtered = self.EigVal[ np.abs(self.EigVal) < 10000. ]
                q_eigenvects     = self.EigVec

                if (rt_flag==True and prim_form==0):
                    neigs = len(eigvals_filtered)
                    eigvals_filtered_tmp = eigvals_filtered
                    eigvals_filtered = np.zeros(2*neigs, dpc)
                    eigvals_filtered[0:neigs] = 1j/np.sqrt(eigvals_filtered_tmp) # I set it to the complex part because growth rate
                    eigvals_filtered[neigs:2*neigs] = -1j/np.sqrt(eigvals_filtered_tmp)

                idx = mod_util.get_idx_of_max(eigvals_filtered)
                omega = eigvals_filtered[idx]#*bsfl.Tscale
                
                if (rt_flag):
                    if   ( mob.boussinesq == -2 ):
                        print("omega (dimensional)     = ", omega)
                        print("omega (non-dimensional) = ", omega*bsfl_ref.Lref/bsfl_ref.Uref)
                    else:
                        print("omega (dimensional)     = ", omega*bsfl_ref.Uref/bsfl_ref.Lref)
                        print("omega (non-dimensional) = ", omega)
                else:
                    pass
                    
            omega_all[i] = omega

        return omega_all, eigvals_filtered, mob, q_eigvect

    def solve_eigenvalue_problem(self, lhs, rhs):
        """
        Function of class SolveGeneralizedEVP that solves the GEVP using linalg.eig (no guess needed)
        """
        self.EigVal, self.EigVec = linalg.eig(lhs, rhs)
        #return self.EigVal, self.EigVec

        #evalue, evect = linalg.eig(lhs, rhs)
        #return evalue, evect

    def solve_stability_secant(self, ny, mid_idx, omega0, omega00, alpha_in, beta_in, \
                               Re, map, mob, Tracking, bsfl, bsfl_ref, Local, baseflowT, rt_flag, prim_form):
        """
        Function of class SolveGeneralizedEVP that solves locally for a single eigenvalue at a time (a guess is required)
        """
        if rt_flag == True:
            tol  = 1.0e-6
        else:
            if baseflowT == 1:
                tol  = 1.0e-8
            elif baseflowT == 2:
                tol  = 1.0e-8
            else:
                sys.exit("XXXXXXXXXXXXXXXXXXXXXX 12345")
            
        u0   = 10 + 1j*10.0
        
        iter = 0
        maxiter = 100

        jxu  = 0

        #
        # Compute u00
        #
        
        # Build main stability matrices and assemble them
        mob.call_to_build_matrices(rt_flag, prim_form, ny, bsfl, bsfl_ref, Re, map)
        mob.assemble_mat_lhs(alpha_in, beta_in, omega00, Tracking, Local, bsfl_ref)

        if (rt_flag == True):
            #print("BC R-T")
            mob.set_bc_rayleigh_taylor_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
        else:
            if baseflowT == 1:
                mob.set_bc_shear_layer_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
            elif baseflowT == 2:
                #print("plane poiseuille")
                mob.set_bc_plane_poiseuille_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
            else:
                sys.exit("No function associated with that value")
            
        # Solve Linear System
        SOL = linalg.solve(mob.mat_lhs, mob.vec_rhs)
        #SOL = np.linalg.lstsq(mob.mat_lhs, mob.vec_rhs, rcond=None)

        #Extract boundary condition
        #mid_idx = mod_util.get_mid_idx(ny)
        u00 = SOL[jxu]

        res = 1
        
        #
        # Main loop
        #
        
        while ( ( abs(u0) > tol**2. or abs(res) > tol ) and iter < maxiter ):
        
            iter=iter+1

            # Assemble stability matrices
            mob.assemble_mat_lhs(alpha_in, beta_in, omega0, Tracking, Local, bsfl_ref)

            if (rt_flag == True):
                mob.set_bc_rayleigh_taylor_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
            else:
                if baseflowT == 1:
                    mob.set_bc_shear_layer_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
                elif baseflowT == 2:
                    #print("plane poiseuille")
                    mob.set_bc_plane_poiseuille_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
                else:
                    sys.exit("No function associated with that value -----  22222222")
            
            #Solve Linear System
            SOL = linalg.solve(mob.mat_lhs, mob.vec_rhs)
            #SOL = np.linalg.lstsq(mob.mat_lhs, mob.vec_rhs, rcond=None)
        
            # Extract boundary condition
            u0  = SOL[jxu]
            q   = SOL
            
            #
            # Update
            #
            
            omegaNEW = omega0 - u0*(omega0-omega00)/(u0-u00)
            
            omega00  = omega0
            omega0   = omegaNEW

            res      = abs(omega0)-abs(omega00)
            
            # New
            u00      = u0
            
            print("Iteration %2d: abs(u0) = %10.5e, abs(res) = %10.5e, omega = %21.11e, %21.11e" % (iter, np.abs(u0), abs(res), omega0.real, omega0.imag))
            
        if ( iter == maxiter ): sys.exit("No Convergence in Secant Method...")

        # Get final eigenvectors
        mob.assemble_mat_lhs(alpha_in, beta_in, omega0, Tracking, Local, bsfl_ref)
        qfinal = linalg.solve(mob.mat_lhs, mob.vec_rhs)

        return omega0, qfinal


        #mod_util.plot_real_imag_part(q[0:ny], "ueig_ps", map.y)
        #mod_util.plot_four_vars_amplitude(q[0:ny], q[ny:2*ny], q[2*ny:3*ny], q[3*ny:4*ny], "u", "v", "w", "p", map.y)

        #print("alpha_in = ", alpha_in)
        #input("Debugguing right here!!!")
        



        #u00 = np.matmul(map.D1, SOL[0:ny])
        #u00 = u00[mid_idx] 

        #print("mid_idx = ", mid_idx)
        #u00 = np.matmul(map.D1, SOL[ny:2*ny])
        #u00 = u00[mid_idx]

        # Using pressure for now
        # I checked from shifted eigenfunctions from global solver:
        # imaginary part of pressure has a max at y=0 and therefore
        # derivative should be zero dp_i/dy(y=0)=0.
        # On the other hand real part of pressure is zero at y=0 but
        # dp_r/dy(y=0) ~= 0 (is not zero) so cannot use here

        #u00 = np.matmul(map.D1, SOL[3*ny:4*ny].imag)
        #u00 = u00[mid_idx]

        #u00 = np.matmul(map.D1, SOL[1*ny:2*ny])
        #u00 = u00[mid_idx] #+ 1j*alpha_in*(-1)

        #print("u00 = ", u00)
        



                #u0 = np.matmul(map.D1, SOL[0:ny])
                #u0 = u0[mid_idx]

                #u0 = np.matmul(map.D1, SOL[ny:2*ny])
                #u0 = u0[mid_idx]

                # Using pressure for now
                # I checked from shifted eigenfunctions from global solver:
                # imaginary part of pressure has a max at y=0 and therefore
                # derivative should be zero dp_i/dy(y=0)=0.
                # On the other hand real part of pressure is zero at y=0 but
                # dp_r/dy(y=0) ~= 0 (is not zero) so cannot use here
                
                #u0 = np.matmul(map.D1, SOL[3*ny:4*ny].imag)
                #u0 = u0[mid_idx]

                # Make sure dv/dy has right value 
                #u0 = np.matmul(map.D1, SOL[1*ny:2*ny])
                #u0 = u0[mid_idx] + 1j*alpha_in*(-1)

                #print("u0 = ", u0)

                #print("-1j*alpha_in*(-1) = ",-1j*alpha_in*(-1))



        




                #omega_imag = omega.imag
                #omega      = alpha[i]/2. + omega_imag*1j
                #print("omega 112233445566= ", omega)




                # if (rt_flag):
                #     if (prim_form==1):
                #         mob.set_bc_rayleigh_taylor(mob.mat_lhs, mob.mat_rhs, ny, map)
                #         #mob.set_matrices_rayleigh_taylor_boussinesq_mixing(ny, bsfl, map, Re)
                #         print("This form")
                #     else:
                #         mob.set_bc_rayleigh_taylor_inviscid(mob.mat_lhs, mob.mat_rhs, ny, map)
                # else:
                #     mob.set_bc_shear_layer(mob.mat_lhs, mob.mat_rhs, ny, map)


                













    # def solve_stability_secant(self, ny, mid_idx, omega0, omega00, alpha_in, beta_in, \
    #                            Re, map, Tracking, bsfl, bsfl_ref, Local, baseflowT, rt_flag, prim_form):
    #     """
    #     Function of class SolveGeneralizedEVP that solves locally for a single eigenvalue at a time (a guess is required)
    #     """
    #     if rt_flag == True:
    #         tol  = 1.0e-10
    #     else:
    #         if baseflowT == 1:
    #             tol  = 1.0e-8
    #         elif baseflowT == 2:
    #             tol  = 1.0e-8
    #         else:
    #             sys.exit("XXXXXXXXXXXXXXXXXXXXXX 12345")
            
    #     u0   = 10 + 1j*10.0
        
    #     iter = 0
    #     maxiter = 100

    #     jxu  = 0

    #     #
    #     # Compute u00
    #     #

    #     # Set Local Operators
    #     #print("alpha_in, beta_in, omega00 = ",alpha_in, beta_in, omega00)

    #     # Create instance for Class BuildMatrices
    #     mob = mbm.BuildMatrices(ny, rt_flag, prim_form)
        
    #     # Build main stability matrices        
    #     mob.call_to_build_matrices(rt_flag, prim_form, ny, bsfl, bsfl_ref, Re, map)

    #     # Build main stability matrices
    #     # if (rt_flag == True):
    #     #    if (prim_form==1):
    #     #        mob.set_matrices_rayleigh_taylor(ny, bsfl, bsfl_ref, map)
    #     #    else:
    #     #        mob.set_matrices_rayleigh_taylor_inviscid_w_equation(ny, bsfl, map)
    #     # else:
    #     #    mob.set_matrices(ny, Re, bsfl, map)

    #     #mob.set_matrices(ny, Re, bsfl, map)

    #     mob.assemble_mat_lhs(alpha_in, beta_in, omega00, Tracking, Local, bsfl_ref)

    #     if (rt_flag == True):
    #         #print("BC R-T")
    #         mob.set_bc_rayleigh_taylor_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
    #     else:
    #         if baseflowT == 1:
    #             mob.set_bc_shear_layer_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
    #         elif baseflowT == 2:
    #             #print("plane poiseuille")
    #             mob.set_bc_plane_poiseuille_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
    #         else:
    #             sys.exit("No function associated with that value")
            
    #     # Solve Linear System
    #     SOL = linalg.solve(mob.mat_lhs, mob.vec_rhs)

    #     #if( np.allclose(np.dot(mob.mat_lhs, SOL), mob.vec_rhs) == False ):
    #         #omega00 = 0.0 - 1j*15.0
    #     #    warnings.warn("Not a properly converged solution.......")
    #         #sys.exit("Not a properly converged solution.......")
    
    #     #Extract boundary condition
    #     #mid_idx = mod_util.get_mid_idx(ny)
    #     u00 = SOL[jxu]
        
    #     #
    #     # Main loop
    #     #
        
    #     while (abs(u0) > tol and iter < maxiter):
        
    #         iter=iter+1

    #         # Set Local Operators
    #         # Create instance for Class BuildMatrices
    #         #mob = mbm.BuildMatrices(ny, rt_flag, prim_form)

    #         # Build main stability matrices
    #         #mob.call_to_build_matrices(rt_flag, prim_form, ny, bsfl, bsfl_ref, Re, map)

    #         # Build main stability matrices
    #         # if (rt_flag == True):
    #         #     if (prim_form==1):
    #         #         mob.set_matrices_rayleigh_taylor(ny, bsfl, bsfl_ref, map)
    #         #     else:
    #         #         sys.exit("Not for this.......")
    #         #         mob.set_matrices_rayleigh_taylor_inviscid_w_equation(ny, bsfl, map)
    #         # else:
    #         #     mob.set_matrices(ny, Re, bsfl, map)

    #         #print("alpha_in, beta_in, omega0 = ", alpha_in, beta_in, omega0)
    #         #mob.set_matrices(ny, Re, bsfl, map)
    #         mob.assemble_mat_lhs(alpha_in, beta_in, omega0, Tracking, Local, bsfl_ref)

    #         if (rt_flag == True):
    #             mob.set_bc_rayleigh_taylor_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
    #         else:
    #             if baseflowT == 1:
    #                 mob.set_bc_shear_layer_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
    #             elif baseflowT == 2:
    #                 #print("plane poiseuille")
    #                 mob.set_bc_plane_poiseuille_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
    #             else:
    #                 sys.exit("No function associated with that value -----  22222222")
            
    #         # if baseflowT == 1:
    #         #     mob.set_bc_shear_layer_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
    #         # elif baseflowT == 2:
    #         #     #print("plane poiseuille")
    #         #     mob.set_bc_plane_poiseuille_secant(mob.mat_lhs, mob.vec_rhs, ny, map, alpha_in, beta_in)
    #         # else:
    #         #     sys.exit("No function associated with that value -----  22222222")

    #         #Solve Linear System
    #         SOL = linalg.solve(mob.mat_lhs, mob.vec_rhs)
        
    #         #if( np.allclose(np.dot(mob.mat_lhs, SOL), mob.vec_rhs) == False ):

    #             #omega0 = omega00
    #             #sys.exit("Singular Matrix, Stopping...")

    #             #break

    #         #else:
       
    #         # Extract boundary condition
    #         u0  = SOL[jxu]
    #         q   = SOL
            
    #         #
    #         # Update
    #         #
            
    #         omegaNEW = omega0 - u0*(omega0-omega00)/(u0-u00)
            
    #         omega00  = omega0
    #         omega0   = omegaNEW
            
    #         # NEW
    #         u00      = u0
            
    #         print("Iteration %2d: abs(u0) = %10.5e, omega = %21.11e, %21.11e" % (iter, np.abs(u0), omega0.real, omega0.imag))
            
    #     if ( iter == maxiter ): sys.exit("No Convergence in Secant Method...")

    #     return omega0, q































    # def solve_stability_problem(self, map, alpha, beta, omega, Re, ny, Tracking, mid_idx, bsfl, bsfl_ref, Local, rt_flag, prim_form, baseflowT, iarr, ire, lmap):
    #     """
    #     Function...
    #     """        
    #     npts = len(alpha)

    #     omega_all = np.zeros(npts, dpc)
        
    #     for i in range(0, npts):

    #         print("")
    #         print("Solving for alpha #%i = %21.11e" % (i, alpha[i]) )
    #         print("---------------------------------------------")
    #         print("")

    #         if Tracking and Local:
    #             print("Tracking and Local!")
    #             print("")

    #             # Create instance for Class BuildMatrices
    #             #mob = mbm.BuildMatrices(ny, rt_flag, prim_form)
                
    #             # Build main stability matrices
    #             #mob.call_to_build_matrices(rt_flag, prim_form, ny, bsfl, bsfl_ref, Re, map)

    #             # CLEAN UP THIS PART SAME AS BELOW FOR GEVP                
    #             # if (rt_flag == True):
    #             #     if (prim_form==1):
    #             #         mob = mbm.BuildMatrices(5*ny, rt_flag, prim_form) # System matrices for RT are 5*ny by 5*ny (variables: u, v, w , p, rho)
    #             #     else:
    #             #         sys.exit("Just used for inviscid debugging 123456789")
    #             #         mob = mbm.BuildMatrices(ny) # Inviscid system matrix for RT based on w-equation only (see Chandrasekhar page 433)
    #             # else:
    #             #     mob = mbm.BuildMatrices(4*ny) # System matrices are 4*ny by 4*ny (variables: u, v, w , p)

    #             # # Build main stability matrices
    #             # if (rt_flag == True):
    #             #     if (prim_form==1):
    #             #         mob.set_matrices_rayleigh_taylor(ny, bsfl, bsfl_ref, map)
    #             #         #mob.set_matrices_rayleigh_taylor_boussinesq_mixing(ny, bsfl, map, Re)
    #             #     else:
    #             #         mob.set_matrices_rayleigh_taylor_inviscid_w_equation(ny, bsfl, bsfl_ref, map)
    #             # else:
    #             #     mob.set_matrices(ny, Re, bsfl, map)
                    
    #             # Extrapolate omega
    #             if (i > 1):
    #                 pass
    #                 # Extrapolate in alpha space
    #                 omega = mod_util.extrapolate_in_alpha(iarr, alpha, i, ire)
    #             elif (ire > 1):
    #                 pass
    #                 # Extrapolate in Reynolds space
    #                 omega = mod_util.extrapolate_in_reynolds(iarr, i, ire)

    #             print("omega here = ", omega)
                
    #             omega, q_eigenvects = self.solve_stability_secant(ny, mid_idx, omega, omega + omega*1e-8, alpha[i], beta, \
    #                                                               Re, map, Tracking, bsfl, bsfl_ref, Local, baseflowT, rt_flag, prim_form)

    #             #print("omega that is written to omega_array = ", omega)
    #             #print("phase speed cr = ", omega.real/alpha[i])
    #             iarr.omega_array[ire, i] = omega

    #             #print("iarr.omega_array[ire, i].real = ", iarr.omega_array[ire, i].real)

    #             # Only tracking a single mode here!!!!!
    #             eigvals_filtered = 0.0

    #             #mod_util.write_eigvects_out(q_eigenvects, map.y, i, ny)
    #         else:

    #             # Create instance for Class BuildMatrices
    #             mob = mbm.BuildMatrices(ny, rt_flag, prim_form)
                
    #             # Build main stability matrices
    #             mob.call_to_build_matrices(rt_flag, prim_form, ny, bsfl, bsfl_ref, Re, map)

    #             # Assemble matrices
    #             if (prim_form==1):
    #                 mob.assemble_mat_lhs(alpha[i], beta, omega, Tracking , Local, bsfl_ref)
    #             else:
    #                 sys.exit("Just used for inviscid debugging")
    #                 mob.assemble_mat_lhs_rt_inviscid(alpha[i], beta, omega, Tracking , Local)

    #             mob.call_to_set_bc(rt_flag, prim_form, ny, map)

    #             #mob.mat_lhs = np.conj(mob.mat_lhs)
    #             #mob.mat_rhs = np.conj(mob.mat_rhs)
                
    #             self.solve_eigenvalue_problem(mob.mat_lhs, mob.mat_rhs)

    #             eigvals_filtered = self.EigVal[ np.abs(self.EigVal) < 10000. ]
    #             q_eigenvects     = self.EigVec

    #             if (rt_flag==True and prim_form==0):
    #                 neigs = len(eigvals_filtered)
    #                 eigvals_filtered_tmp = eigvals_filtered
    #                 eigvals_filtered = np.zeros(2*neigs, dpc)
    #                 eigvals_filtered[0:neigs] = 1j/np.sqrt(eigvals_filtered_tmp) # I set it to the complex part because growth rate
    #                 eigvals_filtered[neigs:2*neigs] = -1j/np.sqrt(eigvals_filtered_tmp)

    #             idx = mod_util.get_idx_of_max(eigvals_filtered)
    #             if (rt_flag):
    #                 print("omega (dimensional)     = ", eigvals_filtered[idx])
    #                 omega = eigvals_filtered[idx]#*bsfl.Tscale
    #                 print("omega (non-dimensional) = ", omega)
    #             else:
    #                 omega = eigvals_filtered[idx]
                    
    #         omega_all[i] = omega

    #     return omega_all, eigvals_filtered, mob







    
