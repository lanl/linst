import sys
import warnings
import numpy as np
import scipy as sp
import scipy.sparse.linalg
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
    def __init__(self, ny, rt_flag, prim_form, mtmp):
        """
        Constructor of class SolveGeneralizedEVP
        """
        self.mtmp = mtmp
        boussi = mtmp.boussinesq
        if (rt_flag == True):
            if (prim_form==1):
                size = 5*ny
                if (boussi == -555): size = 4*ny 
            else:
                size = ny
        else:
            size = 4*ny

        self.size = size
        self.EigVal = np.zeros(size*size, dpc)
        self.EigVec = np.zeros((size, size), dpc)

    def solve_secant_problem(self, map, alpha, beta, omega, Re, ny, Tracking, mid_idx, bsfl, bsfl_ref, Local, rt_flag, prim_form, baseflowT, iarr, ire, lmap):
        """
        Function...
        """        
        npts = len(alpha)

        omega_all = np.zeros(npts, dpc)
        q_eigvect = np.zeros(self.size, dpc)
        
        
        for i in range(0, npts):

            print("")
            print("Solving for wavenumber alpha = %10.5e (%i/%i)" % (alpha[i], i, npts-1) )
            print("------------------------------------------------")
            print("")

            if Tracking and Local:
                # Extrapolate omega
                if (i > 1):
                    #pass
                    # Extrapolate in alpha space
                    omega = mod_util.extrapolate_in_alpha(iarr, alpha, i, ire)
                    print("omega extrapolated = ", omega)
                elif (ire > 1):
                    #pass
                    # Extrapolate in Reynolds space
                    omega = mod_util.extrapolate_in_reynolds(iarr, i, ire)

                # Create instance for Class BuildMatrices
                mob = mbm.BuildMatrices(ny, rt_flag, prim_form)

                print("i, alpha[i], beta, omega = ", i, alpha[i], beta, omega)
                
                omega, q_eigvect = self.solve_stability_secant(ny, mid_idx, omega, omega + omega*1e-5, alpha[i], beta, \
                                                                  Re, map, mob, Tracking, bsfl, bsfl_ref, Local, baseflowT, rt_flag, prim_form)

                iarr.omega_array[ire, i] = omega

                # Only tracking a single mode here!!!!!
                eigvals_filtered = 0.0

                #mod_util.write_eigvects_out(q_eigenvects, map.y, i, ny)

    def solve_general_problem(self, map, alpha, beta, omega, Re, ny, mid_idx, bsfl, bsfl_ref, rt_flag, prim_form, baseflowT, iarr, ire, lmap):
        # Create instance for Class BuildMatrices
        mob = self.mtmp
                
        # Build main stability matrices
        mob.call_to_build_matrices(ny, bsfl, bsfl_ref, map)

        # Assemble matrices
        if (prim_form==1):
            mob.assemble_mat_lhs(alpha[0], beta, omega, bsfl_ref)
        else:
            sys.exit("Just used for inviscid debugging")
            mob.assemble_mat_lhs_rt_inviscid(alpha[0], beta, omega, Tracking , Local)

        mob.call_to_set_bc(ny, map)

        #mob.mat_lhs = np.conj(mob.mat_lhs)
        #mob.mat_rhs = np.conj(mob.mat_rhs)
                
        self.solve_eigenvalue_problem(mob.mat_lhs, mob.mat_rhs)

        eigvals_filtered = self.EigVal[ np.abs(self.EigVal) < 10000. ]
        q_eigenvects     = self.EigVec

        if (rt_flag==True and prim_form==0):
            input("In here")
            
            neigs = len(eigvals_filtered)
            eigvals_filtered_tmp = eigvals_filtered
            eigvals_filtered = np.zeros(2*neigs, dpc)
            eigvals_filtered[0:neigs] = 1j/np.sqrt(eigvals_filtered_tmp) # I set it to the complex part because growth rate
            eigvals_filtered[neigs:2*neigs] = -1j/np.sqrt(eigvals_filtered_tmp)

        idx = mod_util.get_idx_of_max(eigvals_filtered)
        omega = eigvals_filtered[idx]#*bsfl.Tscale                


        if (rt_flag):
            i = 0
            print("ire, i = ", ire, i)
            
            if   ( mob.boussinesq == -2 or mob.boussinesq == 10 ):
                iarr.omega_dim[ire, i] = omega
                iarr.omega_nondim[ire, i] = omega*bsfl_ref.Lref/bsfl_ref.Uref
                print("omega (dimensional)     = ", iarr.omega_dim[ire, i])
                print("omega (non-dimensional) = ", iarr.omega_nondim[ire, i])
                print("omega (non-dimensional Chandrasekhar) = ", iarr.omega_dim[ire, i]*bsfl_ref.Lscale_Chandra/bsfl_ref.Uref_Chandra)
            else:
                iarr.omega_dim[ire, i] = omega*bsfl_ref.Uref/bsfl_ref.Lref
                iarr.omega_nondim[ire, i] = omega
                print("omega (dimensional)     = ", iarr.omega_dim[ire, i])
                print("omega (non-dimensional) = ", iarr.omega_nondim[ire, i])
                print("omega (non-dimensional Chandrasekhar) = ", iarr.omega_dim[ire, i]*bsfl_ref.Lscale_Chandra/bsfl_ref.Uref_Chandra)

        return eigvals_filtered, mob

    def solve_eigenvalue_problem(self, lhs, rhs):
        """
        Function of class SolveGeneralizedEVP that solves the GEVP using linalg.eig (no guess needed)
        """
        self.EigVal, self.EigVec = linalg.eig(lhs, rhs)

        #print("self.EigVec.shape = ", self.EigVec.shape)
        #print("self.EigVal.shape = ", self.EigVal.shape)
        #return self.EigVal, self.EigVec

        #evalue, evect = linalg.eig(lhs, rhs)
        #return evalue, evect

    def solve_stability_secant(self, ny, mid_idx, omega0, omega00, alpha_in, beta_in, \
                               Re, map, mob, Tracking, bsfl, bsfl_ref, Local, baseflowT, rt_flag, prim_form):
        """
        Function of class SolveGeneralizedEVP that solves locally for a single eigenvalue at a time (a guess is required)
        """

        #mid_idx = mod_util.get_mid_idx(ny)
        
        #print("omega0, omega00 = ", omega0, omega00)
        
        if rt_flag == True:
            tol1  = 1.0e-5
            tol2  = 1.0e-5
        else:
            if baseflowT == 1:
                tol  = 1.0e-8
            elif baseflowT == 2:
                tol  = 1.0e-8
            else:
                sys.exit("XXXXXXXXXXXXXXXXXXXXXX 12345")
            
        u0   = 10
        
        iter = 0
        maxiter = 100

        jxu  = 5*ny-1 # 0

        if (rt_flag): jxu  = 3*ny-1 # use 3*ny-1 for w and 5*ny-1 for rho

        #
        # Compute u00
        #
        
        # Build main stability matrices and assemble them
        mob.call_to_build_matrices(rt_flag, prim_form, ny, bsfl, bsfl_ref, Re, map, alpha_in)
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
        #SOL = SOL/np.max(SOL)
        
        #SOL = np.linalg.lstsq(mob.mat_lhs, mob.vec_rhs, rcond=None)

        #print("mid_idx = ", mid_idx)

        #print("SOL = ", SOL)
        #Extract boundary condition
        #print("SOL[0*ny+mid_idx] = ", SOL[0*ny+mid_idx])
        #print("SOL[1*ny+mid_idx] = ", SOL[1*ny+mid_idx])
        #print("SOL[2*ny+mid_idx] = ", SOL[2*ny+mid_idx])
        #print("SOL[3*ny+mid_idx] = ", SOL[3*ny+mid_idx])
        #print("SOL[4*ny+mid_idx] = ", SOL[4*ny+mid_idx])
        
        u00 = SOL[jxu]

        #print("SOL[jxu]=",SOL[jxu])
        #print("u00 = ", u00)

        res = 1
        
        #
        # Main loop
        #
        
        while ( ( abs(u0) > tol2 or abs(res) > tol1 ) and iter < maxiter ):
        #while ( ( abs(u0) > tol**1.5 or abs(res) > tol ) and iter < maxiter ):
        #while ( ( abs(u0) > tol or abs(res) > tol ) and iter < maxiter ):
        #while ( abs(res) > tol and iter < maxiter ):
        
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
            #SOL = SOL/np.max(SOL)
            #SOL = np.linalg.lstsq(mob.mat_lhs, mob.vec_rhs, rcond=None)
        
            # Extract boundary condition
            u0  = SOL[jxu]
            q   = SOL

            #print("SOL[jxu]=",SOL[jxu])
            #print("u0=", u0)
            
            #
            # Update
            #

            #print("u0-u00 = ", u0-u00)
            
            omegaNEW = omega0 - u0*(omega0-omega00)/(u0-u00)
            
            omega00  = omega0
            omega0   = omegaNEW

            res      = abs(omega0)-abs(omega00)
            
            # New
            u00      = u0
            
            print("Iteration %2d: abs(u0) = %10.5e, abs(res) = %10.5e, omega = %21.11e, %21.11e" % (iter, np.abs(u0), abs(res), omega0.real, omega0.imag))
            
        if ( iter == maxiter ): sys.exit("No Convergence in Secant Method...")

        # if   ( mob.boussinesq == -2 or mob.boussinesq == 10 ):
        #     print("omega (dimensional)     = ", omega0.imag)
        #     print("omega (non-dimensional) = ", omega0.imag*bsfl_ref.Lref/bsfl_ref.Uref)
        #     print("omega (non-dimensional Chandrasekhar) = ", omega0.imag*bsfl_ref.Lscale_Chandra/bsfl_ref.Uref_Chandra)
        # else:
        #     print("omega (dimensional)     = ", omega0.imag*bsfl_ref.Uref/bsfl_ref.Lref)
        #     print("omega (non-dimensional) = ", omega0.imag)
        #     print("omega (non-dimensional Chandrasekhar) = ", omega0.imag*bsfl_ref.Uref/bsfl_ref.Lref*bsfl_ref.Lscale_Chandra/bsfl_ref.Uref_Chandra)

        # Get final eigenvectors
        mob.assemble_mat_lhs(alpha_in, beta_in, omega0+1.e-8, Tracking, Local, bsfl_ref)
        qfinal = linalg.solve(mob.mat_lhs, mob.vec_rhs)

        UseEigs=0
        
        if UseEigs==1:
            # Build main stability matrices
            mob.call_to_build_matrices(rt_flag, prim_form, ny, bsfl, bsfl_ref, Re, map, alpha_in)

            # Assemble matrices
            print("omega0 = ",omega0)

            Tracking = False
            Local = False
            mob.assemble_mat_lhs(alpha_in, beta_in, omega0, Tracking , Local, bsfl_ref)
            mob.call_to_set_bc(rt_flag, prim_form, ny, map)

            values, vectors = scipy.sparse.linalg.eigs(mob.mat_lhs, k=1, M=mob.mat_rhs, sigma=omega0, tol=1.e-10)
            
            print("values = ", values)
            print("vectors=",vectors)
            len_vec = len(vectors)
            print("The size of vectors is:", len_vec)

            Tracking = True
            Local = True

            mod_util.plot_real_imag_part(vectors[0*ny:1*ny], "u", map.y, rt_flag, mob)
            mod_util.plot_real_imag_part(vectors[4*ny:5*ny], "r", map.y, rt_flag, mob)

            input("Pause after ARPACK")
        
        plot_eig_vec = 0
        if (plot_eig_vec==1):
            ylimp = 0.0
            mod_util.plot_real_imag_part(qfinal[0*ny:1*ny], "u", map.y, rt_flag, mob, ylimp)
            #mod_util.plot_real_imag_part(qfinal[1*ny:2*ny], "v", map.y, rt_flag, mob, ylimp)
            mod_util.plot_real_imag_part(qfinal[2*ny:3*ny], "w", map.y, rt_flag, mob, ylimp)
            #mod_util.plot_real_imag_part(qfinal[3*ny:4*ny], "p", map.y, rt_flag, mob, ylimp)
            mod_util.plot_real_imag_part(qfinal[4*ny:5*ny], "r", map.y, rt_flag, mob, ylimp)
            
            input("Local solver: check eigenfunctions 2222222222222")


        return omega0, qfinal
