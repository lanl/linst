import sys
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
    def __init__(self, size):
        """
        Constructor of class SolveGeneralizedEVP
        """
        self.size = size
        self.EigVal = np.zeros(size*size, dpc)
        self.EigVec = np.zeros((size, size), dpc)

    def solve_stability_problem(self, mob, map, alpha, beta, omega, Re, ny, Tracking, mid_idx, bsfl, Local):
        """
        Function...
        """        
        ct = 0
        npts = len(alpha)

        omega_all = np.zeros(npts, dpc)
        
        for i in range(0, npts):
            ct = ct + 1
            print("")
            print("Solving for alpha = ", alpha[i])
            print("------------------------------")
            print("")

            if Tracking and Local:
                mob = mbm.BuildMatrices(4*ny) # System matrices are 4*ny by 4*ny
                mob.set_matrices(ny, Re, beta, bsfl, map)
                omega = self.solve_stability_secant(ny, mid_idx, omega, omega + omega*1e-8, alpha[i], beta, Re, mob, map, Tracking, bsfl)
                print("omega = ", omega)
            else:
                mob.assemble_mat_lhs(alpha[i], beta, omega, Tracking , Local)
                mob.set_bc_shear_layer(mob.mat_lhs, mob.mat_rhs, ny, map)

                #mob.mat_lhs = np.conj(mob.mat_lhs)
                #mob.mat_rhs = np.conj(mob.mat_rhs)
                
                self.solve_eigenvalue_problem(mob.mat_lhs, mob.mat_rhs)

                eigvals_filtered = self.EigVal[ np.abs(self.EigVal) < 10000. ]

                idx = mod_util.get_idx_of_max(eigvals_filtered)

                omega = eigvals_filtered[idx]
                print("omega = ", omega)

                omega_all[i] = omega

        return omega_all, eigvals_filtered

    def solve_eigenvalue_problem(self, lhs, rhs):
        """
        Function of class SolveGeneralizedEVP that solves the GEVP using linalg.eig (no guess needed)
        """
        self.EigVal, self.EigVec = linalg.eig(lhs, rhs)
        #return self.EigVal, self.EigVec

        #evalue, evect = linalg.eig(lhs, rhs)
        #return evalue, evect

    def solve_stability_secant(self, ny, mid_idx, omega0, omega00, alpha_in, beta_in, Re, mob, map, Tracking, bsfl):
        """
        Function of class SolveGeneralizedEVP that solves locally for a single eigenvalue at a time (a guess is required)
        """
        tol  = 1.0e-7
        
        u0   = 10 + 1j*0.0
        
        iter = 0
        maxiter = 100
        
        jxu  = 1 # u0 is at j=1

        #
        # Compute u00
        #

        # Set Local Operators
        print("alpha_in, beta_in, omega00 = ",alpha_in, beta_in, omega00)
        mob.set_matrices(ny, Re, beta_in, bsfl, map)
        mob.assemble_mat_lhs(alpha_in, beta_in, omega00, Tracking, Local)
        mob.set_bc_shear_layer_secant(mob.mat_lhs, mob.vec_rhs, ny, map)
    
        # Solve Linear System
        SOL = linalg.solve(mob.mat_lhs, mob.vec_rhs)

        if( np.allclose(np.dot(mob.mat_lhs, SOL), mob.vec_rhs) == False ):
            omega00 = 0.0 - 1j*15.0
            sys.exit("Not a properly converged solution.......")
    
        #Extract boundary condition
        u00 = SOL[jxu]
        
        #
        # Main loop
        #
        
        while (abs(u0) > tol and iter < maxiter):
        
            iter=iter+1

            # Set Local Operators
            mob.set_matrices(ny, Re, beta_in, bsfl, map)
            mob.assemble_mat_lhs(alpha_in, beta_in, omega0, Tracking, Local)
            mob.set_bc_shear_layer_secant(mob.mat_lhs, mob.vec_rhs, ny, map)

            #Solve Linear System
            SOL = linalg.solve(mob.mat_lhs, mob.vec_rhs)
        
            if( np.allclose(np.dot(mob.mat_lhs, SOL), mob.vec_rhs) == False ):

                omega0 = omega00
                sys.exit("Singular Matrix, Stopping...")

                break

            else:
       
                # Extract boundary condition
                u0  = SOL[jxu]
                q   = SOL
                
                #
                # Update
                #
                
                omegaNEW = omega0 - u0*(omega0-omega00)/(u0-u00)
                
                omega00  = omega0
                omega0   = omegaNEW
                
                # NEW
                u00      = u0

                #print('The value of pi is approximately %5.3f.' % math.pi)
                print("Iteration %2d: abs(u0) = %5.5e" % (iter, np.abs(u0)))
                #write(fileID,'(1X,A,I4,A,F25.15,A,F25.15,A,F25.15)') 'Iteration #', iter,', abs(u0) = ',abs(u0),', alpha0 = ',dble(alpha0),' + i',dimag(alpha0)

        print("Converged omega = ", omega0)
        if ( iter == maxiter ): sys.exit("No Convergence in Secant Method...")

        mod_util.plot_real_imag_part(q[0:ny], "ueig_ps", map.y)

        input()
        
        return omega0

