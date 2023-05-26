import sys
import warnings
import numpy as np
import scipy as sp
import scipy.sparse.linalg
from scipy import linalg
import module_utilities as mod_util
import build_matrices as mbm
#import scipy as sp
import matplotlib.pyplot as plt

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

class SolveGeneralizedEVP:
    """
    This class define a solution object and contain the main function that provides the solution 
    to the generalized eigenvalue problem (GEVP)
    """
    def __init__(self, map, bsfl):
        """
        Constructor of class SolveGeneralizedEVP
        """
        self.ny = map.y.size
        self.map = map
        self.bsfl = bsfl
        self.mid_idx = mod_util.get_mid_idx(self.ny)
        
    def solve(self, alpha, beta, omega_guess):
        # Build matrices and solve global/local problem
        self.alpha = alpha
        self.beta = beta
        self.target1 = omega_guess

        self.omega_array = np.zeros((1, np.size(alpha)), dpc)

        self.Local = False

        # if ( np.size(alpha) != 1 ):
        #      if ( alpha[0] == alpha[-1] ):
        #          self.alpha = np.zeros(1, dp)
        #          self.alpha = alpha[0]
             
        # if ( np.size(beta) != 1 ):
        #     if ( beta[0] == beta[-1] ):
        #         self.beta = np.zeros(1, dp)
        #         self.beta = beta[0]

        # if ( np.size(self.Re) != 1 ):
        #     if ( self.Re[0] == self.Re[-1] ):
        #         self.Re = np.zeros(1, dp)
        #         self.Re = self.Re[0]
        
        if ( np.size(alpha) != 1 or np.size(beta) != 1 or np.size(self.Re) != 1 ):
            self.Local = True

        if (self.Local):
            self.solve_secant_problem(alpha, beta, omega_guess)
        else:
            self.eigvals_filtered = self.solve_general_problem(alpha, beta, omega_guess)

    def solve_secant_problem(self, alpha, beta, omega):
        """
        Function...
        """
        ire = 0
        input("Hardcoded ire to 0!!!!!!")

        #nre = len(self.Re)
        npts = len(alpha)
        q_eigvect = np.zeros(np.size(self.vec_rhs), dpc)

        #for ire in range(0, nre):

        #print("")
        #print("Solving for Reynolds number Re = %10.5e (%i/%i)" % (self.Re[ire], ire, nre-1) )
        #print("==================================================")

        
        for i in range(0, npts):
            
            print("")
            print("Solving for wavenumber alpha = %10.5e (%i/%i)" % (alpha[i], i, npts-1) )
            print("------------------------------------------------")
            print("")
            
            # Extrapolate omega
            if (i > 1):
                # Extrapolate in alpha space
                omega = mod_util.extrapolate_in_alpha(self.omega_array, alpha, i, ire)
            elif (ire > 1):
                # Extrapolate in Reynolds space
                omega = mod_util.extrapolate_in_reynolds(self.omega_array, i, ire)
                
            #self.omega_array[ire, i] = self.solve_stability_secant(omega, omega + omega*1e-5, alpha[i], beta, self.Re[ire] )
            self.omega_array[ire, i] = self.solve_stability_secant(omega, omega + omega*1e-5, alpha[i], beta )
            
    def call_to_build_matrices(*args, **kwargs):
        raise NotImplemented("Use a specific equation class, not SolveGeneralizedEVP.")
                
    def solve_general_problem(self, alpha, beta, omega_guess):
        self.call_to_build_matrices()
        self.assemble_mat_lhs(alpha, beta, omega_guess)
        self.call_to_set_bc()

        #mob.mat_lhs = np.conj(mob.mat_lhs)
        #mob.mat_rhs = np.conj(mob.mat_rhs)
                
        self.EigVal, self.EigVec = linalg.eig(self.mat_lhs, self.mat_rhs)

        eigvals_filtered = self.EigVal[ np.abs(self.EigVal) < 10000. ]
        q_eigenvects     = self.EigVec

        idx = mod_util.get_idx_of_max(eigvals_filtered)
        self.omega_max = eigvals_filtered[idx]#*self.bsfl.Tscale
        print("")
        print("Filtered eigenvalue with maximumgrowth rate: ", self.omega_max)

        return eigvals_filtered

    def solve_stability_secant(self, omega0, omega00, alpha_in, beta_in):
        """
        Function of class SolveGeneralizedEVP that solves locally for a single eigenvalue at a time (a guess is required)
        """

        mid_idx = self.mid_idx
        
        if self.bsfl.rt_flag == True:
            tol1  = 1.0e-5
            tol2  = 1.0e-5
            print("tol1, tol2 = ", tol1, tol2)
        else:
            #if baseflowT == 1:
            tol1  = 1.0e-5
            #elif baseflowT == 2:
            tol2  = 1.0e-5
            #else:
            #    sys.exit("XXXXXXXXXXXXXXXXXXXXXX 12345")

        u0   = 10
        
        iter = 0
        maxiter = 100

        print("self.jxu = ", self.jxu)
        
        #jxu  = 5*self.ny-1 # 0
        #if (self.bsfl.rt_flag): jxu = 3*self.ny-1 # use 3*ny-1 for w and 5*ny-1 for rho

        #
        # Compute u00
        #
        
        # Build main stability matrices and assemble them
        self.call_to_build_matrices()
        self.assemble_mat_lhs(alpha_in, beta_in, omega00)
        self.call_to_set_bc_secant(self.mat_lhs, self.vec_rhs, self.ny, self.map, alpha_in, beta_in)
            
        # Solve Linear System
        SOL = linalg.solve(self.mat_lhs, self.vec_rhs)
                
        u00 = SOL[self.jxu]

        res = 1
        
        #
        # Main loop
        #
        
        while ( ( abs(u0) > tol2 or abs(res) > tol1 ) and iter < maxiter ):
        
            iter=iter+1

            # Assemble stability matrices
            self.assemble_mat_lhs(alpha_in, beta_in, omega0)
            # Set BCs
            self.call_to_set_bc_secant(self.mat_lhs, self.vec_rhs, self.ny, self.map, alpha_in, beta_in)

            #Solve Linear System
            SOL = linalg.solve(self.mat_lhs, self.vec_rhs)
        
            # Extract boundary condition
            u0  = SOL[self.jxu]
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
        self.assemble_mat_lhs(alpha_in, beta_in, omega0)
        self.call_to_set_bc_secant(self.mat_lhs, self.vec_rhs, self.ny, self.map, alpha_in, beta_in)
        self.qfinal = linalg.solve(self.mat_lhs, self.vec_rhs)

        # Plot eigenvector locally if desired
        plot_eig_vec = 0
        if (plot_eig_vec==1):
            ylimp = 0.0
            ny = self.ny
            self.get_normalized_eigvects(self.qfinal, self.bsfl.rt_flag)
            #self.plot_real_imag_part(self.qfinal[0*ny:1*ny], "u", self.map.y, self.bsfl.rt_flag, ylimp)
            #self.plot_real_imag_part(self.qfinal[1*ny:2*ny], "v", self.map.y, self.bsfl.rt_flag, ylimp)
            #self.plot_real_imag_part(self.qfinal[2*ny:3*ny], "w", self.map.y, self.bsfl.rt_flag, ylimp)
            #self.plot_real_imag_part(self.qfinal[3*ny:4*ny], "p", self.map.y, self.bsfl.rt_flag, ylimp)
            #self.plot_real_imag_part(self.qfinal[4*ny:5*ny], "r", self.map.y, self.bsfl.rt_flag, ylimp)
            
            self.plot_real_imag_part(self.ueig, "u", self.map.y, self.bsfl.rt_flag, ylimp)
            self.plot_real_imag_part(self.veig, "v", self.map.y, self.bsfl.rt_flag, ylimp)
            #self.plot_real_imag_part(self.weig, "w", self.map.y, self.bsfl.rt_flag, ylimp)
            self.plot_real_imag_part(self.peig, "p", self.map.y, self.bsfl.rt_flag, ylimp)
            #self.plot_real_imag_part(self.reig, "r", self.map.y, self.bsfl.rt_flag, ylimp)

            #self.plot_real_imag_part(np.abs(self.ueig), "u", self.map.y, self.bsfl.rt_flag, ylimp)
            #self.plot_real_imag_part(np.abs(self.veig), "v", self.map.y, self.bsfl.rt_flag, ylimp)
            #self.plot_real_imag_part(np.abs(self.peig), "p", self.map.y, self.bsfl.rt_flag, ylimp)
            
            input("Local solver: check eigenfunctions 2222222222222")

        return omega0
        
    def write_eigvals(self):
        if (self.Local):
            warnings.warn("Ignoring in write_eigvals: Writing eigenvalues (alpha_i vs alpha_r) does not make sense for Local solution")
        else:
            mod_util.write_out_eigenvalues(self.EigVal, self.ny)
    
    def plot_baseflow(self):
        mod_util.plot_baseflow(self.ny, self.map.y, yi, self.bsfl.U, self.bsfl.Up, self.map.D1)
        
    def plot_eigvals(self, ax=None):

        try:
            if not ax:
                ax = plt.gca()
            ax.plot(
                self.eigvals_filtered.real,
                self.eigvals_filtered.imag,
                'ks', markerfacecolor='none'
            )
            ax.set_xlabel(r'$\omega_r$', fontsize=20)
            ax.set_ylabel(r'$\omega_i$', fontsize=20)
            #ax.set_title('Eigenvalue spectrum')
            #ax.set_xlim([-10, 10])
            #ax.set_ylim([-1, 1])
        except AttributeError:
            plt.close()
            warnings.warn("Ignoring in plot_eigvals: Plotting eigenvalues (alpha_i vs alpha_r) does not make sense for Local solution")
        
    def identify_eigenvector_from_target(self):

        if (self.Local):
            q_eigvect = self.qfinal
            
        else:
        
            # Find index of target eigenvalue to extract eigenvector
            idx_tar1, found1 = mod_util.get_idx_of_closest_eigenvalue(self.EigVal, np.abs(self.target1), self.target1)
            
            print("")
            print("Eigenvector will be extracted for eigenvalue omega = ", self.EigVal[idx_tar1])
            q_eigvect = self.EigVec[:,idx_tar1]

            #print("np.size(q_eigvect) = ", np.size(q_eigvect))

        return q_eigvect

    def get_normalized_eigvects(self, q_eigvect, rt_flag):

        ny = self.ny

        if (self.Local):
            ueig_vec = q_eigvect[0*ny:1*ny]
            veig_vec = q_eigvect[1*ny:2*ny]
            weig_vec = q_eigvect[2*ny:3*ny]
            peig_vec = q_eigvect[3*ny:4*ny]

            if (rt_flag):
                if ( self.boussinesq == -555 ):
                    ueig_vec = q_eigvect[0*ny:1*ny]
                    veig_vec = q_eigvect[1*ny:2*ny]
                    reig_vec = q_eigvect[2*ny:3*ny]
                    peig_vec = q_eigvect[3*ny:4*ny]
                    
                    weig_vec = 0.0*ueig_vec
                else:
                    reig_vec = q_eigvect[4*ny:5*ny]
            else:
                reig_vec = 0.0*ueig_vec
            
        else:
            ueig_vec = q_eigvect[0*ny:1*ny]
            veig_vec = q_eigvect[1*ny:2*ny]
            weig_vec = q_eigvect[2*ny:3*ny]
            peig_vec = q_eigvect[3*ny:4*ny]

            if (rt_flag):
                if ( self.boussinesq == -555 ):
                    ueig_vec = q_eigvect[0*ny:1*ny]
                    veig_vec = q_eigvect[1*ny:2*ny]
                    reig_vec = q_eigvect[2*ny:3*ny]
                    peig_vec = q_eigvect[3*ny:4*ny]
                    
                    weig_vec = 0.0*ueig_vec
                else:
                    reig_vec = q_eigvect[4*ny:5*ny]
            else:
                reig_vec = 0.0*ueig_vec
        
        # Normalize eigenvectors
        self.norm_s, self.ueig, self.veig, self.weig, self.peig, self.reig = mod_util.normalize_eigenvectors(ueig_vec, veig_vec, weig_vec, peig_vec, reig_vec, rt_flag)

        # For the Generalized Eigenvalue Problem, I renormalize the whole eigenvector matrix
        if (not self.Local): self.EigVec = self.EigVec/self.norm_s

        
    def plot_eigvects(self, rt_flag):

        ylimp = 0.0
        self.plot_real_imag_part(self.ueig, "u", self.map.y, rt_flag, ylimp)
        self.plot_real_imag_part(self.veig, "v", self.map.y, rt_flag, ylimp)
        self.plot_real_imag_part(self.weig, "w", self.map.y, rt_flag, ylimp)
        self.plot_real_imag_part(self.reig, "r", self.map.y, rt_flag, ylimp)
        self.plot_real_imag_part(self.peig, "p", self.map.y, rt_flag, ylimp)

    def plot_real_imag_part(self, eig_fct, str_var, y, rt_flag, mv):

        symbols = 0
        
        ax = plt.gca()
                    
        # Get ymin-ymax
        ymin, ymax = mod_util.FindResonableYminYmax(eig_fct, y)

        if (self.boussinesq == 998): # Poiseuille flow
            ymin = -1
            ymax = 1

        # Get plot strings
        str_r, str_i = self.get_label_string(str_var)

        # Plot real part
        if (rt_flag and self.boussinesq == -555):
            ax.plot(y, eig_fct.real, 'k-', linewidth=1.5)
            if symbols==1:
                ax.plot(y, eig_fct.real, 'ks-', linewidth=1.5)
            ax.set_xlabel('x', fontsize=16)
            ax.set_ylabel(str_r, fontsize=16)            
        else:
            ax.plot(eig_fct.real, y, 'k-', linewidth=1.5)
            if symbols==1:
                ax.plot(eig_fct.real, y, 'ks-', linewidth=1.5)
            ax.set_xlabel(str_r, fontsize=16)
            ax.set_ylabel('z', fontsize=16)
            
        if (mv==0.):
            ax.set_ylim([ymin, ymax])
        else:
            ax.set_ylim([-mv, mv])

        plt.gcf().subplots_adjust(left=0.17)
        plt.gcf().subplots_adjust(bottom=0.15)

        plt.show()

        ax = plt.gca()
    
        # Plot imaginary part
        if (rt_flag and self.boussinesq == -555):
            ax.plot(y, eig_fct.imag, 'k-', linewidth=1.5)
            if symbols==1:
                ax.plot(y, eig_fct.imag, 'ks-', linewidth=1.5)
            ax.set_xlabel('x', fontsize=16)
            ax.set_ylabel(str_i, fontsize=16)
        else:
            if symbols==1:
                ax.plot(eig_fct.imag, y, 'ks-', linewidth=1.5)
            ax.plot(eig_fct.imag, y, 'k-', linewidth=1.5)
            ax.set_xlabel(str_i, fontsize=16)
            ax.set_ylabel('z', fontsize=16)
                
        if (mv==0.):
            ax.set_ylim([ymin, ymax])
        else:
            ax.set_ylim([-mv, mv])

        plt.gcf().subplots_adjust(left=0.17)
        plt.gcf().subplots_adjust(bottom=0.15)

        plt.show()

    def get_label_string(self, str_var):

        str_p = "unknown variable"
        
        if str_var == "u":
            str_p = r'$\hat{u}$'
        elif str_var == "v":
            str_p = r'$\hat{v}$'
        elif str_var == "w":
            str_p = r'$\hat{w}$'
        elif str_var == "p":
            str_p = r'$\hat{p}$'
        elif str_var == "r":
            str_p = r'$\hat{\rho}$'
        else:
            print("Not an expected string!!!!!")

        str_r = str_p + r'$_r$'
        str_i = str_p + r'$_i$'

        return str_r, str_i
