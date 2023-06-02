import os
import sys
import warnings
try:
    import cupy as cp
except ImportError:
    warnings.warn('Failed to import cupy')
import numpy as np
import scipy as sp
import scipy.sparse.linalg
from scipy import linalg
import module_utilities as mod_util
import build_matrices as mbm
#import scipy as sp
import matplotlib.pyplot as plt

hascupy = 'cupy' in sys.modules

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

        self.Local = self.check_params_setup_loops()

        #print("self.Local = ", self.Local)
        #print("self.Fr = ", self.Fr)
        #print("self.Sc = ", self.Sc)
        
        if (self.Local):
            self.solve_secant_problem(omega_guess)
        else:
            self.eigvals_filtered = self.solve_general_problem(self.alpha, self.beta, omega_guess)

    def solve_secant_problem(self, omega):
        """
        Function...
        """
        q_eigvect = np.zeros(np.size(self.vec_rhs), dpc)

        if   ( self.niter == 2 ):
            self.iterate_both(omega)
        elif ( self.niter == 1):
            ivar = 0
            self.iterate_single(ivar, omega)
            
    def call_to_build_matrices(*args, **kwargs):
        raise NotImplemented("Use a specific equation class, not SolveGeneralizedEVP.")
                
    def solve_general_problem(self, alpha, beta, omega_guess):
        self.call_to_build_matrices(Re_in=self.Re, Fr_in=self.Fr, Sc_in=self.Sc)
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

    def solve_stability_secant(self, omega0, omega00, alpha_in, beta_in, re_in, fr_in, sc_in):
        """
        Function of class SolveGeneralizedEVP that solves locally for a single eigenvalue at a time (a guess is required)
        """
        mid_idx = self.mid_idx
        
        if self.bsfl.rt_flag == True:
            tol1  = 1.0e-5
            tol2  = 1.0e-5
            #print("tol1, tol2 = ", tol1, tol2)
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
        
        #jxu  = 5*self.ny-1 # 0
        #if (self.bsfl.rt_flag): jxu = 3*self.ny-1 # use 3*ny-1 for w and 5*ny-1 for rho

        #
        # Compute u00
        #
        
        # Build main stability matrices and assemble them
        self.call_to_build_matrices(Re_in=re_in, Fr_in=fr_in, Sc_in=sc_in)
        self.assemble_mat_lhs(alpha_in, beta_in, omega00)
        self.call_to_set_bc_secant(self.mat_lhs, self.vec_rhs, self.ny, self.map, alpha_in, beta_in)
            
        # Solve Linear System
        if (hascupy):
            SOL = cp.linalg.solve(self.mat_lhs, self.vec_rhs)
        else:
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
            if (hascupy):
                SOL = cp.linalg.solve(self.mat_lhs, self.vec_rhs)
            else:
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
            
            print("     Iteration %2d: abs(u0) = %10.5e, abs(res) = %10.5e, omega = %21.11e, %21.11e" % (iter, np.abs(u0), abs(res), omega0.real, omega0.imag))
            
        if ( iter == maxiter ): sys.exit("No Convergence in Secant Method...")

        # Get final eigenvectors
        final_ev = 0
        if (final_ev==1):
            self.assemble_mat_lhs(alpha_in, beta_in, omega0)
            self.call_to_set_bc_secant(self.mat_lhs, self.vec_rhs, self.ny, self.map, alpha_in, beta_in)
            if (hascupy):
                self.qfinal = cp.linalg.solve(self.mat_lhs, self.vec_rhs)
            else:
                self.qfinal = linalg.solve(self.mat_lhs, self.vec_rhs)
        else:
            self.qfinal = q

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

    def write_stab_banana(self):

        if ( self.Local ):
            filename      = "Stability_banana.dat"
            work_dir      = "./"
            save_path     = work_dir #+ '/' + folder1
            completeName  = os.path.join(save_path, filename)
            fileoutFinal  = open(completeName,'w')

            if ( np.size(self.npts_list) == 2 ):
                npts1 = self.npts_list[0]
                npts2 = self.npts_list[1]

                Xvar = self.list_iter[0]
                Yvar = self.list_iter[1]
                flag2 = True
            else:
                npts1 = self.npts_list[0]
                npts2 = 1

                Xvar = self.list_iter[0]
                flag2 = False


            #print('VARIABLES = "%s" "%s" "omega_r" "omega_i"\n' % (Xvar, Xvar))
            
            #zname2D       = 'ZONE T="2-D Zone", I = ' + str(npts1) + ', J = ' + str(npts2) + '\n'
            zname2D       = 'ZONE T="2-D Zone", I = ' + str(npts2) + ', J = ' + str(npts1) + '\n'
            fileoutFinal.write('TITLE     = "2D DATA"\n')
            if (flag2):
                fileoutFinal.write('VARIABLES = "%s" "%s" "omega_r" "omega_i"\n' % (Yvar, Xvar))
            else:
                fileoutFinal.write('VARIABLES = "%s" "omega_r" "omega_i"\n' % Xvar)
                
            fileoutFinal.write(zname2D)
            
            # for j in range(0, npts2):
            #     for i in range(0, npts1):
            #         if (flag2):
            #             fileoutFinal.write("%25.15e %25.15e %25.15e %25.15e \n" % ( self.vars_dict[Xvar][i], self.vars_dict[Yvar][j], self.omega_array[j,i].real, self.omega_array[j,i].imag ) )
            #         else:
            #             fileoutFinal.write("%25.15e %25.15e %25.15e \n" % ( self.vars_dict[Xvar][i], self.omega_array[j,i].real, self.omega_array[j,i].imag ) )

            for i in range(0, npts1):
                for j in range(0, npts2):
                    if (flag2):
                        fileoutFinal.write("%25.15e %25.15e %25.15e %25.15e \n" % ( self.vars_dict[Yvar][j], self.vars_dict[Xvar][i], self.omega_array[j,i].real, self.omega_array[j,i].imag ) )
                    else:
                        fileoutFinal.write("%25.15e %25.15e %25.15e \n" % ( self.vars_dict[Xvar][i], self.omega_array[j,i].real, self.omega_array[j,i].imag ) )

            fileoutFinal.close()

            
        else:
            pass


    def check_params_setup_loops(self):    

        Local = False
        self.niter = 0

        #hasSc = hasattr(self, "Sc")
        #hasFr = hasattr(self, "Fr")

        self.mylist = []
        self.npts_list = []
        self.list_iter = []
        self.vars_dict = {}
        
        self.alpha = self.check_size_reset(self.alpha)
        self.beta = self.check_size_reset(self.beta)
        self.Re = self.check_size_reset(self.Re)

        self.Fr = self.check_size_reset(self.Fr)
        self.Sc = self.check_size_reset(self.Sc)

        #if ( hasFr ): self.Fr = self.check_size_reset(self.Fr)
        #if ( hasSc ): self.Sc = self.check_size_reset(self.Sc)

        self.mylist.append("alpha")
        self.vars_dict["alpha"] = self.alpha

        self.mylist.append("beta")
        self.vars_dict["beta"] = self.beta

        self.mylist.append("Re")
        self.vars_dict["Re"] = self.Re

        #if (hasSc):
        self.mylist.append("Sc")
        self.vars_dict["Sc"] = self.Sc

        #if (hasFr):
        self.mylist.append("Fr")
        self.vars_dict["Fr"] = self.Fr

        if ( np.size(self.alpha) != 1 ):
            Local = True
            self.npts_list.append(np.size(self.alpha))
            self.niter = self.niter + 1
            self.list_iter.append("alpha")

        if ( np.size(self.beta) != 1 ):
            sys.exit("consider only a single beta for now")
            Local = True
            self.npts_list.append(np.size(self.beta))
            self.niter = self.niter + 1
            self.list_iter.append("beta")

        if ( np.size(self.Re) != 1 ):
            Local = True
            self.npts_list.append(np.size(self.Re))
            self.niter = self.niter + 1
            self.list_iter.append("Re")

        #if (hasSc):
        if ( np.size(self.Sc) != 1 ):
            Local = True
            self.npts_list.append(np.size(self.Sc))
            self.niter = self.niter + 1
            self.list_iter.append("Sc")

        #if (hasFr):
        if ( np.size(self.Fr) != 1 ):
            Local = True
            self.npts_list.append(np.size(self.Fr))
            self.niter = self.niter + 1
            self.list_iter.append("Fr")

        if ( self.niter > 2 ):
            sys.exit("Maximum number of parameters is 2")

        for i in range(0, len(self.npts_list)):
            print("self.npts_list[i] = ", self.npts_list[i])
            print("self.list_iter[i] = ", self.list_iter[i])

        if (Local):
            if ( np.size(self.npts_list) == 2 ):
                self.omega_array = np.zeros((self.npts_list[1], self.npts_list[0]), dpc)
            else:
                self.omega_array = np.zeros((1, self.npts_list[0]), dpc)

            # You need to use copy other reference to same variable
            self.vars_dict_loc = self.vars_dict.copy()

            
        return Local

    def check_size_reset(self, var_in):
        
        if ( np.size(var_in) != 1 ):
            if ( var_in[0] == var_in[-1] ):
                var_out = np.linspace(var_in[0], var_in[0], 1)

                return var_out
            else:
                return var_in

        else:
            var_out = np.linspace(var_in, var_in, 1)
            return var_out


    def iterate_both(self, omega):

        self.outer_var = self.list_iter[1]
        #print("self.outer_var = ", self.outer_var)

        for i in range(0, self.npts_list[1]):

            self.outer_var_val_loc = self.vars_dict[self.outer_var][i]
            self.vars_dict_loc[self.outer_var] = np.linspace(self.outer_var_val_loc, self.outer_var_val_loc, 1)

            #print("self.vars_dict_loc = ", self.vars_dict_loc)

            print("")
            print("Solving for %s = %10.5e (%i/%i)" % (self.outer_var, self.outer_var_val_loc, i, self.npts_list[1]-1 ))
            print("=====================================")
            print("self.jxu = ", self.jxu)
            print("")

            self.iterate_single(i, omega)

    def iterate_single(self, ivar, omega):

        for i in range(0, self.npts_list[0]):

            idx_a = min(i, np.size(self.vars_dict_loc[self.mylist[0]])-1 )
            idx_b = min(i, np.size(self.vars_dict_loc[self.mylist[1]])-1 )
            idx_re = min(i, np.size(self.vars_dict_loc[self.mylist[2]])-1 )
            idx_sc = min(i, np.size(self.vars_dict_loc[self.mylist[3]])-1 )
            idx_fr = min(i, np.size(self.vars_dict_loc[self.mylist[4]])-1 )

            alpha_in = self.vars_dict_loc[self.mylist[0]][idx_a]
            beta_in = self.vars_dict_loc[self.mylist[1]][idx_b]
            re_in = self.vars_dict_loc[self.mylist[2]][idx_re]
            sc_in = self.vars_dict_loc[self.mylist[3]][idx_sc]
            fr_in = self.vars_dict_loc[self.mylist[4]][idx_fr]

            #print("idx_a, idx_b, idx_re, idx_sc, idx_fr = ", idx_a, idx_b, idx_re, idx_sc, idx_fr)
            #print("alpha_in, beta_in, re_in, fr_in, sc_in = ", alpha_in, beta_in, re_in, sc_in, fr_in)

            print("")
            print("     Solving for %s = %10.5e (%i/%i)" % (self.list_iter[0], self.vars_dict[self.list_iter[0]][i], i, self.npts_list[0]-1 ))
            print("     -------------------------------------")
            
            # Extrapolate omega
            if (i >= 1):
                # Extrapolation for inner loop
                omega = mod_util.extrapolate_in_inner_loop(self.omega_array, self.vars_dict[self.list_iter[0]], i, ivar)
            elif (ivar >= 1):
                # Extrapolation for outer loop
                omega = mod_util.extrapolate_in_outer_loop(self.omega_array, self.vars_dict[self.list_iter[1]], i, ivar)

            #print("omega = ", omega)
            self.omega_array[ivar, i] = self.solve_stability_secant(omega, omega + omega*1e-5, alpha_in, beta_in, re_in, fr_in, sc_in )









            
            #print("self.mylist[0] = ", self.mylist[0])
            #print("self.vars_dict[self.mylist[0]][i] = ", self.vars_dict[self.mylist[0]][i])
            #print("i = ", i)
            #print("self.npts_list[0] = ", self.npts_list[0])

            # self.list_iter[i]

            #print("alpha = ",self.vars_dict["alpha"][])
            #print("self.vars_dict[self.mylist[0]] = ", self.vars_dict[self.mylist[0]])

        #list_restrict = self.mylist
        #print("list_restrict = ", list_restrict)
        #list_restrict.remove(self.list_iter[1])

        #idx_max_a = np.min(np.size(self.vars_dict["alpha"]))
        #idx_max_b = np.min(np.size(self.vars_dict["beta"]))
        #idx_max_Re = np.min(np.size(self.vars_dict["Re"]))


        #print("list_restrict = ", list_restrict)
        #input("check here")

        #print("self.mylist = ", self.mylist)
        #print("self.vars_dict[self.mylist[0]][0] = ", self.vars_dict[self.mylist[0]][0])
        #print("self.vars_dict[self.mylist[1]] = ", self.vars_dict[self.mylist[1]])
        #print("self.vars_dict = ", self.vars_dict)
