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
        elif ( self.niter == 1 ):
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

        tol_nrm = 1.e6
        
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
        maxiter = 20
        
        #jxu  = 5*self.ny-1 # 0
        #if (self.bsfl.rt_flag): jxu = 3*self.ny-1 # use 3*ny-1 for w and 5*ny-1 for rho

        #
        # Compute u00
        #
        #print("self.jxu = ", self.jxu)
        #print("re_in, fr_in, sc_in = ", re_in, fr_in, sc_in )
        
        # Build main stability matrices and assemble them
        self.call_to_build_matrices(Re_in=re_in, Fr_in=fr_in, Sc_in=sc_in)
        self.assemble_mat_lhs(alpha_in, beta_in, omega00)
        self.call_to_set_bc_secant(self.mat_lhs, self.vec_rhs, self.ny, self.map, alpha_in, beta_in)
            
        # Solve Linear System
        if (hascupy):
            SOL = cp.linalg.solve(self.mat_lhs, self.vec_rhs)
        else:
            SOL = linalg.solve(self.mat_lhs, self.vec_rhs)
            #print("np.linalg.norm(SOL) = ", np.linalg.norm(SOL))
                
        u00 = SOL[self.jxu]

        res = 1
        
        #
        # Main loop
        #
        
        while ( ( abs(u0) > tol2 or abs(res) > tol1 ) and iter < maxiter and np.linalg.norm(SOL) < tol_nrm ):
        
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
                #print("np.linalg.norm(SOL) = ", np.linalg.norm(SOL))
        
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

        if ( iter == maxiter or np.linalg.norm(SOL) > tol_nrm ):
            return 0.0, iter, maxiter, np.linalg.norm(SOL), tol_nrm
            #sys.exit("No Convergence in Secant Method...")
        else:
            return omega0, iter, maxiter, np.linalg.norm(SOL), tol_nrm
        
    def write_eigvals(self):
        if (self.Local):
            warnings.warn("Ignoring in write_eigvals: Writing eigenvalues (alpha_i vs alpha_r) does not make sense for Local solution")
        else:
            mod_util.write_out_eigenvalues(self.EigVal, self.ny)

    #def write_eigvects_out_new(self, y, ueig, veig, weig, peig, reig, Local, bsfl_ref):
    def write_eigvects_out_new(self):

        y = self.map.y
        
        ueig = self.ueig
        veig = self.veig
        weig = self.weig
        peig = self.peig
        reig = self.reig
        
        ny = len(y)
        # variables are u, v, w, p
        # data_out = np.column_stack([ y, np.abs(q[idx_u]), np.abs(q[idx_v]), np.abs(q[idx_w]), np.abs(q[idx_p]), \
        #                              q[idx_u].real, q[idx_v].real, q[idx_w].real, q[idx_p].real, \
        #                              q[idx_u].imag, q[idx_v].imag, q[idx_w].imag, q[idx_p].imag                  ])

        data_out = np.column_stack([ y, np.abs(ueig), np.abs(veig), np.abs(weig), np.abs(peig), np.abs(reig), \
                                     ueig.real, veig.real, weig.real, peig.real, reig.real, \
                                     ueig.imag, veig.imag, weig.imag, peig.imag, reig.imag                  ])

        if (self.Local):
            datafile_path = "./Eigenfunctions_Local_Solution_nondimensional_" + str(ny) + "_pts.txt"
        else:
            datafile_path = "./Eigenfunctions_Global_Solution_nondimensional_" + str(ny) + "_pts.txt"
            
        np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e',\
                                                  '%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])


        # # Now write out dimensional eigenfunctions
        # Lref = bsfl_ref.Lref
        # Uref = bsfl_ref.Uref
        # rhoref = bsfl_ref.rhoref

        # ueig = ueig*Uref
        # veig = veig*Uref
        # weig = weig*Uref
        # peig = peig*rhoref*Uref**2.
        # reig = reig*rhoref

        # ydim = y*Lref
    
        # data_out = np.column_stack([ ydim, np.abs(ueig), np.abs(veig), np.abs(weig), np.abs(peig), np.abs(reig), \
        #                              ueig.real, veig.real, weig.real, peig.real, reig.real, \
        #                              ueig.imag, veig.imag, weig.imag, peig.imag, reig.imag                  ])


        # if (self.Local):
        #     datafile_path = "./Eigenfunctions_Local_Solution_dimensional_" + str(ny) + "_pts.txt"
        # else:
        #     datafile_path = "./Eigenfunctions_Global_Solution_dimensional_" + str(ny) + "_pts.txt"
            
        # np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e',\
        #                                           '%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])


    #def write_baseflow_out(self, ynondim, ydim, Rho, Rhop, Rhopp, W, Wp, Rho_dim, Rhop_dim, Rhopp_dim, W_dim, Wp_dim):
    def write_baseflow_out(self, solver):

        y = solver.map.y
        ny = len(y)
        
        # Rho, Rhop, Rhopp,  W and Wp are non-dimensional
        
        #data_out = np.column_stack([ ynondim, ydim, Rho, Rhop, Rhopp, W, Wp, Rho_dim, Rhop_dim, Rhopp_dim, W_dim, Wp_dim ])
        data_out = np.column_stack([ y, y+12.5, solver.bsfl.U, solver.bsfl.Up,
                                     solver.bsfl.W, solver.bsfl.Wp,
                                     solver.bsfl.Rho_nd, solver.bsfl.Rho_nd-1.0,
                                     solver.bsfl.Rhop_nd, solver.bsfl.Rhopp_nd])
        datafile_path = "./Baseflow_" + str(ny) + "_pts.txt"
        
        #np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e',\
        #                                          '%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])
        np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e',\
                                                  '%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])
        
        print("")
        print("Baseflow has been written out")
        print("")

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

            plt.gcf().subplots_adjust(left=0.145)
            plt.gcf().subplots_adjust(bottom=0.145)

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

        plt.gcf().subplots_adjust(left=0.145)
        plt.gcf().subplots_adjust(bottom=0.145)

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

        plt.gcf().subplots_adjust(left=0.145)
        plt.gcf().subplots_adjust(bottom=0.145)

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

        #print(" hasattr(self.bsfl, Rho_nd_2d) = ", hasattr(self.bsfl, "Rho_nd_2d"))

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

        if ( hasattr(self.bsfl, "Rho_nd_2d") ):
            self.mylist.append("Time")
            self.vars_dict["Time"] = self.bsfl.time_out

        if ( np.size(self.alpha) != 1 ):
            Local = True
            self.npts_list.append(np.size(self.alpha))
            self.niter = self.niter + 1
            self.list_iter.append("alpha")

        if ( np.size(self.beta) != 1 ):
            #sys.exit("consider only a single beta for now")
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

        if ( hasattr(self.bsfl, "Rho_nd_2d") ):
            if ( self.bsfl.Rho_nd_2d.ndim == 2 ):
                print("")
                print("Baseflow has multiple timesteps (from PsDNS)")
                print("")
                if ( self.niter != 0 ):
                    sys.exit("Not working for time-variable baseflows!!!!!")

                Local = True
                nt = self.bsfl.Rho_nd_2d.shape[0]
                self.npts_list.append(nt)
                self.niter = self.niter + 1
                self.list_iter.append("Time")

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

            if ( hasattr(self.bsfl, "Rho_nd_2d") == True ):
                self.bsfl.Rho_nd = self.bsfl.Rho_nd_2d[i, :]
                self.bsfl.Rhop_nd = self.bsfl.Rhop_nd_2d[i, :]
                self.bsfl.Rhopp_nd = self.bsfl.Rhopp_nd_2d[i, :]

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
            self.omega_array[ivar, i], iter, maxiter, sol_nrm, tol_nrm = self.solve_stability_secant(omega, omega + omega*1e-5, alpha_in, beta_in, re_in, fr_in, sc_in )
            if ( iter == maxiter or sol_nrm > tol_nrm ):
                break


class PostProcess(SolveGeneralizedEVP):
    """
    This class define a post-processing object
    """
    def __init__(self, solver):
        """
        Constructor of class PostProcess
        """
        #super(PostProcess, self).__init__(*args, **kwargs)
        self.solver = solver

    def compute_inner_prod(self, f, g):
        inner_prod = 0.5*( np.multiply(np.conj(f), g) + np.multiply(f, np.conj(g)) )    
        return inner_prod

    def get_dist_prods_lhs(self):

        Re = self.solver.Re
        
        self.rxx = self.compute_inner_prod(self.solver.ueig, self.solver.ueig)
        self.ryy = self.compute_inner_prod(self.solver.veig, self.solver.veig)
        self.rzz = self.compute_inner_prod(self.solver.weig, self.solver.weig)
        
        self.rxy = self.compute_inner_prod(self.solver.ueig, self.solver.veig)
        self.rxz = self.compute_inner_prod(self.solver.ueig, self.solver.weig)
        self.ryz = self.compute_inner_prod(self.solver.veig, self.solver.weig)

        self.drxxdx = 2.*self.compute_inner_prod(self.solver.ueig, self.dudx)
        self.drxxdy = 2.*self.compute_inner_prod(self.solver.ueig, self.dudy)
        self.drxxdz = 2.*self.compute_inner_prod(self.solver.ueig, self.dudz)
        #
        self.d2rxxdxdy = 2.*( self.compute_inner_prod(self.dudx, self.dudy)
                              +self.compute_inner_prod(self.solver.ueig, self.d2udxdy)
                             ) 

        self.dryydx = 2.*self.compute_inner_prod(self.solver.veig, self.dvdx)
        self.dryydy = 2.*self.compute_inner_prod(self.solver.veig, self.dvdy)
        self.dryydz = 2.*self.compute_inner_prod(self.solver.veig, self.dvdz)
        #
        self.d2ryydy2 = 2.*( self.compute_inner_prod(self.dvdy, self.dvdy)
                             +self.compute_inner_prod(self.solver.veig, self.d2vdy2)
                            ) 

        self.drzzdx = 2.*self.compute_inner_prod(self.solver.weig, self.dwdx)
        self.drzzdy = 2.*self.compute_inner_prod(self.solver.weig, self.dwdy)
        self.drzzdz = 2.*self.compute_inner_prod(self.solver.weig, self.dwdz)

        self.drxydx = self.compute_inner_prod(self.solver.ueig, self.dvdx) + self.compute_inner_prod(self.solver.veig, self.dudx)
        self.drxydy = self.compute_inner_prod(self.solver.ueig, self.dvdy) + self.compute_inner_prod(self.solver.veig, self.dudy)
        self.drxydz = self.compute_inner_prod(self.solver.ueig, self.dvdz) + self.compute_inner_prod(self.solver.veig, self.dudz)
        #
        self.d2rxydx2 = ( 2.*self.compute_inner_prod(self.dudx, self.dvdx)
                          +self.compute_inner_prod(self.solver.ueig, self.d2vdx2)
                          +self.compute_inner_prod(self.solver.veig, self.d2udx2)
                         ) 
        #
        self.d2rxydy2 = ( 2.*self.compute_inner_prod(self.dudy, self.dvdy)
                          +self.compute_inner_prod(self.solver.ueig, self.d2vdy2)
                          +self.compute_inner_prod(self.solver.veig, self.d2udy2)
                         ) 
        #
        self.d2rxydxdy = ( self.compute_inner_prod(self.dudy, self.dvdx)
                           +self.compute_inner_prod(self.solver.ueig, self.d2vdxdy)
                           +self.compute_inner_prod(self.dudx, self.dvdy)
                           +self.compute_inner_prod(self.solver.veig, self.d2udxdy)
                         ) 


        # Build left-hand-sides for Reynolds stress equations
        self.rxx_lhs = 2.*self.omega_i*self.rxx.real
        self.ryy_lhs = 2.*self.omega_i*self.ryy.real
        self.rzz_lhs = 2.*self.omega_i*self.rzz.real
        self.rxy_lhs = 2.*self.omega_i*self.rxy.real

        # Rxx equation
        self.rxx_lap = 2.*( self.compute_inner_prod(self.dudx, self.dudx)
                            +self.compute_inner_prod(self.dudy, self.dudy)
                            +self.compute_inner_prod(self.dudz, self.dudz)
                           )/Re
        
        self.rxx_lap = self.rxx_lap + 2.*( self.compute_inner_prod(self.solver.ueig, self.d2udx2)
                                           +self.compute_inner_prod(self.solver.ueig, self.d2udy2)
                                           +self.compute_inner_prod(self.solver.ueig, self.d2udz2)
                                          )/Re
        
        self.rxx_dissip = 2.*( self.compute_inner_prod(self.dudx, self.dudx)
                               +self.compute_inner_prod(self.dudy, self.dudy)
                               +self.compute_inner_prod(self.dudz, self.dudz)
                              )/Re

        self.rxx_advection = -np.multiply(self.solver.bsfl.U, self.drxxdx) -np.multiply(self.solver.bsfl.W, self.drxxdz)
        self.rxx_production = -2.*np.multiply(self.rxy, self.solver.bsfl.Up)

        self.rxx_pres_diffus = -2.* ( self.compute_inner_prod(self.solver.ueig, self.dpdx) + self.compute_inner_prod(self.solver.peig, self.dudx) )
        self.rxx_pres_strain = 2.*self.compute_inner_prod(self.solver.peig, self.dudx)

        # Ryy equation
        self.ryy_lap = 2.*( self.compute_inner_prod(self.dvdx, self.dvdx)
                            +self.compute_inner_prod(self.dvdy, self.dvdy)
                            +self.compute_inner_prod(self.dvdz, self.dvdz)
                           )/Re
        
        self.ryy_lap = self.ryy_lap + 2.*( self.compute_inner_prod(self.solver.veig, self.d2vdx2)
                                           +self.compute_inner_prod(self.solver.veig, self.d2vdy2)
                                           +self.compute_inner_prod(self.solver.veig, self.d2vdz2)
                                          )/Re
        
        self.ryy_dissip = 2.*( self.compute_inner_prod(self.dvdx, self.dvdx)
                               +self.compute_inner_prod(self.dvdy, self.dvdy)
                               +self.compute_inner_prod(self.dvdz, self.dvdz)
                              )/Re

        self.ryy_advection = -np.multiply(self.solver.bsfl.U, self.dryydx) -np.multiply(self.solver.bsfl.W, self.dryydz)
        self.ryy_production = 0.0*self.ryy_advection

        self.ryy_pres_diffus = -2.*( self.compute_inner_prod(self.solver.veig, self.dpdy) + self.compute_inner_prod(self.solver.peig, self.dvdy) )
        self.ryy_pres_strain = 2.*self.compute_inner_prod(self.solver.peig, self.dvdy)

        # Rxy equation
        self.rxy_lap = 2.*( self.compute_inner_prod(self.dudx, self.dvdx)
                            +self.compute_inner_prod(self.dudy, self.dvdy)
                            +self.compute_inner_prod(self.dudz, self.dvdz)
                           )/Re
        
        self.rxy_lap = self.rxy_lap + ( self.compute_inner_prod(self.solver.ueig, self.d2vdx2)
                                        +self.compute_inner_prod(self.solver.ueig, self.d2vdy2)
                                        +self.compute_inner_prod(self.solver.ueig, self.d2vdz2)
                                       )/Re

        self.rxy_lap = self.rxy_lap + ( self.compute_inner_prod(self.solver.veig, self.d2udx2)
                                        +self.compute_inner_prod(self.solver.veig, self.d2udy2)
                                        +self.compute_inner_prod(self.solver.veig, self.d2udz2)
                                       )/Re

        self.rxy_dissip = 2.*( self.compute_inner_prod(self.dudx, self.dvdx)
                               +self.compute_inner_prod(self.dudy, self.dvdy)
                               +self.compute_inner_prod(self.dudz, self.dvdz)
                              )/Re

        self.rxy_advection = -np.multiply(self.solver.bsfl.U, self.drxydx) -np.multiply(self.solver.bsfl.W, self.drxydz)
        self.rxy_production = -np.multiply(self.ryy, self.solver.bsfl.Up)

        self.rxy_pres_diffus = -( self.compute_inner_prod(self.solver.ueig, self.dpdy)
                               +self.compute_inner_prod(self.solver.peig, self.dudy)
                               +self.compute_inner_prod(self.solver.veig, self.dpdx)
                               +self.compute_inner_prod(self.solver.peig, self.dvdx)
                              )
    
        
        self.rxy_pres_strain = ( self.compute_inner_prod(self.solver.peig, self.dudy)
                                 +self.compute_inner_prod(self.solver.peig, self.dvdx)
                                )
        
        # Rzz equation
        self.rzz_lap = 2.*( self.compute_inner_prod(self.dwdx, self.dwdx)
                            +self.compute_inner_prod(self.dwdy, self.dwdy)
                            +self.compute_inner_prod(self.dwdz, self.dwdz)
                           )/Re
        
        self.rzz_lap = self.rzz_lap + 2.*( self.compute_inner_prod(self.solver.weig, self.d2wdx2)
                                           +self.compute_inner_prod(self.solver.weig, self.d2wdy2)
                                           +self.compute_inner_prod(self.solver.weig, self.d2wdz2)
                                          )/Re
        
        self.rzz_dissip = 2.*( self.compute_inner_prod(self.dwdx, self.dwdx)
                               +self.compute_inner_prod(self.dwdy, self.dwdy)
                               +self.compute_inner_prod(self.dwdz, self.dwdz)
                              )/Re

        self.rzz_advection = -np.multiply(self.solver.bsfl.U, self.drzzdx) -np.multiply(self.solver.bsfl.W, self.drzzdz)
        self.rzz_production = -2.*np.multiply(self.ryz, self.solver.bsfl.Wp)

        self.rzz_pres_diffus = -2.*( self.compute_inner_prod(self.solver.weig, self.dpdz) + self.compute_inner_prod(self.solver.peig, self.dwdz) )
        self.rzz_pres_strain = 2.*self.compute_inner_prod(self.solver.peig, self.dwdz)
        
class PostSingleFluid(PostProcess):
    def __init__(self, solver):
        """
        Constructor of class PostSingleFluid
        """
        super(PostSingleFluid, self).__init__(solver)

    def get_eigenfcts_derivatives(self):

        D1 = self.solver.map.D1
        D2 = self.solver.map.D2
        
        ia = 1j*self.solver.alpha[0]
        ib = 1j*self.solver.beta[0]

        ia2 = ( ia )**2.
        ib2 = ( ib )**2.

        iab = ia*ib

        self.dudx = ia*self.solver.ueig
        self.dudy = np.matmul(D1, self.solver.ueig)
        self.dudz = ib*self.solver.ueig

        #print("self.dudx.shape, self.dudy.shape, self.dudz.shape = ", self.dudx.shape, self.dudy.shape, self.dudz.shape)
        
        self.dvdx = ia*self.solver.veig
        self.dvdy = np.matmul(D1, self.solver.veig)
        self.dvdz = ib*self.solver.veig
        
        self.dwdx = ia*self.solver.weig
        self.dwdy = np.matmul(D1, self.solver.weig)
        self.dwdz = ib*self.solver.weig
        
        self.dpdx = ia*self.solver.peig
        self.dpdy = np.matmul(D1, self.solver.peig)
        self.dpdz = ib*self.solver.peig
        
        self.drdx = ia*self.solver.reig
        self.drdy = np.matmul(D1, self.solver.reig)
        self.drdz = ib*self.solver.reig
        
        self.d2udx2 = ia2*self.solver.ueig
        self.d2udy2 = np.matmul(D2, self.solver.ueig)
        self.d2udz2 = ib2*self.solver.ueig
        self.d2udxdy = ia*np.matmul(D1, self.solver.ueig)
                
        self.d2vdx2 = ia2*self.solver.veig
        self.d2vdy2 = np.matmul(D2, self.solver.veig)
        self.d2vdz2 = ib2*self.solver.veig
        self.d2vdxdy = ia*np.matmul(D1, self.solver.veig)
        
        self.d2wdx2 = ia2*self.solver.weig
        self.d2wdy2 = np.matmul(D2, self.solver.weig)
        self.d2wdz2 = ib2*self.solver.weig
        
        self.d2pdx2 = ia2*self.solver.peig
        self.d2pdy2 = np.matmul(D2, self.solver.peig)
        self.d2pdz2 = ib2*self.solver.peig
        
        self.d2rdx2 = ia2*self.solver.reig
        self.d2rdy2 = np.matmul(D2, self.solver.reig)
        self.d2rdz2 = ib2*self.solver.reig

    def get_balance_reynolds_stress(self, omega_i):

        self.omega_i = omega_i
        
        self.get_eigenfcts_derivatives()
        self.get_dist_prods_lhs()
                
        # Build rhs for Rxx equation                                         
        self.rxx_rhs = ( self.rxx_advection
                         +self.rxx_production
                         +self.rxx_pres_diffus
                         +self.rxx_pres_strain
                         +self.rxx_lap
                         -self.rxx_dissip
                        )

        self.rxx_rhs = self.rxx_rhs.real

        # Build rhs for Ryy equation
        self.ryy_rhs = ( self.ryy_advection
                         +self.ryy_pres_diffus
                         +self.ryy_pres_strain
                         +self.ryy_lap
                         -self.ryy_dissip
                        )

        self.ryy_rhs = self.ryy_rhs.real

        # Build rhs for Rzz equation        
        self.rzz_rhs = ( self.rzz_advection
                         +self.rzz_production
                         +self.rzz_pres_diffus
                         +self.rzz_pres_strain
                         +self.rzz_lap
                         -self.rzz_dissip
                        )

        self.rzz_rhs = self.rzz_rhs.real

        # Build rhs for Rxy equation                 
        self.rxy_rhs = ( self.rxy_advection
                         +self.rxy_production
                         +self.rxy_pres_diffus
                         +self.rxy_pres_strain
                         +self.rxy_lap
                         -self.rxy_dissip
                        )

        self.rxy_rhs = self.rxy_rhs.real
        
        print("")
        print("Balance Rxx equation:")
        print("=====================")
        print("rxx_bal = ", np.amax(np.abs(self.rxx_lhs-self.rxx_rhs)))
        
        print("Balance Ryy equation:")
        print("=====================")
        print("ryy_bal = ", np.amax(np.abs(self.ryy_lhs-self.ryy_rhs)))
        
        print("Balance Rzz equation:")
        print("=====================")
        print("rzz_bal = ", np.amax(np.abs(self.rzz_lhs-self.rzz_rhs)))

        print("Balance Rxy equation:")
        print("=====================")
        print("rxy_bal = ", np.amax(np.abs(self.rxy_lhs-self.rxy_rhs)))
        print("")

        self.get_model_terms()

        self.plot_balance_reynolds_stress(self.rxx_lhs,
                                          self.rxx_rhs,
                                          self.rxx_advection,
                                          self.rxx_production,
                                          self.rxx_pres_diffus,
                                          self.rxx_pres_strain,
                                          self.rxx_lap,
                                          self.rxx_dissip,
                                          self.rxx_pres_strain_model,
                                          self.solver.map.y,
                                          '$R_{xx}$ balance'
                          )
        self.plot_balance_reynolds_stress(self.ryy_lhs,
                                          self.ryy_rhs,
                                          self.ryy_advection,
                                          self.ryy_production,
                                          self.ryy_pres_diffus,
                                          self.ryy_pres_strain,
                                          self.ryy_lap,
                                          self.ryy_dissip,
                                          self.ryy_pres_strain_model,
                                          self.solver.map.y,
                                          '$R_{yy}$ balance'
                          )
        self.plot_balance_reynolds_stress(self.rzz_lhs,
                                          self.rzz_rhs,
                                          self.rzz_advection,
                                          self.rzz_production,
                                          self.rzz_pres_diffus,
                                          self.rzz_pres_strain,
                                          self.rzz_lap,
                                          self.rzz_dissip,
                                          0.0*self.rxx_pres_strain_model,
                                          self.solver.map.y,
                                          '$R_{zz}$ balance'
                                          )
        self.plot_balance_reynolds_stress(self.rxy_lhs,
                                          self.rxy_rhs,
                                          self.rxy_advection,
                                          self.rxy_production,
                                          self.rxy_pres_diffus,
                                          self.rxy_pres_strain,
                                          self.rxy_lap,
                                          self.rxy_dissip,
                                          self.rxy_pres_strain_model,
                                          self.solver.map.y,
                                          '$R_{xy}$ balance'
                                          )
        
        self.plot_reynolds_stress(self.rxx,
                                  self.ryy,
                                  self.rzz,
                                  self.rxy,
                                  self.rxz,
                                  self.ryz,
                                  self.solver.map.y,
                                  'Reynolds stresses'
                          )

        #print("self.rxx_advection = ", self.rxx_advection)

    def plot_balance_reynolds_stress(self, lhs, rhs, advection, production, pres_diffus, pres_strain, laplacian, dissipation, pres_strain_model, y, str_bal):

        # Plot
        ax = plt.gca()
        ax.plot( lhs.real,
                 y,
                 'k',
                 label=r"Lhs",
                 linewidth=1.5 )
        #
        ax.plot( rhs.real,
                 y,
                 'g--',
                 label=r"Rhs",
                 linewidth=1.5 )
        #
        ax.plot( advection.real,
                 y,
                 'r',
                 label=r"Advection",
                 linewidth=1.5 )

        #
        ax.plot( production.real,
                 y,
                 'b',
                 label=r"Production",
                 linewidth=1.5 )
        #
        ax.plot( pres_diffus.real,
                 y,
                 'y-.',
                 label=r"Pres. diffusion",
                 linewidth=1.5 )
        #
        ax.plot( pres_strain.real,
                 y,
                 'm-',
                 label=r"Pres. strain",
                 linewidth=1.5 )
        #
        ax.plot( laplacian.real,
                 y,
                 'c',
                 label=r"Laplacian",
                 linewidth=1.5 )
        #
        ax.plot( dissipation.real,
                 y,
                 'orange',
                 label=r"Dissipation",
                 linewidth=1.5 )
        #
        ax.plot( pres_strain_model.real,
                 y,
                 'm-.',
                 #color='grey',
                 markevery=3,
                 markerfacecolor='none',
                 label=r"Pres. strain model",
                 linewidth=1.5 )

        title_lab = "%s" % str_bal #str('{:.2e}'.format(scale_pd))
        ax.set_xlabel(title_lab, fontsize=20)
        ax.set_ylabel(r'y', fontsize=20)

        plt.gcf().subplots_adjust(left=0.145)
        plt.gcf().subplots_adjust(bottom=0.145)

        #plt.xscale("log")

        ymin, ymax = mod_util.FindResonableYminYmax(lhs, y)
        
        #ax.set_title('Eigenvalue spectrum')
        #ax.set_xlim([-10, 10])
        ax.set_ylim(ymin, ymax)
        ax.legend(loc='upper center', bbox_to_anchor=(0.48, 1.15),   # box with lower left corner at (x, y)
                  ncol=3, fancybox=True, shadow=True, fontsize=10, framealpha = 0.8)
        plt.show()

    def plot_reynolds_stress(self, rxx, ryy, rzz, rxy, rxz, ryz, y, str_bal):

        # Plot
        ax = plt.gca()
        ax.plot( rxx.real,
                 y,
                 'k',
                 label=r"$R_{xx}$",
                 linewidth=1.5 )
        #
        ax.plot( ryy.real,
                 y,
                 'r',
                 label=r"$R_{yy}$",
                 linewidth=1.5 )
        #
        ax.plot( rzz.real,
                 y,
                 'b',
                 label=r"$R_{zz}$",
                 linewidth=1.5 )

        #
        ax.plot( rxy.real,
                 y,
                 'm-.',
                 label=r"$R_{xy}$",
                 linewidth=1.5 )
        #
        line, = ax.plot( ryz.real,
                 y,
                 'y-',
                 label=r"$R_{yz}$",
                 linewidth=1.5 )
        line.set_dashes([8, 4, 2, 4, 2, 4])
        #
        ax.plot( rxz.real,
                 y,
                 ':',
                 color='orange',
                 label=r"$R_{xz}$",
                 linewidth=1.5 )
        #
        title_lab = "%s" % str_bal #str('{:.2e}'.format(scale_pd))
        ax.set_xlabel(title_lab, fontsize=20)
        ax.set_ylabel(r'y', fontsize=20)

        plt.gcf().subplots_adjust(left=0.145)
        plt.gcf().subplots_adjust(bottom=0.145)

        ymin, ymax = mod_util.FindResonableYminYmax(rxy, y)
        
        #ax.set_title('Eigenvalue spectrum')
        #ax.set_xlim([-10, 10])
        ax.set_ylim(ymin, ymax)
        ax.legend(loc='upper center', bbox_to_anchor=(0.48, 1.15),   # box with lower left corner at (x, y)
                  ncol=3, fancybox=True, shadow=True, fontsize=10, framealpha = 0.8)
        plt.show()

    def get_model_terms(self):

        # self.rxx_pres_diffus_model = 
        # self.ryy_pres_diffus_model =
        # self.rxy_pres_diffus_model = 

        Cr1 = 1.
        self.rxx_pres_strain_model = 4./3.*Cr1*np.multiply(self.rxy, self.solver.bsfl.Up)
        self.ryy_pres_strain_model = -2./3.*Cr1*np.multiply(self.rxy, self.solver.bsfl.Up)
        self.rxy_pres_strain_model = Cr1*np.multiply(self.ryy, self.solver.bsfl.Up)
        

class PostRayleighTaylor(PostProcess):
    def __init__(self):
        """
        Constructor of class PostShearLayer
        """
        super(PostRayleighTaylor, self).__init__(*args, **kwargs)

# class PostPoiseuille(PostProcess):
#     def __init__(self):
#         """
#         Constructor of class PostPoiseuille
#         """
#         super(PostPoiseuille, self).__init__(*args, **kwargs)
    

        


            
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
