import os
import sys
import warnings
# try:
#     import cupy as cp
# except ImportError:
#     warnings.warn('Failed to import cupy')

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
        self.nz = map.z.size
        self.map = map
        self.bsfl = bsfl
        self.mid_idx = mod_util.get_mid_idx(self.nz)
        
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
        print("Eigenvalue information:")
        print("=======================")
        print("Filtered eigenvalue with max. growth rate: %9.7f + 1i*%9.7f" % (self.omega_max.real, self.omega_max.imag))

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
        
        #jxu  = 5*self.nz-1 # 0
        #if (self.bsfl.rt_flag): jxu = 3*self.nz-1 # use 3*nz-1 for w and 5*nz-1 for rho

        #
        # Compute u00
        #
        #print("self.jxu = ", self.jxu)
        #print("re_in, fr_in, sc_in = ", re_in, fr_in, sc_in )
        
        # Build main stability matrices and assemble them
        self.call_to_build_matrices(Re_in=re_in, Fr_in=fr_in, Sc_in=sc_in)
        self.assemble_mat_lhs(alpha_in, beta_in, omega00)
        self.call_to_set_bc_secant(self.mat_lhs, self.vec_rhs, self.nz, self.map, alpha_in, beta_in)
            
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
            self.call_to_set_bc_secant(self.mat_lhs, self.vec_rhs, self.nz, self.map, alpha_in, beta_in)

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
            self.call_to_set_bc_secant(self.mat_lhs, self.vec_rhs, self.nz, self.map, alpha_in, beta_in)
            if (hascupy):
                self.qfinal = cp.linalg.solve(self.mat_lhs, self.vec_rhs)
            else:
                self.qfinal = linalg.solve(self.mat_lhs, self.vec_rhs)
        else:
            self.qfinal = q

        # Plot eigenvector locally if desired
        plot_eig_vec = 0
        if (plot_eig_vec==1):
            zlimp = 0.0
            nz = self.nz
            self.get_normalized_eigvects(self.qfinal, self.bsfl.rt_flag)
            #self.plot_real_imag_part(self.qfinal[0*nz:1*nz], "u", self.map.z, self.bsfl.rt_flag, zlimp)
            #self.plot_real_imag_part(self.qfinal[1*nz:2*nz], "v", self.map.z, self.bsfl.rt_flag, zlimp)
            #self.plot_real_imag_part(self.qfinal[2*nz:3*nz], "w", self.map.z, self.bsfl.rt_flag, zlimp)
            #self.plot_real_imag_part(self.qfinal[3*nz:4*nz], "p", self.map.z, self.bsfl.rt_flag, zlimp)
            #self.plot_real_imag_part(self.qfinal[4*nz:5*nz], "r", self.map.z, self.bsfl.rt_flag, zlimp)
            
            self.plot_real_imag_part(self.ueig, "u", self.map.z, self.bsfl.rt_flag, zlimp)
            self.plot_real_imag_part(self.veig, "v", self.map.z, self.bsfl.rt_flag, zlimp)
            #self.plot_real_imag_part(self.weig, "w", self.map.z, self.bsfl.rt_flag, zlimp)
            self.plot_real_imag_part(self.peig, "p", self.map.z, self.bsfl.rt_flag, zlimp)
            #self.plot_real_imag_part(self.reig, "r", self.map.z, self.bsfl.rt_flag, zlimp)

            #self.plot_real_imag_part(np.abs(self.ueig), "u", self.map.z, self.bsfl.rt_flag, zlimp)
            #self.plot_real_imag_part(np.abs(self.veig), "v", self.map.z, self.bsfl.rt_flag, zlimp)
            #self.plot_real_imag_part(np.abs(self.peig), "p", self.map.z, self.bsfl.rt_flag, zlimp)
            
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
            mod_util.write_out_eigenvalues(self.EigVal, self.nz)

    def write_eigvects_out_new(self):

        z = self.map.z
        
        ueig = self.ueig
        veig = self.veig
        weig = self.weig
        peig = self.peig
        reig = self.reig
        
        nz = len(z)
        # variables are u, v, w, p
        # data_out = np.column_stack([ z, np.abs(q[idx_u]), np.abs(q[idx_v]), np.abs(q[idx_w]), np.abs(q[idx_p]), \
        #                              q[idx_u].real, q[idx_v].real, q[idx_w].real, q[idx_p].real, \
        #                              q[idx_u].imag, q[idx_v].imag, q[idx_w].imag, q[idx_p].imag                  ])

        data_out = np.column_stack([ z, np.abs(ueig), np.abs(veig), np.abs(weig), np.abs(peig), np.abs(reig), \
                                     ueig.real, veig.real, weig.real, peig.real, reig.real, \
                                     ueig.imag, veig.imag, weig.imag, peig.imag, reig.imag                  ])

        if (self.Local):
            datafile_path = "./Eigenfunctions_Local_Solution_nondimensional_" + str(nz) + "_pts.txt"
        else:
            datafile_path = "./Eigenfunctions_Global_Solution_nondimensional_" + str(nz) + "_pts.txt"
            
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

        # zdim = z*Lref
    
        # data_out = np.column_stack([ zdim, np.abs(ueig), np.abs(veig), np.abs(weig), np.abs(peig), np.abs(reig), \
        #                              ueig.real, veig.real, weig.real, peig.real, reig.real, \
        #                              ueig.imag, veig.imag, weig.imag, peig.imag, reig.imag                  ])


        # if (self.Local):
        #     datafile_path = "./Eigenfunctions_Local_Solution_dimensional_" + str(nz) + "_pts.txt"
        # else:
        #     datafile_path = "./Eigenfunctions_Global_Solution_dimensional_" + str(nz) + "_pts.txt"
            
        # np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e',\
        #                                           '%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])


    def write_baseflow_out(self):

        z = self.map.z
        nz = len(z)
        
        # Rho, Rhop, Rhopp,  W and Wp are non-dimensional
        
        #data_out = np.column_stack([ znondim, zdim, Rho, Rhop, Rhopp, W, Wp, Rho_dim, Rhop_dim, Rhopp_dim, W_dim, Wp_dim ])
        data_out = np.column_stack([ z, z+12.5, self.bsfl.U, self.bsfl.Up,
                                     self.bsfl.W, self.bsfl.Wp,
                                     self.bsfl.Rho_nd, self.bsfl.Rho_nd-1.0,
                                     self.bsfl.Rhop_nd, self.bsfl.Rhopp_nd])
        datafile_path = "./Baseflow_" + str(nz) + "_pts.txt"
        
        #np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e',\
        #                                          '%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])
        np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e',\
                                                  '%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])
        
        #print("")
        #print("Baseflow has been written out")
        #print("")

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

        nz = self.nz

        if (self.Local):
            ueig_vec = q_eigvect[0*nz:1*nz]
            veig_vec = q_eigvect[1*nz:2*nz]
            weig_vec = q_eigvect[2*nz:3*nz]
            peig_vec = q_eigvect[3*nz:4*nz]

            if (rt_flag):
                if ( self.boussinesq == -555 ):
                    ueig_vec = q_eigvect[0*nz:1*nz]
                    veig_vec = q_eigvect[1*nz:2*nz]
                    reig_vec = q_eigvect[2*nz:3*nz]
                    peig_vec = q_eigvect[3*nz:4*nz]
                    
                    weig_vec = 0.0*ueig_vec
                else:
                    reig_vec = q_eigvect[4*nz:5*nz]
            else:
                reig_vec = 0.0*ueig_vec
            
        else:
            ueig_vec = q_eigvect[0*nz:1*nz]
            veig_vec = q_eigvect[1*nz:2*nz]
            weig_vec = q_eigvect[2*nz:3*nz]
            peig_vec = q_eigvect[3*nz:4*nz]

            if (rt_flag):
                if ( self.boussinesq == -555 ):
                    ueig_vec = q_eigvect[0*nz:1*nz]
                    veig_vec = q_eigvect[1*nz:2*nz]
                    reig_vec = q_eigvect[2*nz:3*nz]
                    peig_vec = q_eigvect[3*nz:4*nz]
                    
                    weig_vec = 0.0*ueig_vec
                else:
                    reig_vec = q_eigvect[4*nz:5*nz]
            else:
                reig_vec = 0.0*ueig_vec

        # Normalize eigenvectors
        self.norm_s, self.ueig, self.veig, self.weig, self.peig, self.reig = mod_util.normalize_eigenvectors(ueig_vec, veig_vec, weig_vec, peig_vec, reig_vec, rt_flag)

        # For the Generalized Eigenvalue Problem, I renormalize the whole eigenvector matrix
        if (not self.Local): self.EigVec = self.EigVec/self.norm_s

        
    def plot_eigvects(self, rt_flag):

        zlimp = 0.0
        self.plot_real_imag_part(self.ueig, "u", self.map.z, rt_flag, zlimp)
        self.plot_real_imag_part(self.veig, "v", self.map.z, rt_flag, zlimp)
        self.plot_real_imag_part(self.weig, "w", self.map.z, rt_flag, zlimp)
        self.plot_real_imag_part(self.reig, "r", self.map.z, rt_flag, zlimp)
        self.plot_real_imag_part(self.peig, "p", self.map.z, rt_flag, zlimp)

    def plot_real_imag_part(self, eig_fct, str_var, z, rt_flag, mv):

        symbols = 0
        
        ax = plt.gca()
                    
        # Get zmin-zmax
        zmin, zmax = mod_util.FindResonableZminZmax(eig_fct, z)

        if (self.boussinesq == 998): # Poiseuille flow
            zmin = -1
            zmax = 1

        # Get plot strings
        str_r, str_i = self.get_label_string(str_var)

        # Plot real part
        if (rt_flag and self.boussinesq == -555):
            ax.plot(z, eig_fct.real, 'k-', linewidth=1.5)
            if symbols==1:
                ax.plot(z, eig_fct.real, 'ks-', linewidth=1.5)
            ax.set_xlabel('x', fontsize=16)
            ax.set_ylabel(str_r, fontsize=16)            
        else:
            ax.plot(eig_fct.real, z, 'k-', linewidth=1.5)
            if symbols==1:
                ax.plot(eig_fct.real, z, 'ks-', linewidth=1.5)
            ax.set_xlabel(str_r, fontsize=16)
            ax.set_ylabel('z', fontsize=16)
            
        if (mv==0.):
            ax.set_ylim([zmin, zmax])
        else:
            ax.set_ylim([-mv, mv])

        plt.gcf().subplots_adjust(left=0.145)
        plt.gcf().subplots_adjust(bottom=0.145)

        plt.show()

        ax = plt.gca()
    
        # Plot imaginary part
        if (rt_flag and self.boussinesq == -555):
            ax.plot(z, eig_fct.imag, 'k-', linewidth=1.5)
            if symbols==1:
                ax.plot(z, eig_fct.imag, 'ks-', linewidth=1.5)
            ax.set_xlabel('x', fontsize=16)
            ax.set_ylabel(str_i, fontsize=16)
        else:
            if symbols==1:
                ax.plot(eig_fct.imag, z, 'ks-', linewidth=1.5)
            ax.plot(eig_fct.imag, z, 'k-', linewidth=1.5)
            ax.set_xlabel(str_i, fontsize=16)
            ax.set_ylabel('z', fontsize=16)
                
        if (mv==0.):
            ax.set_ylim([zmin, zmax])
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





        # phase_ur = np.arctan2(ueig_vec.imag, ueig_vec.real) - np.arctan2(reig_vec.imag, reig_vec.real)
        # phase_vr = np.arctan2(veig_vec.imag, veig_vec.real) - np.arctan2(reig_vec.imag, reig_vec.real)
        # phase_wr = np.arctan2(weig_vec.imag, weig_vec.real) - np.arctan2(reig_vec.imag, reig_vec.real)
        # phase_pr = np.arctan2(peig_vec.imag, peig_vec.real) - np.arctan2(reig_vec.imag, reig_vec.real)
        
        # # Normalize eigenvectors
        # self.norm_s, self.ueig, self.veig, self.weig, self.peig, self.reig = mod_util.normalize_eigenvectors(ueig_vec, veig_vec, weig_vec, peig_vec, reig_vec, rt_flag)

        # phase_ur_2 = np.arctan2(self.ueig.imag, self.ueig.real) - np.arctan2(self.reig.imag, self.reig.real)
        # phase_vr_2 = np.arctan2(self.veig.imag, self.veig.real) - np.arctan2(self.reig.imag, self.reig.real)
        # phase_wr_2 = np.arctan2(self.weig.imag, self.weig.real) - np.arctan2(self.reig.imag, self.reig.real)
        # phase_pr_2 = np.arctan2(self.peig.imag, self.peig.real) - np.arctan2(self.reig.imag, self.reig.real)

        # print("")
        # print("phase u - phase r = ", phase_ur[0:5])
        # print("phase u - phase r (2) = ", phase_ur_2[0:5])

        # print("")
        # print("phase v - phase r = ", phase_vr[0:5])
        # print("phase v - phase r (2) = ", phase_vr_2[0:5])

        # print("")
        # print("phase w - phase r = ", phase_wr[0:5])
        # print("phase w - phase r (2) = ", phase_wr_2[0:5])

        # print("")
        # print("phase p - phase r = ", phase_pr[0:5])
        # print("phase p - phase r (2) = ", phase_pr_2[0:5])

        
        #print("phase r = ", np.arctan2(self.reig.imag, self.reig.real))
        #print("phase r = ", np.arctan2(reig_vec.imag, reig_vec.real))

        
