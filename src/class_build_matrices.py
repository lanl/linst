
#from termios import N_TTY
import module_utilities as mod_util
import numpy as np

import warnings
import sys

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8


class BuildMatrices:
    """
    This class builds the main matrices A (mat_lhs) and B (mat_rhs) 
    required to form the temporal generalized eigenvalue problem A*q = omega*B*q
    """
    #def __init__(self, size): # here I pass 4*ny for regular Navier-Stokes Stability and 5*ny for "Rayleigh-Taylor" Navier-Stokes
    def __init__(self, ny, rt_flag, prim_form): # here I pass 4*ny for regular Navier-Stokes Stability and 5*ny for "Rayleigh-Taylor" Navier-Stokes
        """
        Constructor for class BuilMatrices
        """
        if (rt_flag == True):
            if (prim_form==1):

                size = 5*ny

                """Boussinesq flag:   1 -> Boussinesq -------------> (non-dimensional)
                                 #   10 -> dimensional Boussinesq -> (dimensional)
                                 #   -2 -> Chandrasekhar ----------> (dimensional)
                                 #  -20 -> Chandrasekhar ----------> (non-dimensional)
                                 #   -3 -> Sandoval ---------------> (non-dimensional)
                                 #   -4 -> modified Sandoval ------> (non-dimensional)
                                 # -555 -> compressible -----------> (non-dimensional)
                """
                
                self.boussinesq = 999

                if   (self.boussinesq == 1 ):
                    pass
                    # print("")
                    # print('-----------------------------------------------------')
                    # print('Equations for Rayleight-Taylor set to "Boussinesq"')
                    # print('-----------------------------------------------------')
                    # print("")
                elif (self.boussinesq == 10 ):
                    pass
                    # print("")
                    # print('--------------------------------------------------------------')
                    # print('Equations for Rayleight-Taylor set to "Boussinesq" DIMENSIONAL')
                    # print('--------------------------------------------------------------')
                    # print("")
                elif (self.boussinesq == -2 ):
                    pass
                    # print("")
                    # print('-----------------------------------------------------')
                    # print('Equations for Rayleight-Taylor set to "Chandrasekhar"')
                    # print('-----------------------------------------------------')
                    # print("")
                elif (self.boussinesq == -3 ):
                    pass
                    # print("")
                    # print('-----------------------------------------------------')
                    # print('Equations for Rayleight-Taylor set to "Sandoval"')
                    # print('-----------------------------------------------------')
                    # print("")
                elif (self.boussinesq == -4 ):
                    pass
                    # print("")
                    # print('---------------------------------------------------------')
                    # print('Equations for Rayleight-Taylor set to "Modified Sandoval"')
                    # print('---------------------------------------------------------')
                    # print("")
                elif (self.boussinesq == -20 ):
                    pass
                    # print("")
                    # print('-----------------------------------------------------')
                    # print('Equations for Rayleight-Taylor set to "Chandrasekhar NON-DIMENSIONAL"')
                    # print('-----------------------------------------------------')
                    # print("")
                elif (self.boussinesq == -555 ): # compressible inviscid
                    size = 4*ny
                    # print("")
                    # print('---------------------------------------------------------')
                    # print('Equations for Rayleight-Taylor Compressible Inviscid    "')
                    # print('---------------------------------------------------------')
                    # print("")
                else:
                    sys.exit("Not a proper value for boussinesq flag!")
            else:
                sys.exit("Just used for inviscid debugging 123456789")
                size = ny
        else:
            self.boussinesq = -999
            size = 4*ny

        self.size = size
        # Initialize arrays
        self.mat_a1 = np.zeros((size,size), dpc)
        self.mat_a2 = np.zeros((size,size), dpc)
        self.mat_b1 = np.zeros((size,size), dpc)
        self.mat_b2 = np.zeros((size,size), dpc)

        self.mat_ab = np.zeros((size,size), dpc)
        
        self.mat_d1 = np.zeros((size,size), dpc)
        
        self.mat_lhs = np.zeros((size,size), dpc)
        self.mat_rhs = np.zeros((size,size), dpc)

        self.vec_rhs = np.zeros(size, dpc)

    def set_bc_shear_layer(self, lhs, rhs, ny, map):
        """
        This function sets the boundary conditions for the free shear layer test case:
        1) u, v, w = 0 for y --> +/- infinity
        2) The boundary conditions (called compatibility conditions) for pressure are derived
        by considering the y-momentum equation in the freestream
        """
        ######################################
        # (1) FREESTREAM BOUNDARY-CONDITIONS #
        ######################################
        
        ##################
        # u-velocity BCs #
        ##################
        idx_u_ymin = 0*ny
        idx_u_ymax = 1*ny-1

        # ymin
        lhs[idx_u_ymin,:] = 0.0
        rhs[idx_u_ymin,:] = 0.0        
        lhs[idx_u_ymin, idx_u_ymin] = 1.0

        # ymax
        lhs[idx_u_ymax,:] = 0.0
        rhs[idx_u_ymax,:] = 0.0
        lhs[idx_u_ymax, idx_u_ymax] = 1.0
        
        ##################
        # v-velocity BCs #
        ##################
        idx_v_ymin = 1*ny
        idx_v_ymax = 2*ny-1

        # ymin
        lhs[idx_v_ymin,:] = 0.0
        rhs[idx_v_ymin,:] = 0.0        
        lhs[idx_v_ymin, idx_v_ymin] = 1.0

        # ymax
        lhs[idx_v_ymax,:] = 0.0
        rhs[idx_v_ymax,:] = 0.0
        lhs[idx_v_ymax, idx_v_ymax] = 1.0

        ##################
        # w-velocity BCs #
        ##################
        idx_w_ymin = 2*ny
        idx_w_ymax = 3*ny-1

        # ymin
        lhs[idx_w_ymin,:] = 0.0
        rhs[idx_w_ymin,:] = 0.0        
        lhs[idx_w_ymin, idx_w_ymin] = 1.0

        # ymax
        lhs[idx_w_ymax,:] = 0.0
        rhs[idx_w_ymax,:] = 0.0
        lhs[idx_w_ymax, idx_w_ymax] = 1.0

        ##################
        # pressure BCs #
        ##################

        set_pres_bc = 1

        if set_pres_bc == 1:
            idx_p_ymin = 3*ny
            idx_p_ymax = 4*ny-1
            
            # ymin
            lhs[idx_p_ymin,:] = 0.0
            rhs[idx_p_ymin,:] = 0.0        
            lhs[idx_p_ymin, idx_p_ymin:4*ny] = map.D1[0, :]
            
            # ymax
            lhs[idx_p_ymax,:] = 0.0
            rhs[idx_p_ymax,:] = 0.0
            lhs[idx_p_ymax, idx_p_ymin:4*ny] = map.D1[-1, :]

        ######################################
        #       (2) SYMMETRY BC AT Y=0       #
        ######################################

        set_sym_bc = 0

        if set_sym_bc == 1:

            mid_idx = mod_util.get_mid_idx(ny)
                        
            idx_u_y0 = mid_idx
            idx_v_y0 = mid_idx + ny
            idx_w_y0 = mid_idx + 2*ny
            
            lhs[idx_u_y0, :] = 0.0
            lhs[idx_u_y0, 0:ny] = map.D1[mid_idx, :]

            rhs[idx_u_y0,:] = 0.0
            
            lhs[idx_v_y0, :] = 0.0
            lhs[idx_v_y0, idx_v_y0] = 1.0
            
            lhs[idx_w_y0, :] = 0.0
            lhs[idx_w_y0, idx_w_y0] = 1.0
            
            rhs[idx_v_y0, :] = 0.0
            rhs[idx_w_y0, :] = 0.0        

    def set_matrices(self, ny, Re, bsfl, map): # here I pas ny
        """
        This function builds the stability matrices for the incompressible stability equations
        a1 = [iU   0   0   i;
              0    iU  0   0;
              0    0   iU  0;
              i    0   0   0 ]

        a2 = [1/Re  0     0   0;
              0    1/Re   0   0;
              0     0    1/Re 0;
              0     0     0   0 ]

        b1 = [iW   0   0   0;
              0    iW  0   0;
              0    0   iW  i;
              0    0   i   0 ]

        b2 = [1/Re  0     0   0;
              0    1/Re   0   0;
              0     0    1/Re 0;
              0     0     0   0 ]

        d1 = [-D2/Re   U'      0   0;
                 0   -D2/Re    0   D;
                 0     W'   -D2/Re 0;
                 0     D       0   0 ]
        """
        nt  = 4*ny

        # Identity matrix id
        id      = np.identity(ny)

        # id/Re
        id_r_re = id/Re

        # Create diagonal matrices from baseflow vectors
        dU      = np.diag(bsfl.U)
        dW      = np.diag(bsfl.W)
        dUp     = np.diag(bsfl.Up)
        dWp     = np.diag(bsfl.Wp)

        # print("dU has the following size: ", np.shape(dU))

        # print("dU = ", dU)
        # if (ny % 2) == 0:
        #     #print (“The number is even”)
        #     nout = int(ny/2)-5
        #     print("nout = ", nout)
        #     print("dU[nout, :] = ", dU[nout, :])
        # else:
        #     #print (“The provided number is odd”)
        #     nout = int((ny-1)/2)-5
        #     print("nout = ", nout)
        #     print("dU[nout, :] = ", dU[nout, :])
            
        # print("")
        # print("id_r_re = ",id_r_re)

        # input('hello from here')

        # 1st block indices
        imin = 0
        imax = ny

        self.mat_a1[imin:imax, imin:imax]           = 1j*dU
        self.mat_a1[imin:imax, imin+3*ny:imax+3*ny] = 1j*id

        self.mat_a2[imin:imax, imin:imax]           = id_r_re

        self.mat_b1[imin:imax, imin:imax]           = 1j*dW
        self.mat_b2[imin:imax, imin:imax]           = id_r_re

        self.mat_d1[imin:imax, imin:imax]           = -map.D2/Re
        self.mat_d1[imin:imax, imin+ny:imax+ny]     = dUp

        self.mat_rhs[imin:imax, imin:imax]          = 1j*id

        # 2nd block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a1[imin:imax, imin:imax]           = 1j*dU
        self.mat_a2[imin:imax, imin:imax]           = id_r_re

        self.mat_b1[imin:imax, imin:imax]           = 1j*dW
        self.mat_b2[imin:imax, imin:imax]           = id_r_re

        self.mat_d1[imin:imax, imin:imax]           = -map.D2/Re
        self.mat_d1[imin:imax, imin+2*ny:imax+2*ny] = map.D1

        self.mat_rhs[imin:imax, imin:imax]          = 1j*id

        # 3rd block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a1[imin:imax, imin:imax]       = 1j*dU
        self.mat_a2[imin:imax, imin:imax]       = id_r_re

        self.mat_b1[imin:imax, imin:imax]       = 1j*dW
        self.mat_b1[imin:imax, imin+ny:imax+ny] = 1j*id

        self.mat_b2[imin:imax, imin:imax]       = id_r_re

        self.mat_d1[imin:imax, imin:imax]       = -map.D2/Re
        self.mat_d1[imin:imax, imin-ny:imax-ny] = dWp

        #print("imin = ", imin)
        #print("mat_d1[imin,:] = ",mat_d1[imin,:])

        #input('checking mat_d1')

        self.mat_rhs[imin:imax, imin:imax] = 1j*id   

        # 4th block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a1[imin:imax, imin-3*ny:imax-3*ny] = 1j*id

        self.mat_b1[imin:imax, imin-ny:imax-ny]     = 1j*id

        self.mat_d1[imin:imax, imin-2*ny:imax-2*ny] = map.D1  
    
        #print("")
        #print("self.mat_rhs = ", self.mat_rhs)

        #self.mat_lhs = alpha*mat_a1 + alpha**2.*mat_a2 + beta*mat_b1 + beta**2.*mat_b2 + mat_d1

        # print("norm a1 = ", np.linalg.norm(self.mat_a1))
        # print("norm a2 = ", np.linalg.norm(self.mat_a2))
        
        # print("norm b1 = ", np.linalg.norm(self.mat_b1))
        # print("norm b2 = ", np.linalg.norm(self.mat_b2))

        # print("norm d1 = ", np.linalg.norm(self.mat_d1))
    
    def assemble_mat_lhs(self, alpha, beta, omega, Tracking, Local, bsfl_ref):
        """
        This function assemble the lhs matrix mat_lhs
        """

        if ( self.boussinesq == -555 ):
            self.mat_lhs = self.mat_a1
        else:
            self.mat_lhs = alpha*self.mat_a1 + alpha**2.*self.mat_a2 + beta*self.mat_b1 + beta**2.*self.mat_b2 + self.mat_d1

        check_norms = 0

        if (check_norms == 1):

            mat_a1_norm = np.linalg.norm(self.mat_a1)
            mat_a2_norm = np.linalg.norm(self.mat_a2)
            
            mat_b1_norm = np.linalg.norm(self.mat_b1)
            mat_b2_norm = np.linalg.norm(self.mat_b2)
            
            mat_d1_norm = np.linalg.norm(self.mat_d1)
            
            mat_lhs_norm = np.linalg.norm(self.mat_lhs)
            mat_rhs_norm = np.linalg.norm(self.mat_rhs)

            print("")
            print("alpha, beta, omega: ", alpha, beta, omega)
            print("Matrix norm of mat_lhs = ", mat_lhs_norm)
            print("Matrix norm of mat_rhs = ", mat_rhs_norm)
            print("")
            print("Matrix norm of mat_a1 = ", mat_a1_norm)
            print("Matrix norm of mat_a2 = ", mat_a2_norm)
            print("")
            print("Matrix norm of mat_b1 = ", mat_b1_norm)
            print("Matrix norm of mat_b2 = ", mat_b2_norm)
            print("")
            print("Matrix norm of mat_d1 = ", mat_d1_norm)
            print("")
            
            input("Check norms before!")
            
        if ( self.boussinesq == -3 or self.boussinesq == -4 ):
            self.mat_lhs = self.mat_lhs + alpha*beta*self.mat_ab 
        
        if (Tracking and Local):
            self.mat_lhs = self.mat_lhs - omega*self.mat_rhs
            
        if (check_norms == 1):

            mat_lhs_norm = np.linalg.norm(self.mat_lhs)

            print("")
            print("Matrix norm of mat_lhs = ", mat_lhs_norm)
            print("")
            
            input("Check norms after!")

        #print("np.shape(self.mat_lhs)=", np.shape(self.mat_lhs))
        #print("np.shape(alpha*self.mat_a1)=", np.shape(alpha*self.mat_a1))


    def set_bc_shear_layer_secant(self, lhs, rhs, ny, map, alpha, beta):
        """
        This function sets the boundary conditions for the free shear layer test case:
        SECANT METHOD!!!!!!!!!!!!!!!!!
        """
        ######################################
        # (1) FREESTREAM BOUNDARY-CONDITIONS #
        ######################################
        
        ##################
        # u-velocity BCs #
        ##################
        idx_u_ymin = 0*ny # replace this by p=1 at midplane
        idx_u_ymax = 1*ny-1

        # ymin
        #lhs[idx_u_ymin,:] = 0.0
        #rhs[idx_u_ymin  ] = 0.0        
        #lhs[idx_u_ymin, idx_u_ymin] = 1.0

        mid_idx = mod_util.get_mid_idx(ny)

        # I checked from eigenfunctions from global solver:
        # at y = 0, the real part of pressure is zero, while the imaginary part has a max
        # that is why I am setting p = 0 + 1j        
        lhs[3*ny+mid_idx,:] = 0.0
        rhs[3*ny+mid_idx]   = 0.0 + 1.0*1j
        lhs[3*ny+mid_idx, 3*ny+mid_idx] = 1.0

        #mid_idx = mod_util.get_mid_idx(ny)
        #lhs[mid_idx,:] = 0.0
        #lhs[mid_idx, mid_idx] = 1.0
        #rhs[mid_idx] = 1.0

        # ymax
        lhs[idx_u_ymax,:] = 0.0
        rhs[idx_u_ymax  ] = 0.0
        lhs[idx_u_ymax, idx_u_ymax] = 1.0
        
        ##################
        # v-velocity BCs #
        ##################
        idx_v_ymin = 1*ny
        idx_v_ymax = 2*ny-1

        # ymin
        lhs[idx_v_ymin,:] = 0.0
        rhs[idx_v_ymin  ] = 0.0        
        lhs[idx_v_ymin, idx_v_ymin] = 1.0

        # ymax
        lhs[idx_v_ymax,:] = 0.0
        rhs[idx_v_ymax  ] = 0.0
        lhs[idx_v_ymax, idx_v_ymax] = 1.0

        ##################
        # w-velocity BCs #
        ##################
        idx_w_ymin = 2*ny
        idx_w_ymax = 3*ny-1

        # ymin
        lhs[idx_w_ymin,:] = 0.0
        rhs[idx_w_ymin  ] = 0.0        
        lhs[idx_w_ymin, idx_w_ymin] = 1.0

        # ymax
        lhs[idx_w_ymax,:] = 0.0
        rhs[idx_w_ymax  ] = 0.0
        lhs[idx_w_ymax, idx_w_ymax] = 1.0

        ##################
        # pressure BCs #
        ##################
        
        #mid_idx = mod_util.get_mid_idx(ny)

        # I checked from eigenfunctions from global solver:
        # at y = 0, the real part of pressure is zero, while the imaginary part has a max
        # that is why I am setting p = 0 + 1j
        #lhs[3*ny+mid_idx,:] = 0.0
        #rhs[3*ny+mid_idx]   = 0.0 + 1.0*1j
        #lhs[3*ny+mid_idx, 3*ny+mid_idx] = 1.0

        #lhs[mid_idx,:] = 0.0
        #rhs[mid_idx]   = -1.0
        #lhs[mid_idx, mid_idx] = 1.0

        
        # setting v ==> cannot do that because continuity equation already in there ==> singular matrix
        # set rhs from continuity equation dv/dy = -1j*alpha*u - 1j*beta*w
        #lhs[1*ny+mid_idx,:] = 0.0
        #rhs[1*ny+mid_idx  ] = -1j*alpha -1j*beta    
        #lhs[1*ny+mid_idx, ny:2*ny] = map.D1[mid_idx,:]
        #print("abs(-1j*alpha -1j*beta) = ", abs(-1j*alpha -1j*beta))

        ##################
        # pressure BCs #
        ##################

        set_pres_bc = 0

        if set_pres_bc == 1:
            idx_p_ymin = 3*ny
            idx_p_ymax = 4*ny-1
            
            # ymin
            lhs[idx_p_ymin,:] = 0.0
            rhs[idx_p_ymin  ] = 0.0        
            lhs[idx_p_ymin, idx_p_ymin:4*ny] = map.D1[0, :]
            
            # ymax
            lhs[idx_p_ymax,:] = 0.0
            rhs[idx_p_ymax  ] = 0.0
            lhs[idx_p_ymax, idx_p_ymin:4*ny] = map.D1[-1, :]


    def set_matrices_rayleigh_taylor_chandrasekhar(self, ny, bsfl, bsfl_ref, map): # here I pas ny
        """
        """
        # Identity matrix id
        id      = np.identity(ny)

        #print("bsfl.Mu = ", bsfl.Mu)
        #print("bsfl.Mup = ", bsfl.Mup)
        #print("")
        #print("bsfl.Rho = ", bsfl.Rho)
        #print("bsfl.Rhop = ", bsfl.Rhop)
        #print("")
        
        # Create diagonal matrices from baseflow vectors
        dMu     = np.diag(bsfl.Mu)
        dRho    = np.diag(bsfl.Rho)

        dMup    = np.diag(bsfl.Mup)
        dRhop   = np.diag(bsfl.Rhop)

        Lder    = np.matmul(dMup, map.D1)
        Lder2   = np.matmul(dMu, map.D2)

        grav    = bsfl_ref.gref

        if ( grav != 9.81 ):
            print("")
            print("Chandrasekhar R-T solver, gravity should be dimensional, g = ", grav)
            input("Check gravity")

        #print("")
        #print("R-T Chandrasekhar Equations (dimensional)")
        #print("-----------------------------------------------------------------")

        # 1st block indices
        imin = 0
        imax = ny

        self.mat_a1[imin:imax, imin+2*ny:imax+2*ny] = 1j*dMup
        self.mat_a1[imin:imax, imin+3*ny:imax+3*ny] = -1j*id

        self.mat_a2[imin:imax, imin:imax]           = -dMu

        self.mat_b2[imin:imax, imin:imax]           = -dMu

        self.mat_d1[imin:imax, imin:imax]           = Lder2 + Lder

        self.mat_rhs[imin:imax, imin:imax]          = -1j*dRho

        # 2nd block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -dMu

        self.mat_b1[imin:imax, imin+1*ny:imax+1*ny] = 1j*dMup
        self.mat_b1[imin:imax, imin+2*ny:imax+2*ny] = -1j*id
                
        self.mat_b2[imin:imax, imin:imax]           = -dMu

        self.mat_d1[imin:imax, imin:imax]           = Lder2 + Lder

        self.mat_rhs[imin:imax, imin:imax]          = -1j*dRho

        # 3rd block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -dMu

        self.mat_b2[imin:imax, imin:imax]           = -dMu

        self.mat_d1[imin:imax, imin:imax]           = Lder2 + 2*Lder
        self.mat_d1[imin:imax, imin+1*ny:imax+1*ny] = -map.D1 
        self.mat_d1[imin:imax, imin+2*ny:imax+2*ny] = -grav*id 

        self.mat_rhs[imin:imax, imin:imax]          = -1j*dRho

        # 4th block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a1[imin:imax, imin-3*ny:imax-3*ny] = 1j*id

        self.mat_b1[imin:imax, imin-2*ny:imax-2*ny] = 1j*id

        self.mat_d1[imin:imax, imin-1*ny:imax-1*ny] = map.D1

        # 5th block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_d1[imin:imax, imin-2*ny:imax-2*ny] = -dRhop

        self.mat_rhs[imin:imax, imin:imax]          = -1j*id        
        
    def set_bc_rayleigh_taylor(self, lhs, rhs, ny, map):
        """
        """
        ######################################
        # (1) FREESTREAM BOUNDARY-CONDITIONS #
        ######################################
        
        ##################
        # u-velocity BCs #
        ##################
        idx_u_ymin = 0*ny
        idx_u_ymax = 1*ny-1

        # ymin
        lhs[idx_u_ymin,:] = 0.0
        rhs[idx_u_ymin,:] = 0.0        
        lhs[idx_u_ymin, idx_u_ymin] = 1.0

        # ymax
        lhs[idx_u_ymax,:] = 0.0
        rhs[idx_u_ymax,:] = 0.0
        lhs[idx_u_ymax, idx_u_ymax] = 1.0
        
        ##################
        # v-velocity BCs #
        ##################
        idx_v_ymin = 1*ny
        idx_v_ymax = 2*ny-1

        # ymin
        lhs[idx_v_ymin,:] = 0.0
        rhs[idx_v_ymin,:] = 0.0        
        lhs[idx_v_ymin, idx_v_ymin] = 1.0

        # ymax
        lhs[idx_v_ymax,:] = 0.0
        rhs[idx_v_ymax,:] = 0.0
        lhs[idx_v_ymax, idx_v_ymax] = 1.0

        ##################
        # w-velocity BCs #
        ##################
        idx_w_ymin = 2*ny
        idx_w_ymax = 3*ny-1

        # ymin
        lhs[idx_w_ymin,:] = 0.0
        rhs[idx_w_ymin,:] = 0.0        
        lhs[idx_w_ymin, idx_w_ymin] = 1.0

        # ymax
        lhs[idx_w_ymax,:] = 0.0
        rhs[idx_w_ymax,:] = 0.0
        lhs[idx_w_ymax, idx_w_ymax] = 1.0


        ##################
        # pressure BCs #
        ##################
        set_pres = 1

        if (set_pres==1):
            idx_p_ymin = 3*ny
            idx_p_ymax = 4*ny-1
            
            # ymin
            lhs[idx_p_ymin,:] = 0.0
            rhs[idx_p_ymin  ] = 0.0        
            #lhs[idx_p_ymin, idx_p_ymin:4*ny] = map.D1[0, :]
            lhs[idx_p_ymin, idx_p_ymin     ] = 1.0
            
            # ymax
            lhs[idx_p_ymax,:] = 0.0
            rhs[idx_p_ymax  ] = 0.0
            #lhs[idx_p_ymax, idx_p_ymin:4*ny] = map.D1[-1, :]
            lhs[idx_p_ymax, idx_p_ymax]      = 1.0

        ##################
        # density BCs #
        ##################
        set_dens = 1

        if (set_dens==1):
            idx_r_ymin = 4*ny
            idx_r_ymax = 5*ny-1
            
            # ymin
            lhs[idx_r_ymin,:] = 0.0
            rhs[idx_r_ymin  ] = 0.0        
            lhs[idx_r_ymin, idx_r_ymin] = 1.0
            
            # ymax
            lhs[idx_r_ymax,:] = 0.0
            rhs[idx_r_ymax  ] = 0.0
            lhs[idx_r_ymax, idx_r_ymax] = 1.0
        

    def set_matrices_rayleigh_taylor_inviscid_w_equation(self, ny, bsfl, map):
        """
        Hello
        """
        # Identity matrix id
        id      = np.identity(ny)

        # Create diagonal matrices from baseflow vectors
        dMu     = np.diag(bsfl.Mu)
        dRho    = np.diag(bsfl.Rho)

        dMup    = np.diag(bsfl.Mup)
        dRhop   = np.diag(bsfl.Rhop)

        grav    = self.grav

        print("")
        print("In this solver, gravity should be dimensional, g = ", self.grav)
        input("Check gravity in set_matrices_rayleigh_taylor_inviscid_w_equation")

        # 1st and only block indices
        imin = 0
        imax = ny

        self.mat_a2[imin:imax, imin:imax]  = -dRho

        self.mat_d1[imin:imax, imin:imax]  = np.matmul(dRhop, map.D1) + np.matmul(dRho, map.D2)

        self.mat_rhs[imin:imax, imin:imax] = -grav*dRhop


    def assemble_mat_lhs_rt_inviscid(self, alpha, beta, omega, Tracking, Local):
        """
        This function assemble the lhs matrix mat_lhs
        """

        k2 = alpha**2. + beta**2.
        
        self.mat_lhs = k2*self.mat_a2 + self.mat_d1        

        self.mat_rhs = self.mat_rhs*k2 

    def set_bc_rayleigh_taylor_inviscid(self, lhs, rhs, ny, map):
        """
        """

        # BCs: w=0 at wall and Dw=0 (from continuity)
        ##################
        # w-velocity BCs #
        ##################
        idx_u_ymin = 0*ny
        idx_u_ymax = 1*ny-1

        # ymin ==> w=0
        lhs[0,:]  = 0.0
        rhs[0,:]  = 0.0        
        lhs[0, 0] = 1.0

        # ymin ==> Dw=0
        lhs[1,:] = 0.0
        rhs[1,:] = 0.0        
        lhs[1, 0:ny] = map.D1[0,:]

        # ymax ==> w=0
        lhs[ny-1,:] = 0.0
        rhs[ny-1,:] = 0.0
        lhs[ny-1, ny-1] = 1.0

        # ymax ==> Dw=0
        lhs[ny-2,:] = 0.0
        rhs[ny-2,:] = 0.0
        lhs[ny-2, 0:ny] = map.D1[-1,:]

        
    def set_bc_plane_poiseuille_secant(self, lhs, rhs, ny, map, alpha, beta):
        """
        This function sets the boundary conditions for the free shear layer test case:
        SECANT METHOD!!!!!!!!!!!!!!!!!
        """
        ######################################
        # (1) FREESTREAM BOUNDARY-CONDITIONS #
        ######################################
        
        ##################
        # u-velocity BCs #
        ##################
        idx_u_ymin = 0*ny # replace this by p=1
        idx_u_ymax = 1*ny-1

        # ymin: replce u(ymin) = 0 by p(ymin) = 1
        lhs[idx_u_ymin,:] = 0.0
        lhs[idx_u_ymin, 3*ny ] = 1.0

        # I tried 1 and 1 + 1j but eigenvalues are same (eigenfunctions could be different)
        rhs[idx_u_ymin  ] = 1.0 #+ 1j       
        
        # ymax
        lhs[idx_u_ymax,:] = 0.0
        rhs[idx_u_ymax  ] = 0.0
        lhs[idx_u_ymax, idx_u_ymax] = 1.0
        
        ##################
        # v-velocity BCs #
        ##################
        idx_v_ymin = 1*ny
        idx_v_ymax = 2*ny-1

        # ymin
        lhs[idx_v_ymin,:] = 0.0
        rhs[idx_v_ymin  ] = 0.0        
        lhs[idx_v_ymin, idx_v_ymin] = 1.0

        # ymax
        lhs[idx_v_ymax,:] = 0.0
        rhs[idx_v_ymax  ] = 0.0
        lhs[idx_v_ymax, idx_v_ymax] = 1.0

        ##################
        # w-velocity BCs #
        ##################
        idx_w_ymin = 2*ny
        idx_w_ymax = 3*ny-1

        # ymin
        lhs[idx_w_ymin,:] = 0.0
        rhs[idx_w_ymin  ] = 0.0        
        lhs[idx_w_ymin, idx_w_ymin] = 1.0

        # ymax
        lhs[idx_w_ymax,:] = 0.0
        rhs[idx_w_ymax  ] = 0.0
        lhs[idx_w_ymax, idx_w_ymax] = 1.0


    def set_bc_rayleigh_taylor_secant(self, lhs, rhs, ny, map, alpha, beta):
        """
        """
        ######################################
        # (1) FREESTREAM BOUNDARY-CONDITIONS #
        ######################################

        # NOTE: YOU NEED AN INHOMOGENEOUS BC HERE OTHERWISE SOLUTION = 0
        
        ##################
        # u-velocity BCs #
        ##################
        idx_u_ymin = 0*ny # replace this by rho=1 at midplane
        idx_u_ymax = 1*ny-1

        # ymin
        lhs[idx_u_ymin,:] = 0.0
        rhs[idx_u_ymin  ] = 0.0        
        lhs[idx_u_ymin, idx_u_ymin] = 1.0
        
        mid_idx = mod_util.get_mid_idx(ny)
        
        #lhs[idx_u_ymin,:] = 0.0
        #rhs[idx_u_ymin  ] = 1.0 #+ 1.0*1j # 1.0*1j # 
        #lhs[idx_u_ymin, 4*ny+mid_idx] = 1.0
        
        #print("mid_idx = ", mid_idx)

        # I checked from eigenfunctions from global solver:
        # at y = 0, the real and imaginary parts of density are nonzero, that is why I am setting rho = 1 + 1j
        # although the real part seems really small compared to the imaginary part
        #lhs[idx_u_ymin,:] = 0.0
        #rhs[idx_u_ymin  ] = 1.0 + 1.0*1j
        #lhs[idx_u_ymin, 4*ny+mid_idx] = 1.0

        # This does not work it seems, whatever the IC, it goes really close to it ==>  no convergence towards eigenvalue
        #lhs[4*ny+mid_idx,:] = 0.0
        #rhs[4*ny+mid_idx  ] = 1.0 + 1.0*1j # 1.0*1j # 
        #lhs[4*ny+mid_idx, 4*ny+mid_idx] = 1.0

        # Better it seems
        #lhs[4*ny+1,:] = 0.0
        #rhs[4*ny+1  ] = 1. + 1j #+ 1.0*1j # 1.0*1j # 
        #lhs[4*ny+1, 4*ny+mid_idx] = 1.0

        #print("4*ny+mid_idx = ", 4*ny+mid_idx)
        
        #lhs[5*ny-2,:] = 0.0
        #rhs[5*ny-2  ] = 1.0 + 1.0*1j # 
        #lhs[5*ny-2, 4*ny+mid_idx] = 1.0

        #lhs[5*ny-3,:] = 0.0
        #rhs[5*ny-3  ] = 1.0 # 1.0*1j # 
        #lhs[5*ny-3, 4*ny+mid_idx+1] = 1.0
        
        # ymax
        lhs[idx_u_ymax,:] = 0.0
        rhs[idx_u_ymax  ] = 0.0
        lhs[idx_u_ymax, idx_u_ymax] = 1.0
        
        ##################
        # v-velocity BCs #
        ##################
        idx_v_ymin = 1*ny
        idx_v_ymax = 2*ny-1

        # ymin
        lhs[idx_v_ymin,:] = 0.0
        rhs[idx_v_ymin  ] = 0.0        
        lhs[idx_v_ymin, idx_v_ymin] = 1.0

        # ymax
        lhs[idx_v_ymax,:] = 0.0
        rhs[idx_v_ymax  ] = 0.0
        lhs[idx_v_ymax, idx_v_ymax] = 1.0

        ##################
        # w-velocity BCs #
        ##################
        idx_w_ymin = 2*ny
        idx_w_ymax = 3*ny-1

        # ymin
        lhs[idx_w_ymin,:] = 0.0
        rhs[idx_w_ymin  ] = 0.0        
        lhs[idx_w_ymin, idx_w_ymin] = 1.0

        # ymax
        #lhs[idx_w_ymax,:] = 0.0
        #rhs[idx_w_ymax  ] = 0.0
        #lhs[idx_w_ymax, idx_w_ymax] = 1.0

        # Replace w(ymax)=0 by bc below and iterate in class_solve_gevp to satisfy w(ymax)=0
        lhs[2*ny+mid_idx,:] = 0.0
        rhs[2*ny+mid_idx  ] = 1.0 + 1.0*1j # 1.0*1j # 
        lhs[2*ny+mid_idx, 2*ny+mid_idx] = 1.0

        ##################
        # pressure BCs #
        ##################
        set_pres = 1

        if (set_pres==1):
            idx_p_ymin = 3*ny
            idx_p_ymax = 4*ny-1
            
            # ymin
            lhs[idx_p_ymin,:] = 0.0
            rhs[idx_p_ymin  ] = 0.0        
            #lhs[idx_p_ymin, idx_p_ymin:4*ny] = map.D1[0, :]
            lhs[idx_p_ymin, idx_p_ymin]      = 1.0
            
            # ymax
            lhs[idx_p_ymax,:] = 0.0
            rhs[idx_p_ymax  ] = 0.0
            #lhs[idx_p_ymax, idx_p_ymin:4*ny] = map.D1[-1, :]
            lhs[idx_p_ymax, idx_p_ymax]      = 1.0

        ##################
        # density BCs #
        ##################
        set_dens = 1

        if (set_dens==1):
            idx_r_ymin = 4*ny
            idx_r_ymax = 5*ny-1
            
            # ymin
            lhs[idx_r_ymin,:] = 0.0
            rhs[idx_r_ymin  ] = 0.0
            #lhs[idx_r_ymin, idx_r_ymin:5*ny] = map.D1[0, :]
            lhs[idx_r_ymin, idx_r_ymin] = 1.0
            
            # ymax
            lhs[idx_r_ymax,:] = 0.0
            rhs[idx_r_ymax  ] = 0.0
            #lhs[idx_r_ymax, idx_r_ymin:5*ny] = map.D1[-1, :]
            lhs[idx_r_ymax, idx_r_ymax] = 1.0

            # Replace rho(ymax)=0 by bc below and iterate in class_solve_gevp to satisfy rho(ymax)=0
            #lhs[4*ny+mid_idx,:] = 0.0
            #rhs[4*ny+mid_idx  ] = 1.0 + 1.0*1j # 1.0*1j # 
            #lhs[4*ny+mid_idx, 4*ny+mid_idx] = 1.0

            
    def set_matrices_rayleigh_taylor_boussinesq_mixing(self, ny, bsfl, bsfl_ref, map):
        """
        Equations from Livescu Annual Review of Fluid Mechanics:
        "Turbulence with Large Thermal and Compositional Density Variations"
        see Equations 30, 31, 32
        variables are taken here as (u, v, w, p, rho)^T
        """

        # non-dimensional set of equations
        grav = bsfl_ref.gref/bsfl_ref.gref

        if ( grav != 1. ):
            print("")
            print("Boussinesq R-T solver: gravity should be non-dimensional, g = ", grav)
            input("Check gravity")
        
        Fr2      = bsfl_ref.Fr**2.
        Sc       = bsfl_ref.Sc
        Re       = bsfl_ref.Re

        D1 = map.D1
        D2 = map.D2

        check_norms_output = 0

        if (check_norms_output == 1):
            print("")
            print("R-T Boussinesq Equations (non-dimensional)")
            print("-----------------------------------------------------------------")
            print("Gravity                       = ", grav)
            print("Froude number Fr              = ", np.sqrt(Fr2))
            print("Schmidt number Sc             = ", Sc)
            print("Reynolds number Re            = ", Re)
            print("Peclet number (mass transfer) = ", Re*Sc)
            print("")
            
            
            mat_D1_norm = np.linalg.norm(D1)
            mat_D2_norm = np.linalg.norm(D2)
            
            print("Mat norm of D1: ", mat_D1_norm)
            print("Mat norm of D2: ", mat_D2_norm)
        
        # Identity matrix id
        id      = np.identity(ny)

        # Create diagonal matrices from baseflow vectors
        dRho    = np.diag(bsfl.Rho_nd)
        dRhop   = np.diag(bsfl.Rhop_nd)
        

        # 1st block indices
        imin = 0
        imax = ny

        self.mat_a1[imin:imax, imin+3*ny:imax+3*ny] = -1j*id

        self.mat_a2[imin:imax, imin:imax]           = -id/Re

        self.mat_b2[imin:imax, imin:imax]           = -id/Re

        self.mat_d1[imin:imax, imin:imax]           = D2/Re

        self.mat_rhs[imin:imax, imin:imax]          = -1j*id

        # 2nd block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -id/Re

        self.mat_b1[imin:imax, imin+2*ny:imax+2*ny] = -1j*id
                
        self.mat_b2[imin:imax, imin:imax]           = -id/Re

        self.mat_d1[imin:imax, imin:imax]           = D2/Re

        self.mat_rhs[imin:imax, imin:imax]          = -1j*id

        # 3rd block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -id/Re

        self.mat_b2[imin:imax, imin:imax]           = -id/Re

        self.mat_d1[imin:imax, imin:imax]           = D2/Re
        self.mat_d1[imin:imax, imin+1*ny:imax+1*ny] = -D1 
        self.mat_d1[imin:imax, imin+2*ny:imax+2*ny] = -grav*id/Fr2 

        self.mat_rhs[imin:imax, imin:imax]          = -1j*id

        # 4th block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a1[imin:imax, imin-3*ny:imax-3*ny] = 1j*id

        self.mat_b1[imin:imax, imin-2*ny:imax-2*ny] = 1j*id

        self.mat_d1[imin:imax, imin-1*ny:imax-1*ny] = D1

        # 5th block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -id/(Re*Sc)

        self.mat_b2[imin:imax, imin:imax]           = -id/(Re*Sc)

        self.mat_d1[imin:imax, imin:imax]           = D2/(Re*Sc) 
        self.mat_d1[imin:imax, imin-2*ny:imax-2*ny] = -dRhop

        self.mat_rhs[imin:imax, imin:imax]          = -1j*id

            
    def set_matrices_rayleigh_taylor_sandoval_equations(self, ny, bsfl, bsfl_ref, map):
        """
        Equations from Sndoval PhD Thesis:
        The Dynamics of Variable-Density Turbulence
        variables are taken here as (u, v, w, p, rho)^T
        """

        # Flag to add terms that arise when I do not neglect the w-vel of the baseflow
        AddWTerms = 0
        if (AddWTerms==1):
            print("==============================================================================")
            print("Adding w-vel baseflow terms in set_matrices_rayleigh_taylor_sandoval_equations")
            print("==============================================================================")
        else:
            print("==================================================================================")
            print("NOT Adding w-vel baseflow terms in set_matrices_rayleigh_taylor_sandoval_equations")
            print("==================================================================================")

        # non-dimensional set of equations
        grav = bsfl_ref.gref/bsfl_ref.gref

        if ( grav != 1. ):
            print("")
            print("Sandoval R-T solver: gravity should be non-dimensional, g = ", grav)
            input("Check gravity")

        Fr2      = bsfl_ref.Fr**2.
        Sc       = bsfl_ref.Sc
        Re       = bsfl_ref.Re

        # To match Chandrasekhar equations
        MatchChandrasekhar = 0
        
        SetDivergenceToZero = 0
        SetOtherReScTermsToZero = 0
        coef_zer = 1.
        
        if (MatchChandrasekhar==1): # we want to match data from solver boussinesq = -2
            
            # Note: for very large Scmidt number the right hand side of continuity and mass
            # equations are already automatically zero and therefor the 1/3 terms should be
            # zero too anyway
            print("Matching Chandrasekhar Equations...")
            SetDivergenceToZero = 1
            SetOtherReScTermsToZero = 1
            coef_zer = 0.

        # print("")
        # print('R-T "Sandoval" Equations (non-dimensional)')
        # print("-----------------------------------------------------------------")
        # print("Gravity                       = ", grav)
        # print("Froude number Fr              = ", np.sqrt(Fr2))
        # print("Schmidt number Sc             = ", Sc)
        # print("Reynolds number Re            = ", Re)
        # print("Peclet number (mass transfer) = ", Re*Sc)
        # print("")

        D1 = map.D1
        D2 = map.D2
        
        # Identity matrix id
        id      = np.identity(ny)

        # Create diagonal matrices from baseflow vectors
        dRho    = np.diag(bsfl.Rho_nd)
        dRhop   = np.diag(bsfl.Rhop_nd)
        dRhopp  = np.diag(bsfl.Rhopp_nd)

        rho2    = np.multiply(bsfl.Rho_nd, bsfl.Rho_nd)
        rho3    = np.multiply(rho2, bsfl.Rho_nd)

        #print("bsfl.Rho_nd = ", bsfl.Rho_nd)
        #print("rho2 = ", rho2)
        
        rho_inv  = 1/bsfl.Rho_nd
        rho2_inv = 1/rho2
        rho3_inv = 1/rho3

        drho_inv   = np.diag(rho_inv)
        drho2_inv  = np.diag(rho2_inv)
        drho3_inv  = np.diag(rho3_inv)

        if (AddWTerms==1):
            w2_bsfl = np.multiply(bsfl.Wvel_nd, bsfl.Wvel_nd)
            rhow_bsfl = np.multiply(bsfl.Rho_nd, bsfl.Wvel_nd)
            wdwdz_bsfl = np.multiply(bsfl.Wvel_nd, bsfl.Wvelp_nd)
            trm_wvel_bsfl = np.multiply(bsfl.Rho_nd, bsfl.Wvelp_nd) + np.multiply(bsfl.Wvel_nd, bsfl.Rhop_nd)
            # diagonal matrices
            dtrm_wvel_bsfl = np.diag(trm_wvel_bsfl)
            dwdwdz_bsfl = np.diag(wdwdz_bsfl)
            drhow_bsfl = np.diag(rhow_bsfl)
            dw2_bsfl = np.diag(w2_bsfl)
            dw_bsfl = np.diag(bsfl.Wvel_nd)

        ############################################
        # 1st block indices ==> x-momentum equation
        ############################################
        imin = 0
        imax = ny

        self.mat_a1[imin:imax, imin+2*ny:imax+2*ny] = coef_zer*1j/(3.*Re)*D1
        self.mat_a1[imin:imax, imin+3*ny:imax+3*ny] = -1j*id

        #self.mat_a2[imin:imax, imin:imax]           = -4.*id/(3.*Re)
        self.mat_a2[imin:imax, imin:imax]           = -id/(Re) - coef_zer*1.*id/(3.*Re)

        self.mat_b2[imin:imax, imin:imax]           = -id/Re

        self.mat_ab[imin:imax, imin+1*ny:imax+1*ny] = -coef_zer*id/(3.*Re)

        self.mat_d1[imin:imax, imin:imax]           = D2/Re

        self.mat_rhs[imin:imax, imin:imax]          = -1j*dRho

        if (AddWTerms==1):
            self.mat_d1[imin:imax, imin:imax] = self.mat_d1[imin:imax, imin:imax] -dtrm_wvel_bsfl
            self.mat_d1[imin:imax, imin:imax] = self.mat_d1[imin:imax, imin:imax] -np.matmul(drhow_bsfl, D1)

        ############################################
        # 2nd block indices ==> y-momentum equation
        ############################################
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -id/Re

        self.mat_b1[imin:imax, imin+1*ny:imax+1*ny] = coef_zer*1j/(3.*Re)*D1
        self.mat_b1[imin:imax, imin+2*ny:imax+2*ny] = -1j*id
                
        #self.mat_b2[imin:imax, imin:imax]           = -4.*id/(3.*Re)
        self.mat_b2[imin:imax, imin:imax]           = -id/(Re) -coef_zer*1.*id/(3.*Re)

        self.mat_ab[imin:imax, imin-1*ny:imax-1*ny] = -coef_zer*id/(3.*Re)

        self.mat_d1[imin:imax, imin:imax]           = D2/Re

        self.mat_rhs[imin:imax, imin:imax]          = -1j*dRho

        if (AddWTerms==1):
            self.mat_d1[imin:imax, imin:imax] = self.mat_d1[imin:imax, imin:imax] -dtrm_wvel_bsfl
            self.mat_d1[imin:imax, imin:imax] = self.mat_d1[imin:imax, imin:imax] -np.matmul(drhow_bsfl, D1)

        ############################################
        # 3rd block indices ==> z-momentum equation
        ############################################
        imin = imin + ny
        imax = imax + ny

        self.mat_a1[imin:imax, imin-2*ny:imax-2*ny] = coef_zer*1j/(3.*Re)*D1

        self.mat_a2[imin:imax, imin:imax]           = -id/Re

        self.mat_b1[imin:imax, imin-1*ny:imax-1*ny] = coef_zer*1j/(3.*Re)*D1

        self.mat_b2[imin:imax, imin:imax]           = -id/Re

        #self.mat_d1[imin:imax, imin:imax]           = 4.*D2/(3.*Re)
        self.mat_d1[imin:imax, imin:imax]           = D2/(Re) + coef_zer*1.*D2/(3.*Re)
        
        self.mat_d1[imin:imax, imin+1*ny:imax+1*ny] = -D1 
        self.mat_d1[imin:imax, imin+2*ny:imax+2*ny] = -grav*id/Fr2 

        self.mat_rhs[imin:imax, imin:imax]          = -1j*dRho

        if (AddWTerms==1):
            self.mat_a1[imin:imax, imin-2*ny:imax-2*ny] = self.mat_a1[imin:imax, imin-2*ny:imax-2*ny] -1j*drhow_bsfl
            self.mat_b1[imin:imax, imin-1*ny:imax-1*ny] = self.mat_b1[imin:imax, imin-1*ny:imax-1*ny] -1j*drhow_bsfl

            self.mat_d1[imin:imax, imin:imax] = self.mat_d1[imin:imax, imin:imax] -2.*np.matmul(drhow_bsfl, D1)
            self.mat_d1[imin:imax, imin:imax] = self.mat_d1[imin:imax, imin:imax] -2.*drhow_bsfl
            
            self.mat_d1[imin:imax, imin+2*ny:imax+2*ny] = self.mat_d1[imin:imax, imin+2*ny:imax+2*ny] -2.*dwdwdz_bsfl
            self.mat_d1[imin:imax, imin+2*ny:imax+2*ny] = self.mat_d1[imin:imax, imin+2*ny:imax+2*ny] -np.matmul(dw2_bsfl, D1)
            
            self.mat_rhs[imin:imax, imin+2*ny:imax+2*ny] = -1j*dw_bsfl

        # # To match Boussinesq === THIS DID NOT WORK
        # dRhop   = 0.0*dRhop
        # dRhopp   = 0.0*dRhopp
        # SetDivergenceToZero = 1

        ############################################################################
        # 4th block indices ==> equation: du/dx + dv/dy + dw/dz = 1/(Re*Sc)*( ... )
        ############################################################################
        imin = imin + ny
        imax = imax + ny

        self.mat_a1[imin:imax, imin-3*ny:imax-3*ny] = -1j*id

        self.mat_a2[imin:imax, imin+1*ny:imax+1*ny] = drho_inv/(Re*Sc)

        self.mat_b1[imin:imax, imin-2*ny:imax-2*ny] = -1j*id

        self.mat_b2[imin:imax, imin+1*ny:imax+1*ny] = drho_inv/(Re*Sc)

        self.mat_d1[imin:imax, imin-1*ny:imax-1*ny] = -D1

        mat_tmp1 = 2.*np.matmul(drho2_inv, np.matmul(dRhop,D1))
        mat_tmp2 = np.matmul(drho2_inv, dRhopp)
        mat_tmp3 = -np.matmul(drho_inv, D2)
        mat_tmp4 = -2.*np.matmul(drho3_inv, np.matmul(dRhop,dRhop))

        # BUGGY ===============> #self.mat_d1[imin:imax, imin+1*ny:imax+1*ny] = ( 2.*drho2_inv*dRhop*D1 + drho2_inv*dRhopp - drho_inv*D2 -2.*drho3_inv*dRhop*dRhop )/(Re*Sc)
        self.mat_d1[imin:imax, imin+1*ny:imax+1*ny] = ( mat_tmp1 + mat_tmp2 + mat_tmp3 + mat_tmp4 )/(Re*Sc)

        # Match Chandrasekhar Equations Trick
        if (SetDivergenceToZero==1):
            self.mat_a2[imin:imax, imin+1*ny:imax+1*ny] = 0.0
            self.mat_b2[imin:imax, imin+1*ny:imax+1*ny] = 0.0
            self.mat_d1[imin:imax, imin+1*ny:imax+1*ny] = 0.0

        ###################################################################################
        # 5th block indices ==> equation for mass: drho/dt + u_j*drho/dx_j + rho*du_j/dx_j
        ###################################################################################
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -id/(Re*Sc)

        self.mat_b2[imin:imax, imin:imax]           = -id/(Re*Sc)

        # BUGGY ================> self.mat_d1[imin:imax, imin:imax]           = ( D2 + drho2_inv*dRhop*dRhop - 2.*drho_inv*dRhop*D1 )/(Re*Sc)
        mat_tmp5 = np.matmul(drho2_inv, np.matmul(dRhop,dRhop))
        mat_tmp6 = -2.*np.matmul(drho_inv, np.matmul(dRhop,D1))
                    
        self.mat_d1[imin:imax, imin:imax]           = ( D2 + mat_tmp5 + mat_tmp6 )/(Re*Sc)
        
        self.mat_d1[imin:imax, imin-2*ny:imax-2*ny] = -dRhop

        self.mat_rhs[imin:imax, imin:imax]          = -1j*id

        if (AddWTerms==1):
            self.mat_d1[imin:imax, imin:imax] = self.mat_d1[imin:imax, imin:imax] -np.matmul(dw_bsfl, D1)

        # Match Chandrasekhar Equations Trick
        if (SetOtherReScTermsToZero == 1):
            self.mat_a2[imin:imax, imin:imax]           = 0.0
            self.mat_b2[imin:imax, imin:imax]           = 0.0
            self.mat_d1[imin:imax, imin:imax]           = 0.0
            
        
    def call_to_build_matrices(self, rt_flag, prim_form, ny, bsfl, bsfl_ref, Re, map, alpha_in):

        # Call proper matrix building
        if (rt_flag == True):
            if (prim_form==1):
                #print("in class_build_matrices: rt_flag and prim_form TRUE")
                if   ( self.boussinesq == 1 ):
                    self.set_matrices_rayleigh_taylor_boussinesq_mixing(ny, bsfl, bsfl_ref, map)
                if   ( self.boussinesq == 10 ):
                    self.set_matrices_rayleigh_taylor_boussinesq_mixing_dimensional(ny, bsfl, bsfl_ref, map)
                elif ( self.boussinesq == -2 ):
                    self.set_matrices_rayleigh_taylor_chandrasekhar(ny, bsfl, bsfl_ref, map)
                elif ( self.boussinesq == -20 ):
                    self.set_matrices_rayleigh_taylor_chandrasekhar_non_dimensional(ny, bsfl, bsfl_ref, map)
                elif ( self.boussinesq == -3 ):
                    self.set_matrices_rayleigh_taylor_sandoval_equations(ny, bsfl, bsfl_ref, map)
                elif ( self.boussinesq == -4 ):
                    self.set_matrices_rayleigh_taylor_sandoval_equations_VAR_MU(ny, bsfl, bsfl_ref, map)
                elif ( self.boussinesq == -555 ):
                    self.set_matrices_rayleigh_taylor_compressible_inviscid(ny, bsfl, bsfl_ref, map, alpha_in)
            
            else:
                self.set_matrices_rayleigh_taylor_inviscid_w_equation(ny, bsfl, map)
        else:
            self.set_matrices(ny, Re, bsfl, map)

            
    def call_to_set_bc(self, rt_flag, prim_form, ny, map):

        if (rt_flag):
            if (prim_form==1):
                if   ( self.boussinesq == -555 ):
                    self.set_bc_rayleigh_taylor_compressible_inviscid(self.mat_lhs, self.mat_rhs, ny, map)
                else:
                    self.set_bc_rayleigh_taylor(self.mat_lhs, self.mat_rhs, ny, map)
            else:
                self.set_bc_rayleigh_taylor_inviscid(self.mat_lhs, self.mat_rhs, ny, map)
        else:
            self.set_bc_shear_layer(self.mat_lhs, self.mat_rhs, ny, map)            





    def set_matrices_rayleigh_taylor_sandoval_equations_VAR_MU(self, ny, bsfl, bsfl_ref, map):
        """
        Modified Sandoval Equations (variable mu)
        """

        # non-dimensional set of equations
        grav = bsfl_ref.gref/bsfl_ref.gref

        if ( grav != 1. ):
            print("")
            print("Modified Sandoval R-T solver: gravity should be non-dimensional, g = ", grav)
            input("Check gravity")

        Fr2      = bsfl_ref.Fr**2.
        Sc       = bsfl_ref.Sc
        Re       = bsfl_ref.Re

        # To match Chandrasekhar equations
        MatchChandrasekhar = 0

        r13 = 1./3.
        r23 = 2./3.

        SetDivergenceToZero = 0
        SetOtherReScTermsToZero = 0
        
        if (MatchChandrasekhar==1): # we want to match data from solver boussinesq = -2

            r13 = 0.
            r23 = 0.

            print("Modified Sandoval Equations: Matching Chandrasekhar Equations...")
            SetDivergenceToZero = 1
            SetOtherReScTermsToZero = 1

            
        D1 = map.D1
        D2 = map.D2
        
        # Identity matrix id
        id      = np.identity(ny)

        # Create diagonal matrices from baseflow vectors
        dRho    = np.diag(bsfl.Rho_nd)
        dRhop   = np.diag(bsfl.Rhop_nd)
        dRhopp  = np.diag(bsfl.Rhopp_nd)

        dMu     = np.diag(bsfl.Mu_nd)
        dMup    = np.diag(bsfl.Mup_nd)

        rho2    = np.multiply(bsfl.Rho_nd, bsfl.Rho_nd)
        rho3    = np.multiply(rho2, bsfl.Rho_nd)

        #print("bsfl.Rho_nd = ", bsfl.Rho_nd)
        #print("rho2 = ", rho2)
        
        rho_inv  = 1/bsfl.Rho_nd
        rho2_inv = 1/rho2
        rho3_inv = 1/rho3

        drho_inv   = np.diag(rho_inv)
        drho2_inv  = np.diag(rho2_inv)
        drho3_inv  = np.diag(rho3_inv)


        # 1st block indices
        imin = 0
        imax = ny

        self.mat_a1[imin:imax, imin+2*ny:imax+2*ny] = r13*1j*np.matmul(dMu,D1)/Re + 1j*dMup/Re
        self.mat_a1[imin:imax, imin+3*ny:imax+3*ny] = -1j*id

        self.mat_a2[imin:imax, imin:imax]           = -dMu/Re -r13*dMu/Re

        self.mat_b2[imin:imax, imin:imax]           = -dMu/Re

        self.mat_ab[imin:imax, imin+1*ny:imax+1*ny] = -r13*dMu/Re

        self.mat_d1[imin:imax, imin:imax]           = np.matmul(dMu,D2)/Re + np.matmul(dMup,D1)/Re
        
        self.mat_rhs[imin:imax, imin:imax]          = -1j*dRho

        # 2nd block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -dMu/Re -r13*dMu/Re

        self.mat_b1[imin:imax, imin+1*ny:imax+1*ny] = r13*1j*np.matmul(dMu,D1)/Re + 1j*dMup/Re
        self.mat_b1[imin:imax, imin+2*ny:imax+2*ny] = -1j*id
                
        self.mat_b2[imin:imax, imin:imax]           = -dMu/Re - r13*dMu/Re

        self.mat_ab[imin:imax, imin-1*ny:imax-1*ny] = -r13*dMu/Re

        self.mat_d1[imin:imax, imin:imax]           = np.matmul(dMu,D2)/Re + np.matmul(dMup,D1)/Re

        self.mat_rhs[imin:imax, imin:imax]          = -1j*dRho

        # 3rd block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a1[imin:imax, imin-2*ny:imax-2*ny] = r13*1j*np.matmul(dMu,D1)/Re - r23*1j*dMup/Re

        self.mat_a2[imin:imax, imin:imax]           = -dMu/Re

        self.mat_b1[imin:imax, imin-1*ny:imax-1*ny] = r13*1j*np.matmul(dMu,D1)/Re - r23*1j*dMup/Re

        self.mat_b2[imin:imax, imin:imax]           = -dMu/Re

        self.mat_d1[imin:imax, imin:imax]           = np.matmul(dMu,D2)/Re + r13*np.matmul(dMu,D2)/Re + 2.*np.matmul(dMup,D1)/Re - r23*np.matmul(dMup,D1)/Re
        
        self.mat_d1[imin:imax, imin+1*ny:imax+1*ny] = -D1 
        self.mat_d1[imin:imax, imin+2*ny:imax+2*ny] = -grav*id/Fr2 

        self.mat_rhs[imin:imax, imin:imax]          = -1j*dRho

        # 4th block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a1[imin:imax, imin-3*ny:imax-3*ny] = -1j*id

        self.mat_a2[imin:imax, imin+1*ny:imax+1*ny] = drho_inv/(Re*Sc)

        self.mat_b1[imin:imax, imin-2*ny:imax-2*ny] = -1j*id

        self.mat_b2[imin:imax, imin+1*ny:imax+1*ny] = drho_inv/(Re*Sc)

        self.mat_d1[imin:imax, imin-1*ny:imax-1*ny] = -D1

        mat_tmp1 = 2.*np.matmul(drho2_inv, np.matmul(dRhop,D1))
        mat_tmp2 = np.matmul(drho2_inv, dRhopp)
        mat_tmp3 = -np.matmul(drho_inv, D2)
        mat_tmp4 = -2.*np.matmul(drho3_inv, np.matmul(dRhop,dRhop))

        # BUGGY ===============> #self.mat_d1[imin:imax, imin+1*ny:imax+1*ny] = ( 2.*drho2_inv*dRhop*D1 + drho2_inv*dRhopp - drho_inv*D2 -2.*drho3_inv*dRhop*dRhop )/(Re*Sc)
        self.mat_d1[imin:imax, imin+1*ny:imax+1*ny] = ( mat_tmp1 + mat_tmp2 + mat_tmp3 + mat_tmp4 )/(Re*Sc)

        # Match Chandrasekhar Equations Trick
        if (SetDivergenceToZero==1):
            self.mat_a2[imin:imax, imin+1*ny:imax+1*ny] = 0.0
            self.mat_b2[imin:imax, imin+1*ny:imax+1*ny] = 0.0
            self.mat_d1[imin:imax, imin+1*ny:imax+1*ny] = 0.0

        # 5th block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -id/(Re*Sc)

        self.mat_b2[imin:imax, imin:imax]           = -id/(Re*Sc)

        # BUGGY ================> self.mat_d1[imin:imax, imin:imax]           = ( D2 + drho2_inv*dRhop*dRhop - 2.*drho_inv*dRhop*D1 )/(Re*Sc)
        mat_tmp5 = np.matmul(drho2_inv, np.matmul(dRhop,dRhop))
        mat_tmp6 = -2.*np.matmul(drho_inv, np.matmul(dRhop,D1))
                    
        self.mat_d1[imin:imax, imin:imax]           = ( D2 + mat_tmp5 + mat_tmp6 )/(Re*Sc)
        
        self.mat_d1[imin:imax, imin-2*ny:imax-2*ny] = -dRhop

        self.mat_rhs[imin:imax, imin:imax]          = -1j*id

        # Match Chandrasekhar Equations Trick
        if (SetOtherReScTermsToZero == 1):
            self.mat_a2[imin:imax, imin:imax]           = 0.0
            self.mat_b2[imin:imax, imin:imax]           = 0.0
            self.mat_d1[imin:imax, imin:imax]           = 0.0


    def set_matrices_rayleigh_taylor_compressible_inviscid(self, ny, bsfl, bsfl_ref, map, alpha_in): # here I pas ny
        """
        This system is build for a vertical interface with heavy fluid on left side, light fluid on right side,
        x-axis towards the right and gravity towards the right
        """
        Gamma = 5./3.

        # Create diagonal matrices from baseflow vectors
        dRho    = np.diag(bsfl.Rho_nd)
        dRhop   = np.diag(bsfl.Rhop_nd)

        dPres   = np.diag(bsfl.Pres_nd)
        
        # Identity matrix id
        id      = np.identity(ny)

        # First derivative matrix
        D1      = map.D1
        
        # non-dimensional set of equations
        grav = bsfl_ref.gref/bsfl_ref.gref

        if ( grav != 1. ):
            print("")
            print("Compressible inviscid solver: gravity should be non-dimensional, g = ", grav)
            input("Check gravity")

        opt_real = 0

        if (opt_real==1):
            print("")
            print("opt_real is set to 1 in set_matrices_rayleigh_taylor_compressible_inviscid")
            print("")

        # 1st block indices
        imin = 0
        imax = ny

        self.mat_a1[imin:imax, imin+2*ny:imax+2*ny] = -grav*id
        self.mat_a1[imin:imax, imin+3*ny:imax+3*ny] = D1

        if (opt_real==1):
            self.mat_rhs[imin:imax, imin:imax]          = dRho
        else:
            self.mat_rhs[imin:imax, imin:imax]          = 1j*dRho

        # 2nd block indices
        imin = imin + ny
        imax = imax + ny

        if (opt_real==1):
            self.mat_a1[imin:imax, imin+2*ny:imax+2*ny] = alpha_in*id
        else:
            self.mat_a1[imin:imax, imin+2*ny:imax+2*ny] = 1j*alpha_in*id

        if (opt_real==1):
            self.mat_rhs[imin:imax, imin:imax]          = dRho
        else:
            self.mat_rhs[imin:imax, imin:imax]          = 1j*dRho

        # 3rd block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a1[imin:imax, imin-2*ny:imax-2*ny] = dRhop + np.matmul(dRho, D1)
        if (opt_real==1):
            self.mat_a1[imin:imax, imin-1*ny:imax-1*ny] = alpha_in*dRho
        else:
            self.mat_a1[imin:imax, imin-1*ny:imax-1*ny] = 1j*alpha_in*dRho

        if (opt_real==1):
            self.mat_rhs[imin:imax, imin:imax]          = id
        else:
            self.mat_rhs[imin:imax, imin:imax]          = 1j*id

        # 4th block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a1[imin:imax, imin-3*ny:imax-3*ny] = dRho + Gamma*np.matmul(dPres, D1)

        if (opt_real==1):
            self.mat_a1[imin:imax, imin-2*ny:imax-2*ny] = alpha_in*Gamma*dPres
        else:
            self.mat_a1[imin:imax, imin-2*ny:imax-2*ny] = 1j*alpha_in*Gamma*dPres

        if (opt_real==1):
            self.mat_rhs[imin:imax, imin:imax]          = id
        else:
            self.mat_rhs[imin:imax, imin:imax]          = 1j*id
            

    def set_bc_rayleigh_taylor_compressible_inviscid(self, lhs, rhs, ny, map):
        """
        """
        ######################################
        # (1) FREESTREAM BOUNDARY-CONDITIONS #
        ######################################
        
        ##################
        # u-velocity BCs #
        ##################
        idx_u_ymin = 0*ny
        idx_u_ymax = 1*ny-1

        # ymin
        lhs[idx_u_ymin,:] = 0.0
        rhs[idx_u_ymin,:] = 0.0        
        lhs[idx_u_ymin, idx_u_ymin] = 1.0

        # ymax
        lhs[idx_u_ymax,:] = 0.0
        rhs[idx_u_ymax,:] = 0.0
        lhs[idx_u_ymax, idx_u_ymax] = 1.0
        
        ##################
        # v-velocity BCs #
        ##################
        idx_v_ymin = 1*ny
        idx_v_ymax = 2*ny-1

        # ymin
        lhs[idx_v_ymin,:] = 0.0
        rhs[idx_v_ymin,:] = 0.0        
        lhs[idx_v_ymin, idx_v_ymin] = 1.0

        # ymax
        lhs[idx_v_ymax,:] = 0.0
        rhs[idx_v_ymax,:] = 0.0
        lhs[idx_v_ymax, idx_v_ymax] = 1.0

        ##################
        # density BCs   #
        ##################
        set_dens = 1

        if (set_dens==1):
            idx_r_ymin = 2*ny
            idx_r_ymax = 3*ny-1
            
            # ymin
            lhs[idx_r_ymin,:] = 0.0
            rhs[idx_r_ymin  ] = 0.0        
            lhs[idx_r_ymin, idx_r_ymin] = 1.0
            
            # ymax
            lhs[idx_r_ymax,:] = 0.0
            rhs[idx_r_ymax  ] = 0.0
            lhs[idx_r_ymax, idx_r_ymax] = 1.0

        ##################
        # pressure BCs #
        ##################
        set_pres = 1

        if (set_pres==1):
            idx_p_ymin = 3*ny
            idx_p_ymax = 4*ny-1
            
            # ymin
            lhs[idx_p_ymin,:] = 0.0
            rhs[idx_p_ymin  ] = 0.0        
            lhs[idx_p_ymin, idx_p_ymin] = 1.0
            
            # ymax
            lhs[idx_p_ymax,:] = 0.0
            rhs[idx_p_ymax  ] = 0.0
            lhs[idx_p_ymax, idx_p_ymax] = 1.0


    def set_matrices_rayleigh_taylor_boussinesq_mixing_dimensional(self, ny, bsfl, bsfl_ref, map):
        """
        """
        nu = bsfl_ref.nuref # 2.54841997961264e-06
        Diff = bsfl_ref.Dref #bsfl_ref.nuref # 2.54841997961264e-06
        rho0 = bsfl_ref.rhoref # 1.0526315789473686

        #nu = bsfl_ref.nuref
        #rho0 = bsfl_ref.rhoref
        
        # dimensional set of equations
        grav = bsfl_ref.gref

        if ( grav != 9.81 ):
            print("")
            print("Boussinesq R-T dimensional solver: gravity should be dimensional, g = ", grav)
            input("Check gravity")
        
        D1 = map.D1_dim
        D2 = map.D2_dim
        
        # Identity matrix id
        id      = np.identity(ny)

        # Create diagonal matrices from baseflow vectors
        dRho    = np.diag(bsfl.Rho)
        dRhop   = np.diag(bsfl.Rhop)
        
        # 1st block indices
        imin = 0
        imax = ny

        self.mat_a1[imin:imax, imin+3*ny:imax+3*ny] = -1j*id/rho0

        self.mat_a2[imin:imax, imin:imax]           = -id*nu

        self.mat_b2[imin:imax, imin:imax]           = -id*nu

        self.mat_d1[imin:imax, imin:imax]           = D2*nu

        self.mat_rhs[imin:imax, imin:imax]          = -1j*id

        # 2nd block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -id*nu

        self.mat_b1[imin:imax, imin+2*ny:imax+2*ny] = -1j*id/rho0
                
        self.mat_b2[imin:imax, imin:imax]           = -id*nu

        self.mat_d1[imin:imax, imin:imax]           = D2*nu

        self.mat_rhs[imin:imax, imin:imax]          = -1j*id

        # 3rd block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -id*nu

        self.mat_b2[imin:imax, imin:imax]           = -id*nu

        self.mat_d1[imin:imax, imin:imax]           = D2*nu
        self.mat_d1[imin:imax, imin+1*ny:imax+1*ny] = -D1/rho0
        self.mat_d1[imin:imax, imin+2*ny:imax+2*ny] = -grav*id/rho0

        self.mat_rhs[imin:imax, imin:imax]          = -1j*id

        # 4th block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a1[imin:imax, imin-3*ny:imax-3*ny] = 1j*id

        self.mat_b1[imin:imax, imin-2*ny:imax-2*ny] = 1j*id

        self.mat_d1[imin:imax, imin-1*ny:imax-1*ny] = D1

        # 5th block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -id*Diff

        self.mat_b2[imin:imax, imin:imax]           = -id*Diff

        self.mat_d1[imin:imax, imin:imax]           = D2*Diff 
        self.mat_d1[imin:imax, imin-2*ny:imax-2*ny] = -dRhop

        self.mat_rhs[imin:imax, imin:imax]          = -1j*id

        check_norms = 0

        if (check_norms == 1):

            mat_a1_norm = np.linalg.norm(self.mat_a1)
            mat_a2_norm = np.linalg.norm(self.mat_a2)
            
            mat_b1_norm = np.linalg.norm(self.mat_b1)
            mat_b2_norm = np.linalg.norm(self.mat_b2)
            
            mat_d1_norm = np.linalg.norm(self.mat_d1)
            
            mat_rhs_norm = np.linalg.norm(self.mat_rhs)

            print("Dim: norm a1 = ", mat_a1_norm)
            print("Dim: norm a2 = ", mat_a2_norm)
            
            print("Dim: norm b1 = ", mat_b1_norm)
            print("Dim: norm b2 = ", mat_b2_norm)
            
            print("Dim: norm d1 = ", mat_d1_norm)
            print("Dim: norm rhs = ", mat_rhs_norm)

            print("")
            input("Checking matrices norms for dimensional Boussinesq solver")


    def set_matrices_rayleigh_taylor_chandrasekhar_non_dimensional(self, ny, bsfl, bsfl_ref, map): # here I pas ny
        """
        """
        # Identity matrix id
        id      = np.identity(ny)

        Fr2      = bsfl_ref.Fr**2.
        Re       = bsfl_ref.Re
        
        # Create diagonal matrices from baseflow vectors
        dMu     = np.diag(bsfl.Mu_nd)
        dRho    = np.diag(bsfl.Rho_nd)

        dMup    = np.diag(bsfl.Mup_nd)
        dRhop   = np.diag(bsfl.Rhop_nd)

        Lder    = np.matmul(dMup, map.D1_nondim)
        Lder2   = np.matmul(dMu, map.D2_nondim)

        grav    = bsfl_ref.gref/bsfl_ref.gref

        if ( grav != 1.0 ):
            print("")
            print("Chandrasekhar R-T solver NON-DIMENSIONAL, gravity should be non-dimensional, g = ", grav)
            input("Check gravity")

        # 1st block indices
        imin = 0
        imax = ny

        self.mat_a1[imin:imax, imin+2*ny:imax+2*ny] = 1j*dMup/Re
        self.mat_a1[imin:imax, imin+3*ny:imax+3*ny] = -1j*id

        self.mat_a2[imin:imax, imin:imax]           = -dMu/Re

        self.mat_b2[imin:imax, imin:imax]           = -dMu/Re

        self.mat_d1[imin:imax, imin:imax]           = Lder2/Re + Lder/Re

        self.mat_rhs[imin:imax, imin:imax]          = -1j*dRho

        # 2nd block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -dMu/Re

        self.mat_b1[imin:imax, imin+1*ny:imax+1*ny] = 1j*dMup/Re
        self.mat_b1[imin:imax, imin+2*ny:imax+2*ny] = -1j*id
                
        self.mat_b2[imin:imax, imin:imax]           = -dMu/Re

        self.mat_d1[imin:imax, imin:imax]           = Lder2/Re + Lder/Re

        self.mat_rhs[imin:imax, imin:imax]          = -1j*dRho

        # 3rd block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a2[imin:imax, imin:imax]           = -dMu/Re

        self.mat_b2[imin:imax, imin:imax]           = -dMu/Re

        self.mat_d1[imin:imax, imin:imax]           = Lder2/Re + 2*Lder/Re
        self.mat_d1[imin:imax, imin+1*ny:imax+1*ny] = -map.D1_nondim 
        self.mat_d1[imin:imax, imin+2*ny:imax+2*ny] = -id/Fr2 

        self.mat_rhs[imin:imax, imin:imax]          = -1j*dRho

        # 4th block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_a1[imin:imax, imin-3*ny:imax-3*ny] = 1j*id

        self.mat_b1[imin:imax, imin-2*ny:imax-2*ny] = 1j*id

        self.mat_d1[imin:imax, imin-1*ny:imax-1*ny] = map.D1_nondim

        # 5th block indices
        imin = imin + ny
        imax = imax + ny

        self.mat_d1[imin:imax, imin-2*ny:imax-2*ny] = -dRhop

        self.mat_rhs[imin:imax, imin:imax]          = -1j*id        
