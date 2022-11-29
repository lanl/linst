
#from termios import N_TTY
import module_utilities as mod_util
import numpy as np

import warnings

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8


class BuildMatrices:
    """
    This class builds the main matrices A (mat_lhs) and B (mat_rhs) 
    required to form the temporal generalized eigenvalue problem A*q = omega*B*q
    """
    def __init__(self, size): # here I pass 4*ny for regular Navier-Stokes Stability and 5*ny for "Rayleigh-Taylor" Navier-Stokes
        """
        Constructor for class BuilMatrices
        """
        self.size = size
        # Initialize arrays
        self.mat_a1 = np.zeros((size,size), dpc)
        self.mat_a2 = np.zeros((size,size), dpc)
        self.mat_b1 = np.zeros((size,size), dpc)
        self.mat_b2 = np.zeros((size,size), dpc)
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
    
    def assemble_mat_lhs(self, alpha, beta, omega, Tracking, Local):
        """
        This function assemble the lhs matrix mat_lhs
        """
        self.mat_lhs = alpha*self.mat_a1 + alpha**2.*self.mat_a2 + beta*self.mat_b1 + beta**2.*self.mat_b2 + self.mat_d1
        
        if (Tracking and Local):
            self.mat_lhs = self.mat_lhs - omega*self.mat_rhs
            
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


    def set_matrices_rayleigh_taylor(self, ny, bsfl, map): # here I pas ny
        """
        """
        # Identity matrix id
        id      = np.identity(ny)

        # Create diagonal matrices from baseflow vectors
        dMu     = np.diag(bsfl.Mu)
        dRho    = np.diag(bsfl.Rho)

        dMup    = np.diag(bsfl.Mup)
        dRhop   = np.diag(bsfl.Rhop)

        Lder    = np.matmul(dMup, map.D1)
        Lder2   = np.matmul(dMu, map.D2)

        grav    = 9.81

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

        self.mat_d1[imin:imax, imin:imax]           = Lder2 + Lder
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
            lhs[idx_p_ymin, idx_p_ymin:4*ny] = map.D1[0, :]
            
            # ymax
            lhs[idx_p_ymax,:] = 0.0
            rhs[idx_p_ymax  ] = 0.0
            lhs[idx_p_ymax, idx_p_ymin:4*ny] = map.D1[-1, :]

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

        grav    = 9.81

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
        
        ##################
        # u-velocity BCs #
        ##################
        idx_u_ymin = 0*ny # replace this by rho=1 at midplane
        idx_u_ymax = 1*ny-1

        # ymin
        #lhs[idx_u_ymin,:] = 0.0
        #rhs[idx_u_ymin,:] = 0.0        
        #lhs[idx_u_ymin, idx_u_ymin] = 1.0

        mid_idx = mod_util.get_mid_idx(ny)

        # I checked from eigenfunctions from global solver:
        # at y = 0, the real and imaginary parts of density are nonzero, that is why I am setting rho = 1 + 1j
        lhs[idx_u_ymin,:] = 0.0
        rhs[idx_u_ymin  ] = 1.0 + 1.0*1j
        lhs[idx_u_ymin, 4*ny+mid_idx] = 1.0

        #lhs[4*ny+mid_idx,:] = 0.0
        #rhs[4*ny+mid_idx  ] = 1.0 + 1.0*1j
        #lhs[4*ny+mid_idx, 4*ny+mid_idx] = 1.0

        
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
        set_pres = 1

        if (set_pres==1):
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
            
        
