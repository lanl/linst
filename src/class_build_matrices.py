
#from termios import N_TTY
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
    def __init__(self, size): # here I pass 4*ny
        """
        Constructor for class BuilMatrices
        """
        self.size = size
        # Initialize arrays
        self.mat_lhs = np.zeros((size,size), dpc)
        self.mat_rhs = np.zeros((size,size), dpc)

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

            if (ny % 2) != 0:
                mid_idx = int( (ny-1)/2 )
            else:
                warnings.warn("When ny is even, there is no mid-index")
                mid_idx = int ( ny/2 )

            print("mid_idx=",mid_idx)
            
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

    def set_matrices(self, ny, Re, alpha, beta, bsfl, map): # here I pas ny
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

        mat_a1 = np.zeros((nt, nt), dpc)
        mat_a2 = np.zeros((nt, nt), dpc)
    
        mat_b1 = np.zeros((nt, nt), dpc)
        mat_b2 = np.zeros((nt, nt), dpc)

        mat_d1 = np.zeros((nt, nt), dpc)

        # 1st block indices
        imin = 0
        imax = ny

        mat_a1[imin:imax, imin:imax]           = 1j*dU
        mat_a1[imin:imax, imin+3*ny:imax+3*ny] = 1j*id

        mat_a2[imin:imax, imin:imax]           = id_r_re

        mat_b1[imin:imax, imin:imax]           = 1j*dW
        mat_b2[imin:imax, imin:imax]           = id_r_re

        mat_d1[imin:imax, imin:imax]           = -map.D2/Re
        mat_d1[imin:imax, imin+ny:imax+ny]     = dUp

        self.mat_rhs[imin:imax, imin:imax]     = 1j*id

        # 2nd block indices
        imin = imin + ny
        imax = imax + ny

        mat_a1[imin:imax, imin:imax]           = 1j*dU
        mat_a2[imin:imax, imin:imax]           = id_r_re

        mat_b1[imin:imax, imin:imax]           = 1j*dW
        mat_b2[imin:imax, imin:imax]           = id_r_re

        mat_d1[imin:imax, imin:imax]           = -map.D2/Re
        mat_d1[imin:imax, imin+2*ny:imax+2*ny] = map.D1

        self.mat_rhs[imin:imax, imin:imax]     = 1j*id

        # 3rd block indices
        imin = imin + ny
        imax = imax + ny

        mat_a1[imin:imax, imin:imax]       = 1j*dU
        mat_a2[imin:imax, imin:imax]       = id_r_re

        mat_b1[imin:imax, imin:imax]       = 1j*dW
        mat_b1[imin:imax, imin+ny:imax+ny] = 1j*id

        mat_b2[imin:imax, imin:imax]       = id_r_re

        mat_d1[imin:imax, imin:imax]       = -map.D2/Re
        mat_d1[imin:imax, imin-ny:imax-ny] = dWp

        #print("imin = ", imin)
        #print("mat_d1[imin,:] = ",mat_d1[imin,:])

        #input('checking mat_d1')

        self.mat_rhs[imin:imax, imin:imax] = 1j*id   

        # 4th block indices
        imin = imin + ny
        imax = imax + ny

        mat_a1[imin:imax, imin-3*ny:imax-3*ny] = 1j*id

        mat_b1[imin:imax, imin-ny:imax-ny]     = 1j*id

        mat_d1[imin:imax, imin-2*ny:imax-2*ny] = map.D1  
    
        #print("")
        #print("self.mat_rhs = ", self.mat_rhs)

        self.mat_lhs = alpha*mat_a1 + alpha**2.*mat_a2 + beta*mat_b1 + beta**2.*mat_b2 + mat_d1
    

    
