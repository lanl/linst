
import numpy as np

dp = np.dtype('d')  # double precision
i4 = np.dtype('i4') # integer 4
i8 = np.dtype('i8') # integer 8

# class Mapping:
#     """
#     This class define a general mapping object. The Gauss-Lobatto points are defined between 1 and -1 and therefore,
#     a mapping is generally required to map the interval [1,-1] to the physical domain of interest
#     """
#     def __init__(self, size):
#         self.z    = np.zeros(size, dp)

class MapShearLayer(object):
    def __init__(self, sinf, cheb, l): # passing zinf (sinf), Gauss-Lobatto points (zi), l and differentiation matrices on [1,-1]
        """
        Mapping for shear-layer flows: From [-1,1] -> [zmin,zmax]
        """
        self.l = l
        zi = cheb.xc
        DM = cheb.DM
        self.z = np.zeros(shape=zi.shape)
        sn = (l/sinf)**2.

        npts    = len(zi)
        dzidz   = np.zeros(npts, dp)
        d2zidz2 = np.zeros(npts, dp)
        
        for i in range(len(zi)):
            #print("i=",i)
            self.z[i] = -l*zi[i]/np.sqrt(1.+sn-zi[i]**2.)
        
            dzidz[i]   = -np.sqrt(1.+sn)*l**2./(self.z[i]**2.+l**2.)**(3./2.)
            d2zidz2[i] = 3*self.z[i]*l**2.*np.sqrt(1.+sn)/(self.z[i]**2.+l**2.)**(5./2.)

        #print("dzidz**2. = ",dzidz**2.)
    
        #Scale the differentiation matrix (due to the use of algebraic mapping)
        self.D1 = np.matmul(np.diag(dzidz), DM[:,:,0])
        self.D2 = np.matmul(np.diag(d2zidz2), DM[:,:,0]) + np.matmul(np.diag(dzidz**2.), DM[:,:,1])

        # if ( boussinesq == -2 or boussinesq == 10 ): # dimensional solver
        #     # Only for solver boussinesq == -2
        #     self.D1_nondim = bsfl_ref.Lref*self.D1
        #     self.D2_nondim = bsfl_ref.Lref**2.*self.D2

        #     self.D1_dim = self.D1
        #     self.D2_dim = self.D2
        # else:
        #     self.D1_nondim = self.D1
        #     self.D2_nondim = self.D2

        #     self.D1_dim = 1/bsfl_ref.Lref*self.D1
        #     self.D2_dim = 1/bsfl_ref.Lref*self.D2

        #return self.z, self.D1, self.D2


class MapVoid(object):
    def __init__(self, sinf, cheb, l): # void mapping keep matrices and z as is
        """
        Mapping for Poiseuille flow ==> no mapping
        """

        self.l = l
        self.z  = cheb.xc
        
        #Scale the differentiation matrix (due to the use of algebraic mapping)
        self.D1 = cheb.DM[:,:,0]
        self.D2 = cheb.DM[:,:,1]
