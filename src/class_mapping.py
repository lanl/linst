
import numpy as np

dp = np.dtype('d')  # double precision
i4 = np.dtype('i4') # integer 4
i8 = np.dtype('i8') # integer 8

class Mapping:
    """
    This class define a general mapping object. The Gauss-Lobatto points are defined between 1 and -1 and therefore,
    a mapping is generally required to map the interval [1,-1] to the physical domain of interest
    """
    def __init__(self, size):
        self.size = size
        self.D1   = np.zeros((size,size), dp)
        self.D2   = np.zeros((size,size), dp)
        self.y    = np.zeros(size, dp)

    def map_shear_layer(self, sinf, yi, l, DM): # passing yinf (sinf), Gauss-Lobatto points (yi), l and differentiation matrices on [1,-1]
        """
        Mapping for shear-layer flows: From [-1,1] -> [ymin,ymax]
        """                
        sn = (l/sinf)**2.

        npts    = len(yi)
        dyidy   = np.zeros(npts, dp)
        d2yidy2 = np.zeros(npts, dp)
        
        for i in range(len(yi)):
            #print("i=",i)
            self.y[i] = -l*yi[i]/np.sqrt(1.+sn-yi[i]**2.)
        
            dyidy[i]   = -np.sqrt(1.+sn)*l**2./(self.y[i]**2.+l**2.)**(3./2.)
            d2yidy2[i] = 3*self.y[i]*l**2.*np.sqrt(1.+sn)/(self.y[i]**2.+l**2.)**(5./2.)

        #print("dyidy**2. = ",dyidy**2.)
    
        #Scale the differentiation matrix (due to the use of algebraic mapping)
        self.D1 = np.matmul(np.diag(dyidy), DM[:,:,0])
        self.D2 = np.matmul(np.diag(d2yidy2), DM[:,:,0]) + np.matmul(np.diag(dyidy**2.), DM[:,:,1])

        return self.y, self.D1, self.D2
