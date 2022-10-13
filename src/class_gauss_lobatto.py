
import math
import numpy as np

dp = np.dtype('d')  # double precision
i4 = np.dtype('i4') # integer 4
i8 = np.dtype('i8') # integer 8

# Gauss-Lobatto Class

class GaussLobatto:

    #def cheby(N):    # Compute Gauss-Lobatto points and Spectral Matrices
    """
    This class computes the Gauss-Lobatto points as well as the spectral derivative matrices
    (first and second derivative w.r.t. y matrix operators) on the interval [1,-1]
    """
    
    # From Trefethen Spectral Methods with Matlab

    def __init__(self, size): # here I pass ny
        self.size = size
        # Initialize arrays
        self.xc = np.zeros( size, dp)
        self.D1 = np.zeros((size,size), dp)
        self.D2 = np.zeros((size,size), dp)
        self.DM = np.zeros((size,size,2), dp)

    def cheby(self,N): # here I pas ny-1
        
        for j in range(0, N+1):
            self.xc[j] = math.cos(j*math.pi/N)

    def spectral_diff_matrices(self,N): # here I pas ny-1
        
        for i in range(0, N+1):

            if (i==0 or i==N):
                ci=2
            else:
                ci=1

            for j in range(0, N+1):

                if (j==0 or j==N):
                    cj=2
                else:
                    cj=1

                if (i!=j):
                    self.D1[i,j]=ci*(-1)**(i+j) /( cj*( self.xc[i]-self.xc[j] ))


        for i in range(0, N+1):
            for j in range(0, N+1):
                if(i!=j):
                    self.D1[i,i] = self.D1[i,i]-self.D1[i,j]
    

        self.D2 = np.matmul(self.D1, self.D1)
        self.DM[:,:,0] = self.D1
        self.DM[:,:,1] = self.D2