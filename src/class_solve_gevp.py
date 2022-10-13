import numpy as np
from scipy import linalg
#import scipy as sp

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

class SolveGeneralizedEVP:
    """
    This class define a solution object and contain the main function that provides the solution 
    to the generalized eigenvalue problem (GEVP)
    """
    def __init__(self, size):
        """
        Constructor of class SolveGeneralizedEVP
        """
        self.size = size
        self.EigVal = np.zeros(size*size, dpc)
        self.EigVec = np.zeros((size, size), dpc)

    def solve_eigenvalue_problem(self, lhs, rhs):
        """
        Function of class SolveGeneralizedEVP that solves the GEVP using linalg.eig
        """
        self.EigVal, self.EigVec = linalg.eig(lhs, rhs)
        return self.EigVal, self.EigVec

        #evalue, evect = linalg.eig(lhs, rhs)
        #return evalue, evect


