import numpy as np
from scipy import linalg
#import scipy as sp

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

class SolveGeneralizedEVP:

    def __init__(self, size): # here I pass ny
        self.size = size
        self.EigVal = np.zeros(size, dpc)
        

    def solve_eigenvalue_problem(self, lhs, rhs):
        evalue, evect = linalg.eig(lhs, rhs)
        return evalue, evect


