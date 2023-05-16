import sys
import math
import warnings
import numpy as np
from scipy.special import erf
import matplotlib
matplotlib.use('TkAgg') #----> Specify the backend
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

dp = np.dtype('d')  # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4 = np.dtype('i4') # integer 4
i8 = np.dtype('i8') # integer 8

class MainArrays:
    """
    This class define a general baseflow object
    """
    def __init__(self, n_re, n_alp, ny): # here I pass 4*ny for regular Navier-Stokes Stability and 5*ny for "Rayleigh-Taylor" Navier-Stokes
        """
        Constructor for class MainArrays
        """
        # Initialize arrays
        self.re_array = np.zeros(n_re, dp)

        self.omega_array  = np.zeros((n_re, n_alp), dpc)
        self.omega_dim    = np.zeros((n_re, n_alp), dpc)
        self.omega_nondim = np.zeros((n_re, n_alp), dpc)

        self.l_taylor_dim = np.zeros(ny, dp)
        self.l_integr_dim = np.zeros(ny, dp)

        self.l_taylor_dim_fst = np.zeros(1, dp)
        self.l_integr_dim_fst = np.zeros(1, dp)

        self.eps_dim_r_ke_dim_fst = np.zeros(1, dp)
        self.eps_r_ke_fst = np.zeros(1, dp)
