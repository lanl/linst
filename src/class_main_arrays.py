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
    def __init__(self, n_re, n_alp): # here I pass 4*ny for regular Navier-Stokes Stability and 5*ny for "Rayleigh-Taylor" Navier-Stokes
        """
        Constructor for class MainArrays
        """
        # Initialize arrays
        self.omega_array = np.zeros((n_re, n_alp), dpc)
        self.re_array = np.zeros(n_re, dp)

        self.omega_dim = np.zeros(1, dpc)
        self.omega_nondim = np.zeros(1, dpc)
