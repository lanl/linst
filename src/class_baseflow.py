
import numpy as np

dp = np.dtype('d')  # double precision
i4 = np.dtype('i4') # integer 4
i8 = np.dtype('i8') # integer 8

class Baseflow:
    """
    This class define a general baseflow object
    """
    def __init__(self, size): # here I pass ny
        self.size = size
        self.U    = np.zeros(size, dp)
        self.W    = np.zeros(size, dp)
        self.Up   = np.zeros(size, dp)
        self.Wp   = np.zeros(size, dp)

class HypTan(Baseflow):
    """
    This child class of class Baseflow defines the hyperbolic tangent baseflow for shear-layer applications
    """
    def __init__(self, size, y): # here I pass ny
        self.size = size
        self.U    = 0.5*( 1. + np.tanh(y) )
        self.Up   = 0.5*( 1.0 - np.tanh(y)**2. )

        #self.U    = np.tanh(y)
        #self.Up   = ( 1.0 - np.tanh(y)**2. )

        self.W    = np.zeros(size, dp)
        self.Wp   = np.zeros(size, dp)

class PlanePoiseuille(Baseflow):
    """
    This child class of class Baseflow defines the plane Poiseuille baseflow
    """
    def __init__(self, size, y): # here I pass ny
        self.size = size
        self.U    = 1.0 - y**2.
        self.Up   = -2.0*y

        self.W    = np.zeros(size, dp)
        self.Wp   = np.zeros(size, dp)


