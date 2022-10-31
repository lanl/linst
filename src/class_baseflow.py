import sys
import math
import numpy as np
from scipy.special import erf
import matplotlib
import matplotlib.pyplot as plt


dp = np.dtype('d')  # double precision
i4 = np.dtype('i4') # integer 4
i8 = np.dtype('i8') # integer 8

class Baseflow:
    """
    This class define a general baseflow object
    """
    # def __init__(self, size): # here I pass ny
    #     self.size = size
    #     self.U    = np.zeros(size, dp)
    #     self.W    = np.zeros(size, dp)
    #     self.Up   = np.zeros(size, dp)
    #     self.Wp   = np.zeros(size, dp)

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
        

class RayleighTaylorBaseflow(Baseflow):
    """
    This child class of class Baseflow defines the "Rayleigh-Taylor" baseflow
    """
    def __init__(self, size, y, map): # here I pass ny
        
        self.size = size

        r1 = 1.
        r2 = 1.5

        delta = 0.1

        z = y
        r = r1/2. + (r2-r1)/2.*erf(z/delta)

        pi = math.pi

        self.Rho  = r
        self.Rhop = (r2-r1)/2.*( 2.0*np.exp( -z**2/delta**2 )/(delta*np.sqrt(pi)) )
        
        self.Mu   = 1.0e-5*np.abs(y)
        self.Mup  = 1.0e-5*np.abs(y)

        Rhop_num = np.matmul(map.D1, self.Rho)

        plot = False

        if (plot):
            ptn = plt.gcf().number + 1
            
            f = plt.figure(ptn)
            plt.plot(self.Rho, z, 'cs', label="Rho (analytic)")
            #plt.plot(self.Rhop, z, 'b', label="Rhop (analytic)")
            #plt.plot(Rhop_num, z, 'r--', label="Rhop (numerical)")
            plt.xlabel('drho/dz')
            plt.ylabel('z')
            plt.legend(loc="upper right")
            f.show()

            input('RT baseflow')

            sys.exit("Debug RT")

