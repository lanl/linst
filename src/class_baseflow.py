import sys
import math
import warnings
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
    def __init__(self, size, y, map, Re, grav, k): # here I pass ny
        
        self.size = size

        z  = y
        pi = math.pi
                    
        plot = True
        opt  = 1

        if (opt==0):
            
            r1 = 1.
            r2 = 1.5
            
            delta = 0.1
            
            r = r1/2. + (r2-r1)/2.*erf(z/delta)
            
            self.Rho  = r
            self.Rhop = (r2-r1)/2.*( 2.0*np.exp( -z**2/delta**2 )/(delta*np.sqrt(pi)) )
            
            self.Mu   = 1.0e-5*np.abs(y)
            self.Mup  = 1.0e-5*np.abs(y)
            
        elif(opt==1):

            rho1  = 1.0
            mu1   = 1.e-2

            delta = 0.0005 # see Morgan, Likhachev, Jacobs Fig. 17 legend

            mu2  = 3.*mu1 # that way I get Amu = 0.5
            rho2 = 3.*rho1 # that way I get Atw = 0.5

            rho0 = 0.5*( rho1 + rho2 )
            mu0 = 0.5*( mu1 + mu2 )

            Atw = (rho2-rho1)/(rho2+rho1)
            Amu = (mu2-mu1)/(mu2+mu1)

            print("Atw, Amu = ", Atw, Amu)
            
            self.Rho = rho0*( 1. + Atw*erf(z/delta) )
            self.Mu  = mu0*( 1. + Amu*erf(z/delta) )

            self.Rhop = rho0*Atw*( 2.0*np.exp( -z**2/delta**2 )/(delta*np.sqrt(pi)) )
            self.Mup  = mu0*Amu*( 2.0*np.exp( -z**2/delta**2 )/(delta*np.sqrt(pi)) )
            
            nu0 = mu0/rho0

            self.Tscale = (nu0/grav**2.)**(1./3.)
            self.Lscale = (nu0**2./grav)**(1./3.)

            print('The time scale is %21.11f, the length scale is %21.11f' % (self.Tscale, self.Lscale))

            # I want k_nondim = 2
            k_nondim = k*self.Lscale
            print("k_nondim = ", k_nondim)
            print("delta_nondim (delta*) = ", delta/self.Lscale)

        else:
            
            sys.exit("Not a proper value for flag opt in RT baseflow")

        Rhop_num = np.matmul(map.D1, self.Rho)
        Mup_num  = np.matmul(map.D1, self.Mu)

        if (Re > 1e10):
            warnings.warn("RT baseflow: Re > 1e10 ==> setting viscosity to zero")
            self.Mu   = 0.0*self.Mu
            self.Mup  = 0.0*self.Mup

        if (plot):
            
            ptn = plt.gcf().number + 1
            
            f = plt.figure(ptn)
            plt.plot(self.Rho, z, 'bs', markerfacecolor='none', label="density")
            plt.xlabel('density')
            plt.ylabel('z')
            plt.legend(loc="upper right")
            f.show()

            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Rhop, z, 'b', markerfacecolor='none', label="drho/dz (analytical)")
            plt.plot(Rhop_num, z, 'r--', markerfacecolor='none', label="drho/dz (numerical)")
            plt.xlabel('viscosity')
            plt.ylabel('z')
            plt.legend(loc="upper right")
            f.show()

            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Mu, z, 'bs', markerfacecolor='none', label="viscosity")
            plt.xlabel('viscosity')
            plt.ylabel('z')
            plt.legend(loc="upper right")
            f.show()

            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Mup, z, 'b', markerfacecolor='none', label="dmudz (analytical)")
            plt.plot(Mup_num, z, 'r--', markerfacecolor='none', label="dmudz (numerical)")
            plt.xlabel('viscosity')
            plt.ylabel('z')
            plt.legend(loc="upper right")
            f.show()

            #input('RT baseflow')

            #sys.exit("Debug RT baseflow")
