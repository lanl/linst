import sys
import math
import warnings
import numpy as np
from scipy.special import erf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

dp = np.dtype('d')  # double precision
i4 = np.dtype('i4') # integer 4
i8 = np.dtype('i8') # integer 8

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '14'
plt.rc('font', family='serif')

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

        # I used that to see that as fac goes to zero, perturbation frequency decreases
        # In the limit of no baseflow velocity, the perturbations should be steady
        fac       = 1.0 # 0.001

        hyptan    = 1

        if   (hyptan==1):
            print("")
            print("Using hyperbolic tangent in mixing-layer test case")
            print("")
            self.U    = fac*0.5*( 1. + np.tanh(y) )
            self.Up   = fac*0.5*( 1.0 - np.tanh(y)**2. )
        elif (hyptan==0):
            # Using erf function instead of hyperbolic tangent
            print("")
            print("Using error function (erf) in mixing-layer test case")
            print("")
            pi      = math.pi
            delta   = 1.0
            self.U  = 0.5*( 1.0 + erf(y/delta) )
            self.Up = 0.5*( 2.0*np.exp(-y**2./delta**2.)/( math.sqrt(pi)*delta ) )
        else:
            sys.exit("Not a proper value for flag hyptan")

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

            sys.exit("Need to implement second derivative of baseflow rho for Sandoval equations")
            
        elif(opt==1):

            rho1  = 1.0
            mu1   = 1.e-2 #1.e-2

            rho2 = 1.01*rho1 # use 3 to get Atw = 0.5
            mu2  = 1.01*mu1 # use 3 to get Amu = 0.5

            delta = 0.0005 # see Morgan, Likhachev, Jacobs Fig. 17 legend 0.0005
            
            rho0 = 0.5*( rho1 + rho2 )
            mu0 = 0.5*( mu1 + mu2 )

            Atw = (rho2-rho1)/(rho2+rho1)
            Amu = (mu2-mu1)/(mu2+mu1)
            
            self.Rho = rho0*( 1. + Atw*erf(z/delta) )
            self.Mu  = mu0*( 1. + Amu*erf(z/delta) )

            nu       = np.divide(self.Mu, self.Rho)

            self.Rhop = rho0*Atw*( 2.0*np.exp( -z**2/delta**2 )/(delta*np.sqrt(pi)) )
            self.Mup  = mu0*Amu*( 2.0*np.exp( -z**2/delta**2 )/(delta*np.sqrt(pi)) )

            print("")
            print("Mup is hard-coded to zero")
            print("")
            
            self.Mup = 0.0*self.Mup

            self.Rho_nd   = ( 1. + Atw*erf(z/delta) )
            self.Rhop_nd  = Atw*( 2.0*np.exp( -z**2/delta**2 )/(delta*np.sqrt(pi)) )
            self.Rhopp_nd = 2.*Atw/(delta*np.sqrt(pi))*(-2.*z/delta**2.)*np.exp( -z**2/delta**2 )
            
            nu0 = mu0/rho0

            #print("nu  = ", nu)

            self.Tscale = (nu0/grav**2.)**(1./3.)
            self.Lscale = (nu0**2./grav)**(1./3.)

            print("Atwood number (density) Atw    = ", Atw)
            print("Atwood number (viscosity) Amu  = ", Amu)
            print("Time scale                     = ", self.Tscale)
            print("Length scale                   = ", self.Lscale)
            print("nu0                            = ", nu0)
            print

            # I want k_nondim = 2
            k_nondim = k*self.Lscale
            print("k_nondim                       = ", k_nondim)
            print("delta_nondim (delta*)          = ", delta/self.Lscale)
            print
        else:
            
            sys.exit("Not a proper value for flag opt in RT baseflow")

        Rhop_num     = np.matmul(map.D1, self.Rho)

        Rhop_num_nd  = np.matmul(map.D1, self.Rho_nd)
        Rhopp_num_nd = np.matmul(map.D2, self.Rho_nd)
        
        Mup_num   = np.matmul(map.D1, self.Mu)

        if (Re > 1e10):
            warnings.warn("RT baseflow: Re > 1e10 ==> setting viscosity to zero")
            self.Mu   = 0.0*self.Mu
            self.Mup  = 0.0*self.Mup

        if (plot):

            ViewFullDom = True
            
            zmin = -0.01
            zmax = 0.01
            
            ptn = plt.gcf().number + 1
            
            f = plt.figure(ptn)
            plt.plot(self.Rho, z, 'k', markerfacecolor='none', label="density")
            plt.xlabel(r"$\rho$", fontsize=20)
            plt.ylabel('z', fontsize=20)
            #plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$\rho$")
            f.show()

            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Rhop, z, 'k', markerfacecolor='none', label="drho/dz (analytical)")
            plt.plot(Rhop_num, z, 'r--', markerfacecolor='none', label="drho/dz (numerical)")
            plt.xlabel(r"$d\rho/dz$", fontsize=20)
            plt.ylabel('z', fontsize=20)
            plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$d\rho/dz$")
            f.show()

            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Rhop_nd, z, 'k', markerfacecolor='none', label="drho/dz (analytical)")
            plt.plot(Rhop_num_nd, z, 'r--', markerfacecolor='none', label="drho/dz (numerical)")
            plt.xlabel(r"$d\rho/dz$", fontsize=20)
            plt.ylabel('z', fontsize=20)
            plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$d\rho/dz$ NON-DIM.")
            f.show()


            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Rhopp_nd, z, 'k', markerfacecolor='none', label="d2rho/dz2 (analytical)")
            plt.plot(Rhopp_num_nd, z, 'r--', markerfacecolor='none', label="d2rho/dz2 (numerical)")
            plt.xlabel(r"$d^2\rho/dz^2$", fontsize=20)
            plt.ylabel('z', fontsize=20)
            plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$d^2\rho/dz^2$ NON-DIM.")
            f.show()
            
            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Mu, z, 'k', markerfacecolor='none', label="viscosity")
            plt.xlabel(r"$\mu$", fontsize=20)
            plt.ylabel('z', fontsize=20)
            #plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            f.show()

            ptn = ptn+1

            f, ax = plt.subplots()
            plt.plot(self.Mup, z, 'k', markerfacecolor='none', label="dmudz (analytical)")
            #plt.plot(Mup_num, z, 'r--', markerfacecolor='none', label="dmudz (numerical)")
            plt.xlabel(r"$d\mu/dz$", fontsize=20)
            plt.ylabel('z', fontsize=20)
            #plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])

            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            f.show()

            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(nu, z, 'k', markerfacecolor='none', label=r"$\nu$")
            plt.xlabel(r"$\nu$", fontsize=20)
            plt.ylabel('z', fontsize=20)
            #plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            f.show()

            input('RT baseflow')

            #sys.exit("Debug RT baseflow")




