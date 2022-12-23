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
    def __init__(self):
        pass

    def set_reference_quantities(self, riist):

        if ( riist.Re_min != riist.Re_max and riist.rt_flag ):
            sys.exit("R-T not setup for multiple Reynolds number")

        self.Re     = riist.Re_min
            
        self.Re_min = riist.Re_min
        self.Re_max = riist.Re_max
        self.npts_re = riist.npts_re

        self.At    = riist.At
        self.Fr    = riist.Fr
        self.Sc    = riist.Sc
        
        self.gref  = riist.gref
        self.Uref  = riist.Uref

        # To avoid duplicate, I delete the member of class_readinput that I redefine here
        del riist.Re_min, riist.Re_max, riist.npts_re, riist.At, riist.Fr, riist.Sc, riist.gref, riist.Uref

        #print("self.Fr, self.Re, self.Sc, self.gref, self.Uref = ",self.Fr, self.Re, self.Sc, self.gref, self.Uref)

        # Compute reference length (thickness of mixing zone for Rayleigh-Taylor)
        self.Lref  = self.Uref**2./(self.Fr**2.*self.gref)

        self.nuref = self.Uref*self.Lref/self.Re

        self.Dref  = self.nuref/self.Sc

        WriteHere = True
        if (WriteHere):
            print("")
            print("Main non-dimensional numbers and corresponding reference quantities:")
            print("====================================================================")
            print("Atwood number At               = ", self.At)
            print("Froude number Fr               = ", self.Fr)
            print("Schmidt number Sc              = ", self.Sc)
            print("Reynolds number Re             = ", self.Re)
            print("Reference gravity gref [m/s^2] = ", self.gref)
            print("Reference velocity Uref [m/s]  = ", self.Uref)
            print("")
            print("Other reference quantities:")
            print("---------------------------")
            print("Mass transfer Peclet number ( Pe = Re*Sc )                      = ", self.Re*self.Sc)
            print("Ref. mass diffusivity Dref [m^2/s] ( Dref = nuref/Sc )          = ", self.Dref)
            print("Ref. length scale Lref [m]( Lref = Uref^2/(Fr^2*gref) )         = ", self.Lref)
            print("Ref. kinematic viscosity nuref [m^2/s] ( nuref = Uref*Lref/Re ) = ", self.nuref)
            print("")
            
        # self.Re = 0.0
        # self.Fr = 0.0
        # self.Sc = 0.0
        # self.Uref = 0.0
        # self.gref = 0.0
        # self.Lref = 0.0
        # self.Dref = 0.0
        # self.nuref = 0.0

        # self.Re_min = 0.0
        # self.Re_max = 0.0
        # self.npts_re = 0
        #self.size = size
        #self.U    = np.zeros(size, dp)
        #self.W    = np.zeros(size, dp)
        #self.Up   = np.zeros(size, dp)
        #self.Wp   = np.zeros(size, dp)

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
    def __init__(self, size, y, map, bsfl_ref, mtmp): # here I pass ny
        
        self.size = size

        #zdim = y
        #zcoord = zdim #y/bsfl_ref.Lref
        
        if ( mtmp.boussinesq == -2 ): # dimensional solver
            # Baseflow coordinate
            zdim = y
            # Plotting coordinate
            zcoord = y
        else:                         # non-dimensional solvers
            # Plotting coordinate
            zcoord = y
            # Baseflow coordinate
            zdim = y*bsfl_ref.Lref
            
        pi = math.pi

        At   = bsfl_ref.At
        Re   = bsfl_ref.Re
        Uref = bsfl_ref.Uref
        grav = bsfl_ref.gref
        Lref = bsfl_ref.Lref 
        nuref = bsfl_ref.nuref

        #print("At, Re, Uref, grav, Lref, nuref = ", At, Re, Uref, grav, Lref, nuref)

        # 
        # delta_star = 0.01 ==> delta_star is dimensionless delta ======> see Morgan, Likhachev, Jacobs Fig. 17 legend 0.0005
        #

        # Here I take delta as dimensionless, and delta_dim is its dimensional value
        delta     = 0.02
        delta_dim = delta*Lref
        
        # Note: delta is the characteristic thickness of the diffusion layer
        print("Setting up R-T baseflow")
        print("-----------------------")
        print("delta     = ", delta)
        print("delta_dim = ", delta_dim)
                    
        plot = True
        opt  = 1

        if (opt==0):

            sys.exit("Check this before using in terms of bsfl_ref and Rhopp")
            
            r1 = 1.
            r2 = 1.5
            
            #Lref = 0.1
            
            r = r1/2. + (r2-r1)/2.*erf(zdim/Lref)
            
            self.Rho  = r
            self.Rhop = (r2-r1)/2.*( 2.0*np.exp( -zdim**2/Lref**2 )/(Lref*np.sqrt(pi)) )
            
            self.Mu   = 1.0e-5*np.abs(y)
            self.Mup  = 1.0e-5*np.abs(y)
            
        elif(opt==1):

            fac   = (1.+At)/(1.-At)
            div   = 1. + fac
            
            # Dimensional density of material 1 and 2
            rho1  = 1.0
            rho2  = rho1*fac #1.01*rho1 # use 3 to get Atw = 0.5 ==== 1.01

            # Dimensional dynamic viscosity of material 1 and 2 (consistent with Reynolds number)
            mu1   = (rho1+rho2)/Re*Uref*Lref/div # 1.e-6 #1.e-2
            mu2   = mu1*fac                      #1.01*mu1 # use 3 to get Amu = 0.5  ==== 1.01

            # Reference density and dynamic viscosity taken as average value between material 1 and 2
            rhoref = 0.5*( rho1 + rho2 )
            muref  = 0.5*( mu1 + mu2 )

            print("")
            print("Setting reference density and dynamic viscosity:")
            print("------------------------------------------------")
            print("rhoref = ", rhoref)
            print("muref  = ", muref)
            print("")
            bsfl_ref.rhoref = rhoref
            bsfl_ref.muref = muref

            Atw_check = (rho2-rho1)/(rho2+rho1)
            Amu_check = (mu2-mu1)/(mu2+mu1)

            #print("np.abs(At-Atw_check), np.abs(At-Amu_check) = ", np.abs(At-Atw_check), np.abs(At-Amu_check))
            
            tol_c = 1.e-12
            if ( np.abs(At-Atw_check) > tol_c or np.abs(At-Amu_check) > tol_c ):
                sys.exit("Atwood number inconsistency!!!!!!!")

            # Since the reference length Lref is dimensional, then zdim needs to be dimensional
                
            # Compute dimensional density and its derivatives (d/dy_dim and d^2/dy_dim^2)
            self.Rho   = rhoref*( 1. + At*erf(zdim/delta_dim) )
            self.Rhop  = rhoref*At*( 2.0*np.exp( -zdim**2/delta_dim**2 )/(delta_dim*np.sqrt(pi)) )
            self.Rhopp = 2.0*rhoref*At/(delta_dim*np.sqrt(pi))*(-2.*zdim/delta_dim**2.)*np.exp( -zdim**2/delta_dim**2 )

            # To Match Chandrasekhar and simplified Sandoval Equations, I need to use constant mu
            # in Chandrasekhar equations, i.e. mu = muref
            UseConstantMu = True
            if (UseConstantMu):
                print("")
                print("Using constant dynamic viscosity in baseflow setup")
                print("")
                self.Mu  = muref*np.ones(size)
                self.Mup = 0.0*self.Mu
            else:
                # Compute dimensional viscosity and its derivatives
                self.Mu  = muref*( 1. + At*erf(zdim/delta_dim) )
                self.Mup = muref*At*( 2.0*np.exp( -zdim**2/delta_dim**2 )/(delta_dim*np.sqrt(pi)) )
                
            # Compute dimensional kinematic viscosity
            nu         = np.divide(self.Mu, self.Rho)
            nu0        = muref/rhoref

            if ( np.abs(nu0-nuref) > tol_c ):
               sys.exit("Kinematic viscosity inconsistency!!!!!!!")
            
            # Compute non-dimensional density and its derivatives (d/dy and d^2/dy^2)
            self.Rho_nd   = 1/rhoref*self.Rho
            # drho_dim/dy_dim = rhoref/Lref*(drho/dy) ==> drho/dy = Lref/rhoref*(drho_dim/dy_dim)
            self.Rhop_nd  = Lref/rhoref*self.Rhop
            # d^2rho/dy^2 = Lref^2/rhoref*(d^2rho_dim/dy_dim^2) 
            self.Rhopp_nd = Lref**2./rhoref*self.Rhopp

            # Compute non-dimensional viscosity and its derivatives
            self.Mu_nd  = 1/muref*self.Mu
            self.Mup_nd = Lref/muref*self.Mup

            # Compute Chandrasekhar length and time scales
            self.Tscale = (nu0/grav**2.)**(1./3.)
            self.Lscale = (nu0**2./grav)**(1./3.)

            for iblk in range(0,1):
                print("")
            
            # Check Reynolds number
            if ( np.abs(Re-Uref*Lref/nu0) > tol_c ):
                sys.exit("Reynolds number inconsistency!!!!!!!")
            
            #print("Input Reynolds number                = ", Re)
            #print("Re-computed Reynolds number          = ", Uref*Lref/nu0)
            
            #print("Atwood number At                     = ", At)
            #print("Atwood number (viscosity) Amu        = ", At)
            print("Chandrasekhar time scale   = ", self.Tscale)
            print("Chandrasekhar length scale = ", self.Lscale)
            print("nu0                        = ", nu0)

            # Check that kinematic viscosity is constant
            for ii in range(0, size):
                #print("np.abs(nu0-nu[ii]), tol_c = ", np.abs(nu0-nu[ii]), tol_c)
                if ( np.abs(nu0-nu[ii]) > tol_c ):
                    warnings.warn("Kinematic viscosity is not constant!!!!!!!!")
                    #sys.exit("Kinematic viscosity is not constant!!!!!!!!")
            
            #print("nu0-np.amax(nu)                      = ", nu0-np.amax(nu))
            #print("nu0-np.amin(nu)                      = ", nu0-np.amin(nu))
            #print("")

            # I want k_nondim = 2
            #k_nondim = k*self.Lscale
            #print("k_nondim                             = ", k_nondim)
            #print("delta_nondim (Lref*)                 = ", Lref/self.Lscale)
            #print
        else:
            sys.exit("Not a proper value for flag opt in RT baseflow")

        # Compute numerical derivatives to compare with analytical ones
        if ( mtmp.boussinesq == -2 ): # dimensional solver
            D1_dim = map.D1
            D2_dim = map.D2

            D1_nondim = Lref*D1_dim
            D2_nondim = Lref**2.*D2_dim
        else:
            D1_dim = 1/Lref*map.D1    # non-dimensional solvers
            D2_dim = 1/Lref**2.*map.D2

            D1_nondim = map.D1
            D2_nondim = map.D2

        Rhop_num     = np.matmul(D1_dim, self.Rho)
        Rhopp_num    = np.matmul(D2_dim, self.Rho)

        Rhop_num_nd  = np.matmul(D1_nondim, self.Rho_nd)
        Rhopp_num_nd = np.matmul(D2_nondim, self.Rho_nd)
        
        Mup_num      = np.matmul(D1_dim, self.Mu)

        #if (Re > 1e10):
        #    warnings.warn("RT baseflow: Re > 1e10 ==> setting viscosity to zero")
        #    self.Mu   = 0.0*self.Mu
        #    self.Mup  = 0.0*self.Mup

        if (plot):

            ViewFullDom = True
            
            zmin = -0.01
            zmax = 0.01
            
            ptn = plt.gcf().number + 1
            
            f = plt.figure(ptn)
            plt.plot(self.Rho, zcoord, 'k', markerfacecolor='none', label="density")
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
            plt.plot(self.Rhop, zcoord, 'k', markerfacecolor='none', label="drho/dz (analytical)")
            plt.plot(Rhop_num, zcoord, 'r--', markerfacecolor='none', label="drho/dz (numerical)")
            plt.xlabel(r"$d\rho/dz$", fontsize=20)
            plt.ylabel('z', fontsize=20)
            plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$d\rho/dz$ DIMENSIONAL")
            f.show()

            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Rhopp, zcoord, 'k', markerfacecolor='none', label="d2rho/dz2 (analytical)")
            plt.plot(Rhopp_num, zcoord, 'r--', markerfacecolor='none', label="d2rho/dz2 (numerical)")
            plt.xlabel(r"$d^2\rho/dz^2$", fontsize=20)
            plt.ylabel('z', fontsize=20)
            plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$d^2\rho/dz^2$ DIMENSIONAL")
            f.show()

            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Rhop_nd, zcoord, 'k', markerfacecolor='none', label="drho/dz (analytical)")
            plt.plot(Rhop_num_nd, zcoord, 'r--', markerfacecolor='none', label="drho/dz (numerical)")
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
            plt.plot(self.Rhopp_nd, zcoord, 'k', markerfacecolor='none', label="d2rho/dz2 (analytical)")
            plt.plot(Rhopp_num_nd, zcoord, 'r--', markerfacecolor='none', label="d2rho/dz2 (numerical)")
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
            plt.plot(self.Mu, zcoord, 'k', markerfacecolor='none', label="viscosity")
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
            plt.plot(self.Mup, zcoord, 'k', markerfacecolor='none', label="dmudz (analytical)")
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
            plt.plot(nu, zcoord, 'k', markerfacecolor='none', label=r"$\nu$")
            plt.xlabel(r"$\nu$", fontsize=20)
            plt.ylabel('z', fontsize=20)
            #plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            f.show()

            print("")
            input('RT baseflow plots ... strike any key to proceed')

            #sys.exit("Debug RT baseflow")




