import sys
import math
import warnings
import numpy as np
from scipy.special import erf
import matplotlib
matplotlib.use('TkAgg') #----> Specify the backend
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import module_utilities as mod_util

dp = np.dtype('d')  # double precision
i4 = np.dtype('i4') # integer 4
i8 = np.dtype('i8') # integer 8

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '14'
plt.rc('font', family='serif')

nondim_flag = 1
""" nondim_flag:
        0 -> my old nondimensionalisation
        1 -> nondimensionalization with Uref = sqrt(g*Lref), Lref = h = thickness layer
             ==> Fr = 1
        2 -> nondimensionalization with Lref = 1 and set Uref = 1 in input file
             ==> in that case 1/Fr^2 = g and Re = 1/nuref
        3 -> Chandrasekahr non-dimensionalization ==> set Re=1, Fr=1 in input file
"""

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
        self.Sc    = riist.Sc # based on nu_ref because nu might not be constant
        
        self.gref  = riist.gref
        self.Uref  = riist.Uref

        # To avoid duplicate, I delete the member of class_readinput that I redefine here
        del riist.Re_min, riist.Re_max, riist.npts_re, riist.At, riist.Fr, riist.Sc, riist.gref, riist.Uref

        #print("self.Fr, self.Re, self.Sc, self.gref, self.Uref = ",self.Fr, self.Re, self.Sc, self.gref, self.Uref)

        if (nondim_flag==1):
            print("")
            print("USING NEW NON-DIMENSIONALIZATION")
            print("")
            # Lref is here the dimensional thickness of the layer
            self.Lref  = 0.001 #0.00025484199796126404 #0.001
            
            self.nuref = np.sqrt(self.gref)*self.Lref**(3./2.)/self.Re
            self.Uref  = np.sqrt(self.gref*self.Lref)
            self.Fr    = self.Uref/np.sqrt(self.gref*self.Lref)

            print("Setting Lref to: ", self.Lref)
            print("")
            print("Overwritting Uref and Froude...")
            print("")
        
        elif (nondim_flag==0):
            print("")
            print("USING OLD NON-DIMENSIONALIZATION")
            print("")

            # Compute reference length (thickness of mixing zone for Rayleigh-Taylor)
            self.Lref  = self.Uref**2./(self.Fr**2.*self.gref)
            self.nuref = self.Uref*self.Lref/self.Re

        elif (nondim_flag==2):
            print("")
            print("USING NON-DIMENSIONALIZATION WITH LREF=1 AND UREF=1")
            print("SET UREF=1 IN INPUT FILE")
            print("")

            # This gives a dimensional solution where 1/Re = nu and 1/Fr^2 = g ==> basically dimensional form

            self.Lref = 1.0
            
            if (self.Uref != 1.0 or self.Lref != 1.0 ):
                sys.exit("Uref and Lref are supposed to be equal to 1!!!!")

            # Simplified because Lref = Uref = 1
            self.nuref = 1/self.Re
            self.Fr    = 1/np.sqrt(self.gref)

        elif (nondim_flag==3):
            print("")
            print("CHANDRASEKHAR NON-DIMENSIONALIZATION")
            print("")
            
            self.Lref  = 0.1019367991845056 #0.00005
            self.nuref = np.sqrt(self.gref)*self.Lref**(1.5)

            self.Uref  = (self.nuref*self.gref)**(1./3.)
            
            self.Fr = self.Uref/np.sqrt(self.gref*self.Lref)
            self.Re = self.Uref*self.Lref/self.nuref

            print("")
            print("Reynolds and Froude #, now set to:", self.Re, self.Fr)
            print("")

            tol_h = 1.0e-14
            if ( np.abs(self.Re-1.0) > tol_h ):
                sys.exit("Reynolds number should be 1")
            if ( np.abs(self.Fr-1.0) > tol_h ):
                sys.exit("Froude number should be 1")

                
        # Set dimensional diffusivity reference value
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
            if   (nondim_flag==1 or nondim_flag==2 or nondim_flag==3):
                print("Ref. length scale Lref [m] set to Lref                          = ", self.Lref)
            elif (nondim_flag==0):
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
            sys.exit("hyptan==1 ===> check before use")
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
            sys.exit("hyptan==0 ===> check before use")
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
    def __init__(self, size, y, bsfl_ref): # here I pass ny
        self.size = size
        self.U    = 1.0 - y**2.
        self.Up   = -2.0*y
        
        self.W    = np.zeros(size, dp)
        self.Wp   = np.zeros(size, dp)

        bsfl_ref.Tscale_Chandra = 0.0
        bsfl_ref.Lscale_Chandra = 0.0
        bsfl_ref.Uref_Chandra   = 0.0
        
        

class RayleighTaylorBaseflow(Baseflow):
    """
    This child class of class Baseflow defines the "Rayleigh-Taylor" baseflow
    """
    def __init__(self, size, y, map, bsfl_ref, mtmp): # here I pass ny

        ny = size
        self.size = size
        
        UseConstantMu = True #False
        UseNumericalDerivatives = False
        plotF = True
        opt  = 1

        if ( mtmp.boussinesq == -555 ): # compressible inviscid solver
            opt = 3
            print("")
            print("Swithching to opt = 3 for compressible inviscid solver")
            print("")


        if ( mtmp.boussinesq == -2 or mtmp.boussinesq == 10 ): # dimensional solver
            # Baseflow coordinate
            zdim = y
            znondim = y/bsfl_ref.Lref
            # Plotting coordinate
            zcoord = y
        else:                         # non-dimensional solvers
            # Plotting coordinate
            zcoord = y
            # Baseflow coordinate
            zdim = y*bsfl_ref.Lref
            znondim = y
            
        pi = math.pi

        At   = bsfl_ref.At
        Re   = bsfl_ref.Re
        Sc   = bsfl_ref.Sc
        
        Uref = bsfl_ref.Uref
        grav = bsfl_ref.gref
        Lref = bsfl_ref.Lref 
        nuref = bsfl_ref.nuref

        # Numerical differentiation operators (to compare with analytical baseflow derivatives)
        if ( mtmp.boussinesq == -2 or mtmp.boussinesq == 10 ): # dimensional solver
            D1_dim = map.D1
            D2_dim = map.D2

            D1_nondim = Lref*D1_dim
            D2_nondim = Lref**2.*D2_dim
        else:
            D1_dim = 1/Lref*map.D1    # non-dimensional solvers
            D2_dim = 1/Lref**2.*map.D2

            D1_nondim = map.D1
            D2_nondim = map.D2

        self.D1_dim = D1_dim
        self.D2_dim = D2_dim

        self.D1_nondim = D1_nondim
        self.D2_nondim = D2_nondim

        #print("At, Re, Uref, grav, Lref, nuref = ", At, Re, Uref, grav, Lref, nuref)

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

            # 
            # delta_star = 0.01 ==> delta_star is dimensionless delta ======> see Morgan, Likhachev, Jacobs Fig. 17 legend 0.0005
            #
            
            # delta is dimensionless [-], and delta_dim is dimensional [m]
            if (nondim_flag==1): # this is the "new" non-dimensionalization
                
                delta = 1.0 # cannot modify this one here becasue delta = delta_star/Lref = delta_star/delta_star
                delta_dim = delta*Lref
                if ( delta != 1.0 ):
                    sys.exit("Non-dimensional layer thickness (delta) must be one in this non-dimensionalization")
                    
            elif(nondim_flag==0 or nondim_flag==2 or nondim_flag==3): # this is the "old" non-dimensionalization
                delta     = 0.1 #0.01 #0.01 #0.0001 #0.01 #0.01 #0.00005 #0.02
                delta_dim = delta*Lref

            self.delta = delta
            self.delta_dim = delta_dim
            
            # Note: delta is the characteristic thickness of the diffusion layer
            print("Setting up R-T baseflow")
            print("-----------------------")
            print("delta     = ", delta)
            print("delta_dim = ", delta_dim)

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
            
            bsfl_ref.rhoref = rhoref
            bsfl_ref.muref = muref

            print("")
            print("Setting reference density and dynamic viscosity:")
            print("------------------------------------------------")
            print("rhoref = ", rhoref)
            print("muref  = ", muref)
            print("")
            print("Muref recomputed from Reynolds number: Muref=rhoref*Uref*Lref/Re = ", rhoref*Uref*Lref/Re)
            print("")

            Atw_check = (rho2-rho1)/(rho2+rho1)
            Amu_check = (mu2-mu1)/(mu2+mu1)

            #print("np.abs(At-Atw_check), np.abs(At-Amu_check) = ", np.abs(At-Atw_check), np.abs(At-Amu_check))
            
            tol_c = 1.e-10
            if ( np.abs(At-Atw_check) > tol_c or np.abs(At-Amu_check) > tol_c ):
                sys.exit("Atwood number inconsistency!!!!!!!")

            # Since the reference length Lref is dimensional, then zdim needs to be dimensional
                
            # Compute dimensional density and its derivatives (d/dy_dim and d^2/dy_dim^2)
            #print("zdim/delta_dim = ", zdim/delta_dim)
            #print("")
            #print("znondim/delta = ", znondim/delta)
            print("np.amax(np.abs(zdim/delta_dim-znondim/delta)) = ", np.amax(np.abs(zdim/delta_dim-znondim/delta)))

            # Compute non-dimensional density and its derivatives (d/dy and d^2/dy^2)
            self.Rho_nd = 1. + At*erf(znondim/delta)
            self.Rhop_nd = At*( 2.0*np.exp( -znondim**2/delta**2 )/( delta*np.sqrt(pi) ) )
            self.Rhopp_nd = 2*At/(delta*np.sqrt(pi))*(-2.*znondim/delta**2.)*np.exp(-znondim**2./delta**2.)
            
            self.Rho   = rhoref*( 1. + At*erf(zdim/delta_dim) )
            self.Rhop  = rhoref*At*( 2.0*np.exp( -zdim**2/delta_dim**2 )/( delta_dim*np.sqrt(pi) ) )
            self.Rhopp = 2.0*rhoref*At/(delta_dim*np.sqrt(pi))*(-2.*zdim/delta_dim**2.)*np.exp( -zdim**2/delta_dim**2 )

            self.Rho_new   = rhoref*self.Rho_nd
            self.Rhop_new  = rhoref/Lref*self.Rhop_nd
            self.Rhopp_new = rhoref/Lref**2.*self.Rhopp_nd

            print("")
            print("np.amax(np.abs(self.Rho-self.Rho_new)) = ", np.amax(np.abs(self.Rho-self.Rho_new)))
            print("np.amax(np.abs(self.Rhop-self.Rhop_new)) = ", np.amax(np.abs(self.Rhop-self.Rhop_new)))
            print("np.amax(np.abs(self.Rhopp-self.Rhopp_new)) = ", np.amax(np.abs(self.Rhopp-self.Rhopp_new)))
            print("")

            # To Match Chandrasekhar and simplified Sandoval Equations, I need to use constant mu
            # in Chandrasekhar equations, i.e. mu = muref
            
            if (UseConstantMu):
                print("")
                print("             ----------------------------------------------------")
                print("==========> | Using constant dynamic viscosity in baseflow setup |")
                print("             ----------------------------------------------------")
                print("")
                self.Mu  = muref*np.ones(size)
                self.Mup = 0.0*self.Mu
            else:
                print("")
                print("             --------------------------------------------------------")
                print("==========> | Using non-constant dynamic viscosity in baseflow setup |")
                print("             --------------------------------------------------------")

                print("")
                print("")

                #sys.exit("Check this before using again ==> i.e. dimensional vs. non-dimensional")

                if ( mtmp.boussinesq == 1 ): 
                    sys.exit("Boussinesq equations were derived considering mu = constant")

                if ( mtmp.boussinesq == -3 ): 
                    sys.exit("Sandoval equations were derived considering mu = constant")
                
                # Compute dimensional viscosity and its derivatives
                self.Mu  = muref*( 1. + At*erf(zdim/delta_dim) )
                self.Mup = muref*At*( 2.0*np.exp( -zdim**2/delta_dim**2 )/(delta_dim*np.sqrt(pi)) )
                
            # Compute dimensional kinematic viscosity
            nu         = np.divide(self.Mu, self.Rho)
            nu0        = muref/rhoref

            if ( np.abs(nu0-nuref) > tol_c ):
                print("np.abs(nu0-nuref) = ", np.abs(nu0-nuref))
                sys.exit("Kinematic viscosity inconsistency!!!!!!!")
               
            # Compute non-dimensional density and its derivatives (d/dy and d^2/dy^2)
            self.Rho_nd_old   = 1/rhoref*self.Rho
            # drho_dim/dy_dim = rhoref/Lref*(drho/dy) ==> drho/dy = Lref/rhoref*(drho_dim/dy_dim)
            self.Rhop_nd_old  = Lref/rhoref*self.Rhop
            # d^2rho/dy^2 = Lref^2/rhoref*(d^2rho_dim/dy_dim^2) 
            self.Rhopp_nd_old = Lref**2./rhoref*self.Rhopp

            print("np.amax(np.abs(self.Rho_nd-self.Rho_nd_old)) = ", np.amax(np.abs(self.Rho_nd-self.Rho_nd_old)))
            print("np.amax(np.abs(self.Rhop_nd-self.Rhop_nd_old)) = ", np.amax(np.abs(self.Rhop_nd-self.Rhop_nd_old)))
            print("np.amax(np.abs(self.Rhopp_nd-self.Rhopp_nd_old)) = ", np.amax(np.abs(self.Rhopp_nd-self.Rhopp_nd_old)))
            
            ##############################
            # COMPUTE BASEFLOW W-VELOCITY
            ##############################

            div_bsfl = -1./(Re*Sc)*( -np.divide( np.multiply(self.Rhop_nd, self.Rhop_nd), np.multiply(self.Rho_nd, self.Rho_nd) ) + np.divide(self.Rhopp_nd, self.Rho_nd) )
            div_bsfl_dim = -bsfl_ref.Dref*( -np.divide( np.multiply(self.Rhop, self.Rhop), np.multiply(self.Rho, self.Rho) ) + np.divide(self.Rhopp, self.Rho) )

            # Non-dimensional baseflow w-velocity and its z-derivative
            self.Wvel_nd  = -1./(Re*Sc)*np.divide(self.Rhop_nd, self.Rho_nd)
            self.Wvelp_nd  = div_bsfl

            Wvel_nd_num = mod_util.trapezoid_integration_cum(div_bsfl, znondim)
            Wvel_num = mod_util.trapezoid_integration_cum(div_bsfl_dim, zdim)

            # Dimensional baseflow w-velocity (to check that non-dimensionalization is consistent)
            self.Wvel = -bsfl_ref.Dref*np.divide(self.Rhop, self.Rho)
            self.Wvel_nd_after = self.Wvel/Uref

            self.Wvelp = div_bsfl_dim
            self.Wvelp_nd_after = self.Wvelp*Lref/Uref
            
            self.Wvelp_nd_num = np.matmul(D1_nondim, self.Wvel_nd)

            # Write baseflow to file
            mod_util.write_baseflow_out(znondim, zdim, self.Rho_nd, self.Rhop_nd, self.Rhopp_nd, self.Wvel_nd, self.Wvelp_nd, self.Rho, self.Rhop, self.Rhopp, self.Wvel, self.Wvelp)

            # Compute non-dimensional viscosity and its derivatives
            self.Mu_nd  = 1/muref*self.Mu
            self.Mup_nd = Lref/muref*self.Mup

            # Compute numerical derivatives and compare with analytical ones
            Rhop_num     = np.matmul(D1_dim, self.Rho)
            Rhopp_num    = np.matmul(D2_dim, self.Rho)
            
            Rhop_num_nd  = np.matmul(D1_nondim, self.Rho_nd)
            Rhopp_num_nd = np.matmul(D2_nondim, self.Rho_nd)

            # Check that baseflow is good enough by comparing exact to numerical derivatives
            max_Rhop_num = np.max(np.abs(Rhop_num))
            max_Rhopp_num = np.max(np.abs(Rhopp_num))
            #
            max_Rhop_num_nd = np.max(np.abs(Rhop_num_nd))
            max_Rhopp_num_nd = np.max(np.abs(Rhopp_num_nd))
            #
            max_Rhop  = np.max(np.abs(self.Rhop))
            max_Rhopp = np.max(np.abs(self.Rhopp))
            #
            max_Rhop_nd  = np.max(np.abs(self.Rhop_nd))
            max_Rhopp_nd = np.max(np.abs(self.Rhopp_nd))

            tol_bsfl = 0.01
            if ( np.abs(max_Rhop_num-max_Rhop)/max_Rhop > tol_bsfl or
                 np.abs(max_Rhop_num_nd-max_Rhop_nd)/max_Rhop_nd > tol_bsfl or
                 np.abs(max_Rhopp_num-max_Rhopp)/max_Rhopp > tol_bsfl or
                 np.abs(max_Rhopp_num_nd-max_Rhopp_nd)/max_Rhopp_nd > tol_bsfl ):
                print("")
                print("np.abs(max_Rhop_num-max_Rhop)/max_Rhop = ", np.abs(max_Rhop_num-max_Rhop)/max_Rhop)
                print("np.abs(max_Rhop_num_nd-max_Rhop_nd)/max_Rhop_nd = ", np.abs(max_Rhop_num_nd-max_Rhop_nd)/max_Rhop_nd)
                print("np.abs(max_Rhopp_num-max_Rhopp)/max_Rhopp = ", np.abs(max_Rhopp_num-max_Rhopp)/max_Rhopp)
                print("np.abs(max_Rhopp_num_nd-max_Rhopp_nd)/max_Rhopp_nd = ", np.abs(max_Rhopp_num_nd-max_Rhopp_nd)/max_Rhopp_nd)
                print("")
                sys.exit("Baseflow is not properly resolved: check ymax and lmap and re-run!")
            else:
                print("")
                print("Baseflow is appropriately resolved ===> proceeding")
                print("")
                
            Mup_num      = np.matmul(D1_dim, self.Mu)

            # Compute Chandrasekhar length and time scales
            print("Computing Chandrasekhar length and time scales")
            print("==============================================")
            print("using nu0  = ", nu0)
            print("using grav = ", grav)
            bsfl_ref.Tscale_Chandra = (nu0/grav**2.)**(1./3.)
            bsfl_ref.Lscale_Chandra = (nu0**2./grav)**(1./3.)
            bsfl_ref.Uref_Chandra   = bsfl_ref.Lscale_Chandra/bsfl_ref.Tscale_Chandra 

            print("Chandrasekhar time scale [s]      = ", bsfl_ref.Tscale_Chandra)
            print("Chandrasekhar length scale [m]    = ", bsfl_ref.Lscale_Chandra)
            print("Chandrasekhar velocity scale [m]  = ", bsfl_ref.Uref_Chandra)
            print("Chandrasekhar Reynolds number [m] = ", rhoref*bsfl_ref.Uref_Chandra*bsfl_ref.Lscale_Chandra/muref)


            for iblk in range(0,1):
                print("")
            
            # Check Reynolds number
            if ( np.abs(Re-Uref*Lref/nu0) > tol_c ):
                print("np.abs(Re-Uref*Lref/nu0) = ", np.abs(Re-Uref*Lref/nu0))
                warnings.warn("Reynolds number inconsistency!!!!!!!")
                print("")
            
            #print("Input Reynolds number                = ", Re)
            #print("Re-computed Reynolds number          = ", Uref*Lref/nu0)
            
            #print("Atwood number At                     = ", At)
            #print("Atwood number (viscosity) Amu        = ", At)

            Flag = 0
            # Check that kinematic viscosity is constant
            for ii in range(0, size):
                #print("np.abs(nu0-nu[ii]), tol_c = ", np.abs(nu0-nu[ii]), tol_c)
                if ( np.abs(nu0-nu[ii]) > tol_c ):
                    Flag=1
                    break

            if (Flag==1):
                warnings.warn("Kinematic viscosity is not constant!!!!!!!!")
                print("")
                    #sys.exit("Kinematic viscosity is not constant!!!!!!!!")
            
            #print("nu0-np.amax(nu)                      = ", nu0-np.amax(nu))
            #print("nu0-np.amin(nu)                      = ", nu0-np.amin(nu))
            #print("")

            # I want k_nondim = 2
            #k_nondim = k*self.Lscale
            #print("k_nondim                             = ", k_nondim)
            #print("delta_nondim (Lref*)                 = ", Lref/self.Lscale)
            #print

            # PLOT W-VELOCITY

            PlotW = 0
            if (PlotW==1):
            
                ptn = plt.gcf().number + 1
                
                f = plt.figure(ptn)
                #plt.plot(self.Wvel_nd_after, zcoord, 'k', linewidth=1.5, label="Baseflow wvel (nondim after)")
                plt.plot(Wvel_nd_num, znondim, 'k', linewidth=1.5, label="Baseflow wvel (nondim, numerical)")
                plt.plot(self.Wvel_nd, znondim, 'r--', linewidth=1.5, label="Baseflow wvel (nondim)")
                
                plt.xlabel("w-velocity [-]", fontsize=14)
                plt.ylabel('z [-]', fontsize=16)
                plt.gcf().subplots_adjust(left=0.16)
                plt.gcf().subplots_adjust(bottom=0.15)
                #plt.ylim([xmin, xmax])
                plt.legend(loc="upper right")
                f.show()
                
                ptn = plt.gcf().number + 1
                
                f = plt.figure(ptn)
                plt.plot(Wvel_num, zdim, 'k', linewidth=1.5, label="Baseflow wvel (dimensional, numerical)")
                plt.plot(self.Wvel, zdim, 'r--', linewidth=1.5, label="Baseflow wvel (dimensional)")
                
                plt.xlabel("w-velocity [m/s]", fontsize=14)
                plt.ylabel('z [m]', fontsize=16)
                plt.gcf().subplots_adjust(left=0.16)
                plt.gcf().subplots_adjust(bottom=0.15)
                #plt.ylim([xmin, xmax])
                plt.legend(loc="upper right")
                f.show()
                
                ptn = plt.gcf().number + 1
                
                f = plt.figure(ptn)
                plt.plot(self.Wvelp_nd, zcoord, 'k', linewidth=1.5, label="Baseflow d(wvel)/dz (nondim, exact)")
                plt.plot(self.Wvelp_nd_num, zcoord, 'r--', linewidth=1.5, label="Baseflow d(wvel)/dz (nondim, numerical)")
                plt.plot(self.Wvelp_nd_after, zcoord, 'b-.', linewidth=1.5, label="Baseflow d(wvel)/dz (nondim, after)")
                
                plt.xlabel("d(w-velocity)/dz [-]", fontsize=14)
                plt.ylabel('z', fontsize=16)
                plt.gcf().subplots_adjust(left=0.16)
                plt.gcf().subplots_adjust(bottom=0.15)
                #plt.ylim([xmin, xmax])
                plt.legend(loc="upper right")
                f.show()
                
                input("Check baseflow w-velocity!!!!!")


        elif(opt==3):

            sys.exit("This needs to be checked!!!!!!")
            
            x = y
            nx = len(y)
            xmin = x[0]

            print("xmin = ", xmin)

            delta  = At/( 2.0*( 1.0-np.sqrt(1-At**2.) ) )
            pres_c = 50
            x_c = xmin
            
            self.Rho_nd  = (1.0-At*np.tanh(delta*x))/(1.0+At)
            self.Rhop_nd = -At/(1.0+At)*delta/np.cosh(delta*x)**2.
            
            pres_bsfl_num = mod_util.trapezoid_integration_cum(self.Rho_nd, x)
            pres_bsfl_num = pres_bsfl_num + pres_c
            
            self.Pres_nd = pres_c + 1.0/(1.0+At)*( x-x_c + At/delta*np.log( np.cosh(delta*x_c)/np.cosh(delta*x) ) )

            # Take derivative of pressure and plot (see if you recover density)
            dpdx = np.zeros(nx, dp)
            for i in range(1, nx-1):
                dx = x[i+1]-x[i-1]
                dpdx[i] = ( self.Pres_nd[i+1] - self.Pres_nd[i-1] )/dx
        
            ptn = plt.gcf().number + 1
            
            f = plt.figure(ptn)
            plt.plot(x, self.Rho_nd, 'k', linewidth=1.5, label="Baseflow density")
            plt.plot(x[1:nx-2], dpdx[1:nx-2], 'r-.', linewidth=1.5, label="Numerical derivative of pressure")
            
            plt.xlabel("x", fontsize=14)
            plt.ylabel('density', fontsize=16)
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            #plt.ylim([xmin, xmax])
            plt.legend(loc="upper right")
            f.show()    
            
            ptn = ptn + 1
            
            f = plt.figure(ptn)
            plt.plot(x, self.Pres_nd, 'k', linewidth=1.5, label="Analytic integration")
            plt.plot(x, pres_bsfl_num, 'r-.', linewidth=1.5, label="Numerical integration")
            plt.xlabel(r"x", fontsize=14)
            plt.ylabel('pressure', fontsize=16)
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            #plt.ylim([xmin, xmax])
            plt.legend(loc="upper right")
            f.show()
            
            input("Plots for compressible inviscid solver")
            
        else:
            sys.exit("Not a proper value for flag opt in RT baseflow")


        ###################################
        # Compute layer thickness
        ###################################
        thick_crit = 0.01 # criterion
        
        delta_rho = rho2-rho1
        rho_test1 = rho1 + thick_crit*delta_rho
        rho_test2 = rho2 - thick_crit*delta_rho

        rho_thick1 = rho_test1*np.ones(ny)
        rho_thick2 = rho_test2*np.ones(ny)

        self.H_thick = mod_util.GetLayerThicknessRT(ny, zdim, rho_test1, rho_test2, self.Rho)
        self.h_thick = mod_util.GetLayerThicknessRT_CabotCook(ny, zdim, self.Rho, rho1, rho2)
        print("")
        print("RT Layer Thickness")
        print("==================")
        print("H_thick (x percent criterion), h_thick (Cabot & Cook), H_thick/h_thick = ", self.H_thick, self.h_thick, self.H_thick/self.h_thick)

        self.h_thick2 = mod_util.GetLayerThicknessRT_Other(ny, zdim, self. Rho, rho1, rho2)

        print("h_thick2 (other formula) = ", self.h_thick2)
        print("2*h_thick2 (other formula) = ", 2*self.h_thick2)
        print("")

        #if (Re > 1e10):
        #    warnings.warn("RT baseflow: Re > 1e10 ==> setting viscosity to zero")
        #    self.Mu   = 0.0*self.Mu
        #    self.Mup  = 0.0*self.Mup

        zmin = -0.005
        zmax = 0.005
        
        ptn = plt.gcf().number + 1
        
        f = plt.figure(ptn)

        max_tot = np.zeros(2, dp)
        max_tot[0] = np.amax(np.abs(-np.divide(np.multiply(self.Rhop, self.Rhop), self.Rho )))
        max_tot[1] = np.amax(np.abs(self.Rhopp))

        max_glob = np.amax(max_tot)
        
        plt.plot(-np.divide(np.multiply(self.Rhop, self.Rhop), self.Rho )/max_glob, zcoord, 'k', markerfacecolor='none', label=r"$\dfrac{-1}{\bar{\rho}}\dfrac{\partial \bar{\rho}}{\partial z}\dfrac{\partial \bar{\rho}}{\partial z}$")
        plt.plot(self.Rhopp/max_glob, zcoord, 'r', markerfacecolor='none', label=r"$\dfrac{\partial^2 \bar{\rho}}{\partial z^2}$")
        
        #plt.xlabel(r"$\rho$", fontsize=20)
        plt.ylabel('z', fontsize=20)
        plt.legend(loc="upper right")
        plt.gcf().subplots_adjust(left=0.16)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.ylim([zmin, zmax])
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.title(r"Check validity of Boussinesq")
        f.show()

        if (plotF and not opt == 3):

            ViewFullDom = True
            
            zmin = -0.01
            zmax = 0.01

            #
            # Plot rho-rho_ref (dimensional)
            #
            
            ptn = plt.gcf().number + 1
            
            f = plt.figure(ptn)
            plt.plot(self.Rho-rhoref, zdim, 'k', markerfacecolor='none', label=r"$\rho-\rho_{ref}$ (dimensional)")
            
            plt.xlabel(r"$\rho-\rho_{ref}$", fontsize=20)
            plt.ylabel('z (dimensional)', fontsize=20)
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$\rho-\rho_{ref}$")
            plt.legend(loc="upper center")
            f.show()

            #
            # Plot rho (dimensional)
            #
            
            ptn = plt.gcf().number + 1
            
            f = plt.figure(ptn)
            plt.plot(self.Rho, zdim, 'k', markerfacecolor='none', label="density (dimensional)")
            plt.plot(rho_thick1, zdim, 'r', markerfacecolor='none', label="density (dimensional, rho_thick1)")
            plt.plot(rho_thick2, zdim, 'b', markerfacecolor='none', label="density (dimensional, rho_thick2)")

            plt.xlabel(r"$\bar{\rho}$", fontsize=20)
            plt.ylabel('z (dimensional)', fontsize=20)
            #plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$\rho$")
            plt.legend(loc="upper center")
            f.show()

            #
            # Plot rho (non-dimensional)
            #
            
            ptn = plt.gcf().number + 1
            
            f = plt.figure(ptn)
            plt.plot(self.Rho_nd, znondim, 'k', markerfacecolor='none')#, label="density (non-dim)")

            plt.xlabel(r"$\bar{\rho}$", fontsize=20)
            #plt.ylabel('z (non-dim.)', fontsize=20)
            plt.ylabel('z', fontsize=20)

            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.ylim([-10, 10])

            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            #plt.title(r"$\rho$ vs z (Non-Dim.)")
            #plt.legend(loc="upper center")
            f.show()

            #
            # Plot d(rho)/dz (dimensional)
            #
            
            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Rhop, zdim, 'k', markerfacecolor='none', label="drho/dz (analytical)")
            plt.plot(Rhop_num, zdim, 'r--', markerfacecolor='none', label="drho/dz (numerical)")
            plt.xlabel(r"$d\rho/dz$", fontsize=20)
            plt.ylabel('z (dimensional)', fontsize=20)
            plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$d\rho/dz$ DIMENSIONAL")
            f.show()

            #
            # Plot d^2(rho)/dz^2 (dimensional)
            #

            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Rhopp, zdim, 'k', markerfacecolor='none', label="d2rho/dz2 (analytical)")
            plt.plot(Rhopp_num, zdim, 'r--', markerfacecolor='none', label="d2rho/dz2 (numerical)")
            plt.xlabel(r"$d^2\rho/dz^2$", fontsize=20)
            plt.ylabel('z (dimensional)', fontsize=20)
            plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$d^2\rho/dz^2$ DIMENSIONAL")
            f.show()

            #
            # Plot d(rho)/dz (non-dimensional)
            #

            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Rhop_nd, znondim, 'k', markerfacecolor='none')#, label="drho/dz (analytical)")
            #plt.plot(Rhop_num_nd, znondim, 'r--', markerfacecolor='none', label="drho/dz (numerical)")
            plt.xlabel(r"$d\bar{\rho}/dz$", fontsize=20)
            plt.ylabel('z', fontsize=20)
            #plt.ylabel('z (non-dimensional)', fontsize=20)
            #plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)

            plt.ylim([-10, 10])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            #plt.title(r"$d\rho/dz$ (Non-Dim.)")
            f.show()
            
            #
            # Plot d(rho)/dz (non-dimensional)
            #

            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Rhop_nd, znondim, 'k', markerfacecolor='none', label="drho/dz (analytical)")
            plt.plot(Rhop_num_nd, znondim, 'r--', markerfacecolor='none', label="drho/dz (numerical)")
            plt.xlabel(r"$d\rho/dz$", fontsize=20)
            plt.ylabel('z (non-dimensional)', fontsize=20)
            plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$d\rho/dz$ NON-DIM.")
            f.show()

            #
            # Plot d^2(rho)/dz^2 (non-dimensional)
            #

            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Rhopp_nd, znondim, 'k', markerfacecolor='none', label="d2rho/dz2 (analytical)")
            plt.plot(self.Rhopp_nd_old, znondim, 'r--', markerfacecolor='none', label="d2rho/dz2 (straight nondim.)")
            #plt.plot(Rhopp_num_nd, zcoord, 'r--', markerfacecolor='none', label="d2rho/dz2 (numerical)")
            plt.xlabel(r"$d^2\rho/dz^2$", fontsize=20)
            plt.ylabel('z (non-dimensional)', fontsize=20)
            plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$d^2\rho/dz^2$ NON-DIM.")
            f.show()

            #
            # Plot dynamic viscosity Mu (dimensional)
            #            
            
            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(self.Mu, zdim, 'k', markerfacecolor='none', label="viscosity")
            plt.xlabel(r"$\mu$", fontsize=20)
            plt.ylabel('z (dimensional)', fontsize=20)
            #plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$\mu$ Dimensional")
            f.show()

            #
            # Plot d(Mu)/dz (dimensional)
            #            

            ptn = ptn+1

            f, ax = plt.subplots()
            plt.plot(self.Mup, zdim, 'k', markerfacecolor='none', label="dmudz (analytical)")
            #plt.plot(Mup_num, z, 'r--', markerfacecolor='none', label="dmudz (numerical)")
            plt.xlabel(r"$d\mu/dz$", fontsize=20)
            plt.ylabel('z (dimensional)', fontsize=20)
            #plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])

            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$d\mu/dz$ Dimensional")
            f.show()

            #
            # Plot kinematic viscosity nu (dimensional)
            #            

            ptn = ptn+1
            
            f = plt.figure(ptn)
            plt.plot(nu, zdim, 'k', markerfacecolor='none', label=r"$\nu$")
            plt.xlabel(r"$\nu$", fontsize=20)
            plt.ylabel('z (dimensional)', fontsize=20)
            #plt.legend(loc="upper right")
            plt.gcf().subplots_adjust(left=0.16)
            plt.gcf().subplots_adjust(bottom=0.15)
            if (not ViewFullDom):
                plt.ylim([zmin, zmax])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title(r"$\nu$ Dimensional")
            f.show()


            if ( mtmp.boussinesq == -2 or mtmp.boussinesq == 10 ): # dimensional solver

                ptn = ptn + 1
                # Baseflow coordinate
                zdim = y
                # Plotting coordinate
                zcoord = y

                f = plt.figure(ptn)
                plt.plot(self.Rho, znondim, 'k', markerfacecolor='none', label="density")
                plt.xlabel(r"$\rho$", fontsize=20)
                plt.ylabel('z (non-dimensional)', fontsize=20)
                #plt.legend(loc="upper right")
                plt.gcf().subplots_adjust(left=0.16)
                plt.gcf().subplots_adjust(bottom=0.15)
                if (not ViewFullDom):
                    plt.ylim([zmin, zmax])
                plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                plt.title(r"$\rho$")
                f.show()

            
            print("")
            input('RT baseflow plots ... strike any key to proceed')

            # Close all plots so far
            mod_util.close_previous_plots()

            #sys.exit("Debug RT baseflow")

            if (UseNumericalDerivatives):
                print("")
                print("Using numerical derivatives in RayleighTaylorBaseflow")
                self.Rhop  = np.matmul(D1_dim, self.Rho)
                self.Rhopp = np.matmul(D2_dim, self.Rho)
                self.Mup   = np.matmul(D1_dim, self.Mu)

                self.Rhop_nd  = np.matmul(D1_nondim, self.Rho_nd)
                self.Rhopp_nd = np.matmul(D2_nondim, self.Rho_nd)
                self.Mup_nd   = np.matmul(D1_nondim, self.Mu_nd)
                
            else:
                print("")
                print("Using analytical derivatives in RayleighTaylorBaseflow")
                


                

                



