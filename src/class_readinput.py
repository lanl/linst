
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
i4 = np.dtype('i4') # integer 4
i8 = np.dtype('i8') # integer 8

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '14'
plt.rc('font', family='serif')


class ReadInput:
    """
    This class define a general input object
    """
    def __init__(self):
        pass


    def read_input_file(self, inFile):

        with open(inFile) as f:
            lines = f.readlines()

        nlines  = len(lines)

        # Solver type: 1 -> Rayleigh-Taylor, 2 -> Mixing-layer/Poiseuille solver
        i=0
        tmp = lines[i].split("#")[0]
        self.SolverT = int(tmp)
        
        # baseflowT: 1 -> hyperbolic-tangent baseflow, 2 -> plane Poiseuille baseflow
        i=i+1
        
        tmp = lines[i].split("#")[0]
        self.baseflowT = int(tmp)
        
        # ny
        i=i+1
        
        tmp = lines[i].split("#")[0]
        self.ny  = int(tmp)
        
        # Re_min, Re_max, npts_re
        i=i+1
        
        tmp = lines[i].split("#")[0]
        tmp = tmp.split(",")
        
        self.Re_min = float(tmp[0])
        self.Re_max = float(tmp[1])
        self.npts_re  = int(tmp[2])
        
        if (self.Re_min == self.Re_max): self.npts_re = 1

        #Re  = float(tmp)
        
        # Froude number, Schmidt number, reference velocity Uref and acceleration gref
        i=i+1
        
        tmp = lines[i].split("#")[0]
        tmp = tmp.split(",")

        self.At   = float(tmp[0])
        self.Fr   = float(tmp[1])
        self.Sc   = float(tmp[2])
        self.Uref = float(tmp[3])
        self.gref = float(tmp[4])

        # alpha_min, alpha_max and npts_alpha
        i=i+1
        
        tmp = lines[i].split("#")[0]
        tmp = tmp.split(",")
    
        self.alpha_min = float(tmp[0])
        self.alpha_max = float(tmp[1])
        self.npts_alp  = int(tmp[2])
        
        if (self.alpha_min == self.alpha_max): self.npts_alp = 1
        
        self.alpha = np.linspace(self.alpha_min, self.alpha_max, self.npts_alp)
        
        # beta
        i=i+1
        
        tmp = lines[i].split("#")[0]
        self.beta  = float(tmp)
        
        # yinf
        i=i+1
        
        tmp = lines[i].split("#")[0]
        self.yinf  = float(tmp)
        
        # lmap
        i=i+1
        
        tmp  = lines[i].split("#")[0]
        self.lmap = float(tmp)
        
        # target
        i=i+1
        
        tmp = lines[i].split("#")[0]
        tmp = tmp.split(",")
        
        self.target = float(tmp[0]) + 1j*float(tmp[1])

        for iblk in range(0,10):
            print("")
            
        print("Reading input file: all inputs are non-dimensional except Uref, gref and target")
        print("===============================================================================")
        print("====> Uref and gref have to be provided dimensional")
        print("====> target: dimensional for solver boussinesq = -2 only, non-dim. otherwise")
        print("")
        print("SolverT                     = ", self.SolverT)
        print("baseflowT                   = ", self.baseflowT)
        print("ny                          = ", self.ny)
        print("Re_min                      = ", self.Re_min)
        print("Re_max                      = ", self.Re_max)
        print("npts_re                     = ", self.npts_re)
        #print("Reference velocity Uref     = ", self.Uref)
        #print("Reference acceleration gref = ", self.gref)
        #print("Atwood number               = ", self.At)
        #print("Froude number               = ", self.Fr)
        #print("Schmidt number              = ", self.Sc)
        print("alpha_min [-]               = ", self.alpha_min)
        print("alpha_max [-]               = ", self.alpha_max)
        print("npts_alp                    = ", self.npts_alp)
        print("beta [-]                    = ", self.beta)
        print("yinf [-]                    = ", self.yinf)
        print("lmap [-]                    = ", self.lmap)
        print("target                      = ", self.target)
        print("==> Eigenfunction will be extracted for omega = ", self.target)

        # print("")
        # print("Main non-dimensional numbers and corresponding reference quantities:")
        # print("--------------------------------------------------------------------")
        # print("Atwood number At            = ", self.At)
        # print("Froude number Fr            = ", self.Fr)
        # print("Schmidt number Sc           = ", self.Sc)
        # print("Reynolds number Re          = ", self.Re)
        # print("Reference gravity gref      = ", self.gref)
        # print("Reference velocity Uref     = ", self.Uref)
        # print("")
        # print("Remaining computed quantities:")
        # print("------------------------------")
        # print("Mass transfer Peclet number ( Pe = Re*Sc )           = ", self.Re*self.Sc)
        # print("Reference mass diffusivity Dref ( nuref/Sc )         = ", self.Dref)
        # print("Reference length scale Lref ( Uref^2/(Fr^2*gref) )   = ", self.Lref)
        # print("Reference kinematic viscosity nuref ( Uref*Lref/Re ) = ", self.nuref)
        # print("")

        
        print("")
        
        self.rt_flag = False
        
        if (self.SolverT==1):
            self.rt_flag = True
        elif (self.SolverT==2):
            pass
        else:
            sys.exit("Not a proper value for flag SolverT!!!!")
    
        #return rt_flag, SolverT, baseflowT, ny, Re_min, Re_max, npts_re, alpha_min, alpha_max, npts_alp, alpha, beta, yinf, lmap, target
