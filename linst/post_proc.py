
tol = 1.e-14
uniform_val = 0.001 #0.015 # 0.005  ### 0.1 -> 10 markers, 0.01 -> 100 markers,....
marker_sz = 4
taylor_opt = 1

import os
import sys
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

import module_utilities as mod_util
import solve_gevp as solgevp

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

class PostProcess(solgevp.SolveGeneralizedEVP):
    """
    This class define a post-processing object
    """
    def __init__(self, solver):
        """
        Constructor of class PostProcess
        """
        #super(PostProcess, self).__init__(*args, **kwargs)
        self.solver = solver

    def compute_inner_prod(self, f, g):
        inner_prod = 0.5*( np.multiply(np.conj(f), g) + np.multiply(f, np.conj(g)) )    
        return inner_prod

    def get_disturbance_kinetic_energy(self):
        # Disturbance kinetic energy
        uu = self.compute_inner_prod(self.solver.ueig, self.solver.ueig)
        vv = self.compute_inner_prod(self.solver.veig, self.solver.veig)
        ww = self.compute_inner_prod(self.solver.weig, self.solver.weig)
        self.ke_dist = 0.5*( uu + vv + ww )

        if ( np.amax(np.abs(self.ke_dist.imag)) > tol ):
            sys.exit("Disturbance kinetic energy should be real!")
        else:
            self.ke_dist = self.ke_dist.real

    def get_eigenfcts_derivatives(self):

        #
        # For RT and Single Fluid: the vertical direction is z
        #

        D1 = self.solver.map.D1
        D2 = self.solver.map.D2
        
        ia = 1j*self.solver.alpha[0]
        ib = 1j*self.solver.beta[0]

        ia2 = ( ia )**2.
        ib2 = ( ib )**2.

        iab = ia*ib

        # u-derivatives
        self.dudx = ia*self.solver.ueig
        self.dudy = ib*self.solver.ueig
        self.dudz = np.matmul(D1, self.solver.ueig)

        self.d2udx2 = ia2*self.solver.ueig
        self.d2udy2 = ib2*self.solver.ueig
        self.d2udz2 = np.matmul(D2, self.solver.ueig)
        self.d2udxdy = iab*self.solver.ueig

        # v-derivatives
        self.dvdx = ia*self.solver.veig
        self.dvdy = ib*self.solver.veig
        self.dvdz = np.matmul(D1, self.solver.veig)

        self.d2vdx2 = ia2*self.solver.veig
        self.d2vdy2 = ib2*self.solver.veig
        self.d2vdz2 = np.matmul(D2, self.solver.veig)
        self.d2vdxdy = iab*self.solver.veig
        
        # w-derivatives
        self.dwdx = ia*self.solver.weig
        self.dwdy = ib*self.solver.weig
        self.dwdz = np.matmul(D1, self.solver.weig)

        self.d2wdx2 = ia2*self.solver.weig
        self.d2wdy2 = ib2*self.solver.weig
        self.d2wdz2 = np.matmul(D2, self.solver.weig)

        # p-derivatives
        self.dpdx = ia*self.solver.peig
        self.dpdy = ib*self.solver.peig
        self.dpdz = np.matmul(D1, self.solver.peig)

        self.d2pdx2 = ia2*self.solver.peig
        self.d2pdy2 = ib2*self.solver.peig
        self.d2pdz2 = np.matmul(D2, self.solver.peig)

        # rho-derivatives: not relevant for single fluid
        self.drdx = ia*self.solver.reig
        self.drdy = ib*self.solver.reig
        self.drdz = np.matmul(D1, self.solver.reig)
                
        self.d2rdx2 = ia2*self.solver.reig
        self.d2rdy2 = ib2*self.solver.reig
        self.d2rdz2 = np.matmul(D2, self.solver.reig)

    def get_reynolds_stresses_related_quantities(self):

        print("")
        print("Computation of Reynolds stresses ====> I did not include the constant term pi/alpha or ...")
        print("")

        Re = self.solver.Re

        if ( not hasattr(self, 'dudx') ):
            self.get_eigenfcts_derivatives()

        self.rxx = self.compute_inner_prod(self.solver.ueig, self.solver.ueig)
        self.ryy = self.compute_inner_prod(self.solver.veig, self.solver.veig)
        self.rzz = self.compute_inner_prod(self.solver.weig, self.solver.weig)
        
        self.rxy = self.compute_inner_prod(self.solver.ueig, self.solver.veig)
        self.rxz = self.compute_inner_prod(self.solver.ueig, self.solver.weig)
        self.ryz = self.compute_inner_prod(self.solver.veig, self.solver.weig)

        self.drxxdx = 2.*self.compute_inner_prod(self.solver.ueig, self.dudx)
        self.drxxdy = 2.*self.compute_inner_prod(self.solver.ueig, self.dudy)
        self.drxxdz = 2.*self.compute_inner_prod(self.solver.ueig, self.dudz)
        #
        self.d2rxxdxdy = 2.*( self.compute_inner_prod(self.dudx, self.dudy)
                              +self.compute_inner_prod(self.solver.ueig, self.d2udxdy)
                             ) 

        self.d2rxxdy2 = 2.*( self.compute_inner_prod(self.dudy, self.dudy)
                             +self.compute_inner_prod(self.solver.ueig, self.d2udy2)
                             ) 

        self.dryydx = 2.*self.compute_inner_prod(self.solver.veig, self.dvdx)
        self.dryydy = 2.*self.compute_inner_prod(self.solver.veig, self.dvdy)
        self.dryydz = 2.*self.compute_inner_prod(self.solver.veig, self.dvdz)
        #
        self.d2ryydy2 = 2.*( self.compute_inner_prod(self.dvdy, self.dvdy)
                             +self.compute_inner_prod(self.solver.veig, self.d2vdy2)
                            ) 

        self.drzzdx = 2.*self.compute_inner_prod(self.solver.weig, self.dwdx)
        self.drzzdy = 2.*self.compute_inner_prod(self.solver.weig, self.dwdy)
        self.drzzdz = 2.*self.compute_inner_prod(self.solver.weig, self.dwdz)

        self.drxydx = self.compute_inner_prod(self.solver.ueig, self.dvdx) + self.compute_inner_prod(self.solver.veig, self.dudx)
        self.drxydy = self.compute_inner_prod(self.solver.ueig, self.dvdy) + self.compute_inner_prod(self.solver.veig, self.dudy)
        self.drxydz = self.compute_inner_prod(self.solver.ueig, self.dvdz) + self.compute_inner_prod(self.solver.veig, self.dudz)

        self.drxzdx = self.compute_inner_prod(self.solver.ueig, self.dwdx) + self.compute_inner_prod(self.solver.weig, self.dudx)
        self.drxzdy = self.compute_inner_prod(self.solver.ueig, self.dwdy) + self.compute_inner_prod(self.solver.weig, self.dudy)
        self.drxzdz = self.compute_inner_prod(self.solver.ueig, self.dwdz) + self.compute_inner_prod(self.solver.weig, self.dudz)

        self.dryzdx = self.compute_inner_prod(self.solver.veig, self.dwdx) + self.compute_inner_prod(self.solver.weig, self.dvdx)
        self.dryzdy = self.compute_inner_prod(self.solver.veig, self.dwdy) + self.compute_inner_prod(self.solver.weig, self.dvdy)
        self.dryzdz = self.compute_inner_prod(self.solver.veig, self.dwdz) + self.compute_inner_prod(self.solver.weig, self.dvdz)
        
        #
        self.d2rxydx2 = ( 2.*self.compute_inner_prod(self.dudx, self.dvdx)
                          +self.compute_inner_prod(self.solver.ueig, self.d2vdx2)
                          +self.compute_inner_prod(self.solver.veig, self.d2udx2)
                         ) 
        #
        self.d2rxydy2 = ( 2.*self.compute_inner_prod(self.dudy, self.dvdy)
                          +self.compute_inner_prod(self.solver.ueig, self.d2vdy2)
                          +self.compute_inner_prod(self.solver.veig, self.d2udy2)
                         ) 
        #
        self.d2rxydxdy = ( self.compute_inner_prod(self.dudy, self.dvdx)
                           +self.compute_inner_prod(self.solver.ueig, self.d2vdxdy)
                           +self.compute_inner_prod(self.dudx, self.dvdy)
                           +self.compute_inner_prod(self.solver.veig, self.d2udxdy)
                         ) 

        #
        # Build left-hand-sides for Reynolds stress equations
        #
        
        # Normal terms
        self.rxx_lhs = 2.*self.omega_i*self.rxx.real
        self.ryy_lhs = 2.*self.omega_i*self.ryy.real
        self.rzz_lhs = 2.*self.omega_i*self.rzz.real

        # Cross terms
        self.rxy_lhs = 2.*self.omega_i*self.rxy.real
        self.rxz_lhs = 2.*self.omega_i*self.rxz.real
        self.ryz_lhs = 2.*self.omega_i*self.ryz.real

        #
        # Rxx equation
        #
        self.rxx_lap = 2.*( self.compute_inner_prod(self.dudx, self.dudx)
                            +self.compute_inner_prod(self.dudy, self.dudy)
                            +self.compute_inner_prod(self.dudz, self.dudz)
                           )/Re
        
        self.rxx_lap = self.rxx_lap + 2.*( self.compute_inner_prod(self.solver.ueig, self.d2udx2)
                                           +self.compute_inner_prod(self.solver.ueig, self.d2udy2)
                                           +self.compute_inner_prod(self.solver.ueig, self.d2udz2)
                                          )/Re
        
        self.rxx_dissip = 2.*( self.compute_inner_prod(self.dudx, self.dudx)
                               +self.compute_inner_prod(self.dudy, self.dudy)
                               +self.compute_inner_prod(self.dudz, self.dudz)
                              )/Re

        self.rxx_advection = -np.multiply(self.solver.bsfl.U, self.drxxdx) -np.multiply(self.solver.bsfl.V, self.drxxdy)
        self.rxx_production = -2.*np.multiply(self.rxz, self.solver.bsfl.Up)

        self.rxx_pres_diffus = -2.* ( self.compute_inner_prod(self.solver.ueig, self.dpdx) + self.compute_inner_prod(self.solver.peig, self.dudx) )
        self.rxx_pres_strain = 2.*self.compute_inner_prod(self.solver.peig, self.dudx)

        # No gravity in x-direction
        self.rxx_grav_prod = 0.0*self.compute_inner_prod(self.solver.peig, self.dudx)

        #
        # Ryy equation
        #
        self.ryy_lap = 2.*( self.compute_inner_prod(self.dvdx, self.dvdx)
                            +self.compute_inner_prod(self.dvdy, self.dvdy)
                            +self.compute_inner_prod(self.dvdz, self.dvdz)
                           )/Re
        
        self.ryy_lap = self.ryy_lap + 2.*( self.compute_inner_prod(self.solver.veig, self.d2vdx2)
                                           +self.compute_inner_prod(self.solver.veig, self.d2vdy2)
                                           +self.compute_inner_prod(self.solver.veig, self.d2vdz2)
                                          )/Re
        
        self.ryy_dissip = 2.*( self.compute_inner_prod(self.dvdx, self.dvdx)
                               +self.compute_inner_prod(self.dvdy, self.dvdy)
                               +self.compute_inner_prod(self.dvdz, self.dvdz)
                              )/Re

        self.ryy_advection = -np.multiply(self.solver.bsfl.U, self.dryydx) -np.multiply(self.solver.bsfl.V, self.dryydy)

        self.ryy_production = -2.*np.multiply(self.ryz, self.solver.bsfl.Vp) 

        self.ryy_pres_diffus = -2.*( self.compute_inner_prod(self.solver.veig, self.dpdy) + self.compute_inner_prod(self.solver.peig, self.dvdy) )
        self.ryy_pres_strain = 2.*self.compute_inner_prod(self.solver.peig, self.dvdy)

        # No gravity in y-direction
        self.ryy_grav_prod = 0.0*self.compute_inner_prod(self.solver.peig, self.dudx)
        
        #
        # Rzz equation
        #
        
        self.rzz_lap = 2.*( self.compute_inner_prod(self.dwdx, self.dwdx)
                            +self.compute_inner_prod(self.dwdy, self.dwdy)
                            +self.compute_inner_prod(self.dwdz, self.dwdz)
                           )/Re
        
        self.rzz_lap = self.rzz_lap + 2.*( self.compute_inner_prod(self.solver.weig, self.d2wdx2)
                                           +self.compute_inner_prod(self.solver.weig, self.d2wdy2)
                                           +self.compute_inner_prod(self.solver.weig, self.d2wdz2)
                                          )/Re
        
        self.rzz_dissip = 2.*( self.compute_inner_prod(self.dwdx, self.dwdx)
                               +self.compute_inner_prod(self.dwdy, self.dwdy)
                               +self.compute_inner_prod(self.dwdz, self.dwdz)
                              )/Re

        self.rzz_advection = -np.multiply(self.solver.bsfl.U, self.drzzdx) -np.multiply(self.solver.bsfl.V, self.drzzdy)
        self.rzz_production = 0.0*self.rzz_advection

        self.rzz_pres_diffus = -2.*( self.compute_inner_prod(self.solver.weig, self.dpdz) + self.compute_inner_prod(self.solver.peig, self.dwdz) )
        self.rzz_pres_strain = 2.*self.compute_inner_prod(self.solver.peig, self.dwdz)

        # This is only for RT
        if ( self.solver.Fr != -10. ):
            print("")
            print("Presence of a nonzero gravity term in Rzz equation ==> must be RT equations")
            self.rzz_grav_prod = -2.*self.compute_inner_prod(self.solver.reig, self.solver.weig)/self.solver.Fr**2.
        else:
            self.rzz_grav_prod = 0.0*self.compute_inner_prod(self.solver.reig, self.solver.weig)


        #
        # Rxy equation
        #
        self.rxy_lap = 2.*( self.compute_inner_prod(self.dudx, self.dvdx)
                            +self.compute_inner_prod(self.dudy, self.dvdy)
                            +self.compute_inner_prod(self.dudz, self.dvdz)
                           )/Re
        
        self.rxy_lap = self.rxy_lap + ( self.compute_inner_prod(self.solver.ueig, self.d2vdx2)
                                        +self.compute_inner_prod(self.solver.ueig, self.d2vdy2)
                                        +self.compute_inner_prod(self.solver.ueig, self.d2vdz2)
                                       )/Re

        self.rxy_lap = self.rxy_lap + ( self.compute_inner_prod(self.solver.veig, self.d2udx2)
                                        +self.compute_inner_prod(self.solver.veig, self.d2udy2)
                                        +self.compute_inner_prod(self.solver.veig, self.d2udz2)
                                       )/Re

        self.rxy_dissip = 2.*( self.compute_inner_prod(self.dudx, self.dvdx)
                               +self.compute_inner_prod(self.dudy, self.dvdy)
                               +self.compute_inner_prod(self.dudz, self.dvdz)
                              )/Re

        self.rxy_advection = -np.multiply(self.solver.bsfl.U, self.drxydx) -np.multiply(self.solver.bsfl.V, self.drxydy)
        self.rxy_production = -np.multiply(self.ryz, self.solver.bsfl.Up) -np.multiply(self.rxz, self.solver.bsfl.Vp)

        self.rxy_pres_diffus = -( self.compute_inner_prod(self.solver.ueig, self.dpdy)
                               +self.compute_inner_prod(self.solver.peig, self.dudy)
                               +self.compute_inner_prod(self.solver.veig, self.dpdx)
                               +self.compute_inner_prod(self.solver.peig, self.dvdx)
                              )
                                                                                                          
        self.rxy_pres_strain = ( self.compute_inner_prod(self.solver.peig, self.dudy)
                                 +self.compute_inner_prod(self.solver.peig, self.dvdx)
                                )

        print("max(v*dpdx) = ", np.max(self.compute_inner_prod(self.solver.veig, self.dpdx)))
        print("min(v*dpdx) = ", np.min(self.compute_inner_prod(self.solver.veig, self.dpdx)))
        
        print("max(p*dvdx) = ", np.max(self.compute_inner_prod(self.solver.peig, self.dvdx)))
        print("min(p*dvdx) = ", np.min(self.compute_inner_prod(self.solver.peig, self.dvdx)))

        # No gravity because gravity is only in z equation and this equation is formed using u and v equations
        self.rxy_grav_prod = 0.0*self.compute_inner_prod(self.solver.peig, self.dudx)

        #
        # Rxz equation
        #
        self.rxz_lap = 2.*( self.compute_inner_prod(self.dudx, self.dwdx)
                            +self.compute_inner_prod(self.dudy, self.dwdy)
                            +self.compute_inner_prod(self.dudz, self.dwdz)
                           )/Re
        
        self.rxz_lap = self.rxz_lap + ( self.compute_inner_prod(self.solver.ueig, self.d2wdx2)
                                        +self.compute_inner_prod(self.solver.ueig, self.d2wdy2)
                                        +self.compute_inner_prod(self.solver.ueig, self.d2wdz2)
                                       )/Re

        self.rxz_lap = self.rxz_lap + ( self.compute_inner_prod(self.solver.weig, self.d2udx2)
                                        +self.compute_inner_prod(self.solver.weig, self.d2udy2)
                                        +self.compute_inner_prod(self.solver.weig, self.d2udz2)
                                       )/Re

        self.rxz_dissip = 2.*( self.compute_inner_prod(self.dudx, self.dwdx)
                               +self.compute_inner_prod(self.dudy, self.dwdy)
                               +self.compute_inner_prod(self.dudz, self.dwdz)
                              )/Re

        self.rxz_advection = -np.multiply(self.solver.bsfl.U, self.drxzdx) -np.multiply(self.solver.bsfl.V, self.drxzdy)
        self.rxz_production = -np.multiply(self.rzz, self.solver.bsfl.Up)

        self.rxz_pres_diffus = -( self.compute_inner_prod(self.solver.ueig, self.dpdz)
                               +self.compute_inner_prod(self.solver.peig, self.dudz)
                               +self.compute_inner_prod(self.solver.weig, self.dpdx)
                               +self.compute_inner_prod(self.solver.peig, self.dwdx)
                              )
                                                                                                          
        self.rxz_pres_strain = ( self.compute_inner_prod(self.solver.peig, self.dudz)
                                 +self.compute_inner_prod(self.solver.peig, self.dwdx)
                                )

        # For RT flows
        if ( self.solver.Fr != -10. ):
            print("")
            print("Presence of a nonzero gravity term in Rxz equations ==> must be RT equations")
            self.rxz_grav_prod = -self.compute_inner_prod(self.solver.reig, self.solver.ueig)/self.solver.Fr**2.
        else:
            self.rxz_grav_prod = 0.0*self.compute_inner_prod(self.solver.reig, self.solver.weig)

        #
        # Ryz equation
        #
        self.ryz_lap = 2.*( self.compute_inner_prod(self.dvdx, self.dwdx)
                            +self.compute_inner_prod(self.dvdy, self.dwdy)
                            +self.compute_inner_prod(self.dvdz, self.dwdz)
                           )/Re
        
        self.ryz_lap = self.ryz_lap + ( self.compute_inner_prod(self.solver.veig, self.d2wdx2)
                                        +self.compute_inner_prod(self.solver.veig, self.d2wdy2)
                                        +self.compute_inner_prod(self.solver.veig, self.d2wdz2)
                                       )/Re

        self.ryz_lap = self.ryz_lap + ( self.compute_inner_prod(self.solver.weig, self.d2vdx2)
                                        +self.compute_inner_prod(self.solver.weig, self.d2vdy2)
                                        +self.compute_inner_prod(self.solver.weig, self.d2vdz2)
                                       )/Re

        self.ryz_dissip = 2.*( self.compute_inner_prod(self.dvdx, self.dwdx)
                               +self.compute_inner_prod(self.dvdy, self.dwdy)
                               +self.compute_inner_prod(self.dvdz, self.dwdz)
                              )/Re

        self.ryz_advection = -np.multiply(self.solver.bsfl.U, self.dryzdx) -np.multiply(self.solver.bsfl.V, self.dryzdy)
        self.ryz_production = -np.multiply(self.rzz, self.solver.bsfl.Vp)

        self.ryz_pres_diffus = -( self.compute_inner_prod(self.solver.veig, self.dpdz)
                               +self.compute_inner_prod(self.solver.peig, self.dvdz)
                               +self.compute_inner_prod(self.solver.weig, self.dpdy)
                               +self.compute_inner_prod(self.solver.peig, self.dwdy)
                              )
                                                                                                          
        self.ryz_pres_strain = ( self.compute_inner_prod(self.solver.peig, self.dvdz)
                                 +self.compute_inner_prod(self.solver.peig, self.dwdy)
                                )

        # For RT flows
        if ( self.solver.Fr != -10. ):
            print("")
            print("Presence of a nonzero gravity term in Ryz equations ==> must be RT equations")
            self.ryz_grav_prod = -self.compute_inner_prod(self.solver.reig, self.solver.veig)/self.solver.Fr**2.
        else:
            self.ryz_grav_prod = 0.0*self.compute_inner_prod(self.solver.reig, self.solver.weig)

        #
        # Write data
        #
        pdudx = self.compute_inner_prod(self.solver.peig, self.dudx)
        pdvdy = self.compute_inner_prod(self.solver.peig, self.dvdy)
        pdwdz = self.compute_inner_prod(self.solver.peig, self.dwdz)
        
        udpdx = self.compute_inner_prod(self.solver.ueig, self.dpdx)
        vdpdy = self.compute_inner_prod(self.solver.veig, self.dpdy)
        wdpdz = self.compute_inner_prod(self.solver.weig, self.dpdz)
        
        ddxpu = pdudx + udpdx
        ddypv = pdvdy + vdpdy
        ddzpw = pdwdz + wdpdz
        
        pdrdx = self.compute_inner_prod(self.solver.peig, self.drdx)
        pdrdy = self.compute_inner_prod(self.solver.peig, self.drdy)
        pdrdz = self.compute_inner_prod(self.solver.peig, self.drdz)

        rdpdx = self.compute_inner_prod(self.solver.reig, self.dpdx)
        rdpdy = self.compute_inner_prod(self.solver.reig, self.dpdy)
        rdpdz = self.compute_inner_prod(self.solver.reig, self.dpdz)
        
        ddxpr = pdrdx + rdpdx
        ddypr = pdrdy + rdpdy
        ddzpr = pdrdz + rdpdz
        
        save_path = "./"
        fname     = "Mean_Pressure_Related_Quantities_LST.txt"
        fullname  = os.path.join(save_path, fname)
        fhandel   = open(fullname,'w')
        fhandel.write('VARIABLES = "z" "pdudx" "pdvdy" "pdwdz" "udpdx" "vdpdy" "wdpdz" "ddxpu" "ddypv" "ddzpw" "pdrdx" "pdrdy" "pdrdz" "ddxpr" "ddypr" "ddzpr"\n')
        np.savetxt(fhandel, np.transpose([ self.solver.map.z, pdudx.real, pdvdy.real, pdwdz.real, udpdx.real, vdpdy.real, wdpdz.real, ddxpu.real, \
                                           ddypv.real, ddzpw.real, pdrdx.real, pdrdy.real, pdrdz.real, ddxpr.real, ddypr.real, ddzpr.real ]), \
                      fmt="%25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e")
        fhandel.close()

    def get_balance_reynolds_stress_equations(self, omega_i):

        # This should work both for single fluid and RT Boussinesq

        self.omega_i = omega_i
        
        self.get_eigenfcts_derivatives()
        self.get_reynolds_stresses_related_quantities()
                
        # Build rhs for Rxx equation                                         
        self.rxx_rhs = ( self.rxx_advection
                         +self.rxx_production
                         +self.rxx_pres_diffus
                         +self.rxx_pres_strain
                         +self.rxx_lap
                         -self.rxx_dissip
                        )

        self.rxx_rhs = self.rxx_rhs.real

        # Build rhs for Ryy equation
        self.ryy_rhs = ( self.ryy_advection
                         +self.ryy_pres_diffus
                         +self.ryy_pres_strain
                         +self.ryy_lap
                         -self.ryy_dissip
                        )

        self.ryy_rhs = self.ryy_rhs.real

        # Build rhs for Rzz equation        
        self.rzz_rhs = ( self.rzz_advection
                         +self.rzz_production
                         +self.rzz_pres_diffus
                         +self.rzz_pres_strain
                         +self.rzz_lap
                         -self.rzz_dissip
                         +self.rzz_grav_prod # this term is only in RT (not present in shear-layer)
                        )

        self.rzz_rhs = self.rzz_rhs.real

        # Build rhs for Rxy equation                 
        self.rxy_rhs = ( self.rxy_advection
                         +self.rxy_production
                         +self.rxy_pres_diffus
                         +self.rxy_pres_strain
                         +self.rxy_lap
                         -self.rxy_dissip
                        )

        self.rxy_rhs = self.rxy_rhs.real

        # Build rhs for Rxz equation                 
        self.rxz_rhs = ( self.rxz_advection
                         +self.rxz_production
                         +self.rxz_pres_diffus
                         +self.rxz_pres_strain
                         +self.rxz_lap
                         -self.rxz_dissip
                         +self.rxz_grav_prod # this term is only in RT (not present in shear-layer)
                        )

        self.rxz_rhs = self.rxz_rhs.real

        # Build rhs for Ryz equation                 
        self.ryz_rhs = ( self.ryz_advection
                         +self.ryz_production
                         +self.ryz_pres_diffus
                         +self.ryz_pres_strain
                         +self.ryz_lap
                         -self.ryz_dissip
                         +self.ryz_grav_prod # this term is only in RT (not present in shear-layer)
                        )

        self.ryz_rhs = self.ryz_rhs.real

        print("Reynold's stresses balance equations")
        print("====================================")
        print("Growth rate used for lhs computation: ", self.omega_i)
        print("")

        print("Rxx equation:")
        print("-------------")
        print("rxx_bal = ", np.amax(np.abs(self.rxx_lhs-self.rxx_rhs)))
        print("")
        
        print("Ryy equation:")
        print("-------------")
        print("ryy_bal = ", np.amax(np.abs(self.ryy_lhs-self.ryy_rhs)))
        print("")
        
        print("Rzz equation:")
        print("-------------")
        print("rzz_bal = ", np.amax(np.abs(self.rzz_lhs-self.rzz_rhs)))
        print("")

        print("Rxy equation:")
        print("-------------")
        print("rxy_bal = ", np.amax(np.abs(self.rxy_lhs-self.rxy_rhs)))
        print("")

        print("Rxz equation:")
        print("-------------")
        print("rxz_bal = ", np.amax(np.abs(self.rxz_lhs-self.rxz_rhs)))
        print("")

        print("Ryz equation:")
        print("-------------")
        print("ryz_bal = ", np.amax(np.abs(self.ryz_lhs-self.ryz_rhs)))
        print("")
        
        print("Reynolds stress derivatives (max. value)")
        print("----------------------------------------")
        print("drxx/dx = %21.11e" % np.amax(np.abs(self.drxxdx)))
        print("drxx/dy = %21.11e" % np.amax(np.abs(self.drxxdy)))
        print("drxx/dz = %21.11e" % np.amax(np.abs(self.drxxdz)))
        
        print("dryy/dx = %21.11e" % np.amax(np.abs(self.dryydx)))
        print("dryy/dy = %21.11e" % np.amax(np.abs(self.dryydy)))
        print("dryy/dz = %21.11e" % np.amax(np.abs(self.dryydz)))

        print("drzz/dx = %21.11e" % np.amax(np.abs(self.drzzdx)))
        print("drzz/dy = %21.11e" % np.amax(np.abs(self.drzzdy)))
        print("drzz/dz = %21.11e" % np.amax(np.abs(self.drzzdz)))

        print("drxy/dx = %21.11e" % np.amax(np.abs(self.drxydx)))
        print("drxy/dy = %21.11e" % np.amax(np.abs(self.drxydy)))
        print("drxy/dz = %21.11e" % np.amax(np.abs(self.drxydz)))

        print("drxz/dx = %21.11e" % np.amax(np.abs(self.drxzdx)))
        print("drxz/dy = %21.11e" % np.amax(np.abs(self.drxzdy)))
        print("drxz/dz = %21.11e" % np.amax(np.abs(self.drxzdz)))

        print("dryz/dx = %21.11e" % np.amax(np.abs(self.dryzdx)))
        print("dryz/dy = %21.11e" % np.amax(np.abs(self.dryzdy)))
        print("dryz/dz = %21.11e" % np.amax(np.abs(self.dryzdz)))

        print("")

        max_ddx_pv = np.max( np.abs( self.compute_inner_prod(self.solver.veig, self.dpdx) + self.compute_inner_prod(self.solver.peig, self.dvdx) ))
        max_ddy_pu = np.max( np.abs( self.compute_inner_prod(self.solver.ueig, self.dpdy) + self.compute_inner_prod(self.solver.peig, self.dudy) ))

        print("rxy_pres_diffus components: max(d/dx(p'v')),  max(d/dy(p'u')) = ", max_ddx_pv, max_ddy_pu )

        print("")

        print("Kinetic energy balance equations")
        print("================================")
        if ( not hasattr(self, 'ke_dist') ):
            self.get_disturbance_kinetic_energy()

        self.ke_lhs = 2.*self.omega_i*self.ke_dist
        self.ke_rhs = 0.5*( self.rxx_rhs + self.ryy_rhs + self.rzz_rhs )

        print("ke_bal = ", np.amax(np.abs(self.ke_lhs-self.ke_rhs)))
        print("")

        self.get_model_terms()

        self.plot_balance_reynolds_stress(self.rxx_lhs,
                                          self.rxx_rhs,
                                          self.rxx_advection,
                                          self.rxx_production,
                                          self.rxx_pres_diffus,
                                          self.rxx_pres_diffus_model,
                                          self.rxx_pres_strain,
                                          self.rxx_pres_strain_model,
                                          self.rxx_pres_strain_model_girimaji,
                                          self.rxx_lap,
                                          self.rxx_dissip,
                                          self.rxx_grav_prod,
                                          self.solver.map.z,
                                          '$R_{xx}$-eq. balance'
                          )
        self.plot_balance_reynolds_stress(self.ryy_lhs,
                                          self.ryy_rhs,
                                          self.ryy_advection,
                                          self.ryy_production,
                                          self.ryy_pres_diffus,
                                          self.ryy_pres_diffus_model,
                                          self.ryy_pres_strain,
                                          self.ryy_pres_strain_model,
                                          self.ryy_pres_strain_model_girimaji,
                                          self.ryy_lap,
                                          self.ryy_dissip,
                                          self.ryy_grav_prod,
                                          self.solver.map.z,
                                          '$R_{yy}$-eq. balance'
                          )
        self.plot_balance_reynolds_stress(self.rzz_lhs,
                                          self.rzz_rhs,
                                          self.rzz_advection,
                                          self.rzz_production,
                                          self.rzz_pres_diffus,
                                          self.rzz_pres_diffus_model,
                                          self.rzz_pres_strain,
                                          self.rzz_pres_strain_model,
                                          self.rzz_pres_strain_model_girimaji,
                                          self.rzz_lap,
                                          self.rzz_dissip,
                                          self.rzz_grav_prod,
                                          self.solver.map.z,
                                          '$R_{zz}$-eq. balance'
                                          )
        self.plot_balance_reynolds_stress(self.rxy_lhs,
                                          self.rxy_rhs,
                                          self.rxy_advection,
                                          self.rxy_production,
                                          self.rxy_pres_diffus,
                                          self.rxy_pres_diffus_model,
                                          self.rxy_pres_strain,
                                          self.rxy_pres_strain_model,
                                          self.rxy_pres_strain_model_girimaji,
                                          self.rxy_lap,
                                          self.rxy_dissip,
                                          self.rxy_grav_prod,
                                          self.solver.map.z,
                                          '$R_{xy}$-eq. balance'
                                          )
        
        self.plot_reynolds_stresses(self.rxx,
                                  self.ryy,
                                  self.rzz,
                                  self.rxy,
                                  self.rxz,
                                  self.ryz,
                                  self.solver.map.z,
                                  'Reynolds stresses'
                          )

        
    def plot_balance_reynolds_stress(self, lhs, rhs, advection, production, pres_diffus, pres_diffus_model, pres_strain, pres_strain_model, \
                                     pres_strain_model_girimaji, laplacian, dissipation, grav_prod, z, str_bal):

        # Plot
        ax = plt.gca()

        # model flag
        flag_model = 0

        # lhs, rhs flag
        lr_flag = 1

        if (lr_flag==1):
            
            # Lhs
            if ( np.amax(np.abs(lhs.real)) > tol ):
                ax.plot( lhs.real,
                         z,
                         'r',
                         label=r"Lhs",
                         linewidth=1.5 )
            # Rhs
            if ( np.amax(np.abs(rhs.real)) > tol ):
                line1, = ax.plot( rhs.real,
                                  z,
                                  'k',
                                  label=r"Rhs",
                                  linewidth=1.5 )
                line1.set_dashes([4,4])
        #

        # Advection
        if ( np.amax(np.abs(advection.real)) > tol ):
            ax.plot( advection.real,
                     z,
                     'r',
                     label=r"Advection",
                     linewidth=1.5 )

        # Production
        if ( np.amax(np.abs(production.real)) > tol ):
            ax.plot( production.real,
                     z,
                     'b',
                     label=r"Production",
                     linewidth=1.5 )
            
        # Pressure diffusion
        if ( np.amax(np.abs(pres_diffus.real)) > tol ):
            ax.plot( pres_diffus.real,
                     z,
                     'b',
                     label=r"Press. Diff.",
                     linewidth=1.5
                    )
        # Pressure diffusion model
        if ( np.amax(np.abs(pres_diffus_model.real)) > tol and flag_model == 1 ):
            # Scale model pressure diffusion
            max_pd = np.max(np.abs(pres_diffus.real))
            max_pd_mod = np.max(np.abs(pres_diffus_model.real))
            scale_pd = max_pd/max_pd_mod    
            #label_pd_mod = "Press. Diff. model (%s)" % str('{:.2e}'.format(scale_pd))
            label_pd_mod = "Press. Diff. model"  
            
            #ax.plot( pres_diffus_model.real*scale_pd,
            ax.plot( pres_diffus_model.real,
                     z,
                     'bs',
                     markevery=uniform_val,
                     markerfacecolor='none',
                     markersize=marker_sz,
                     label=label_pd_mod
                    )
            
            # ax.plot( pres_diffus_model.real*scale_pd,
            # z,
            # 'gs',
            # markevery=uniform_val, 
            # #markerfacecolor='none',
            # markersize=marker_sz,
            # label=label_pd_mod)
            
        # Pressure strain
        if ( np.amax(np.abs(pres_strain.real)) > tol ):
            ax.plot( pres_strain.real,
                     z,
                     'c',
                     label=r"Press. Strain",
                     linewidth=1.5
                    )

        # Pressure strain model
        if ( str_bal == '$R_{zz}$-eq. balance' and hasattr(self, 'rzz_pres_strain_model_dani_new') and flag_model == 1 ):
            ax.plot( self.rzz_pres_strain_model_dani_new.real,
                     z,
                     'cs',
                     markerfacecolor='none',
                     markevery=uniform_val,
                     markersize=marker_sz,
                     label=r"Press. Strain model (inhomog.)"
                    )

        # Pressure strain model
        if ( np.amax(np.abs(pres_strain_model.real)) > tol and flag_model == 1 ):

            ax.plot( pres_strain_model.real,
                     z,
                     'cs',
                     markerfacecolor='none',
                     markevery=uniform_val,
                     markersize=marker_sz,
                     label=r"Press. Strain model"
                    )
            
            # ax.plot( pres_strain_model.real,
            #          z,
            #          'mo',
            #          markevery=uniform_val, 
            #          #markerfacecolor='none',
            #          markersize=marker_sz,
            #          label=r"Press. Strain model")
        #
        # if ( np.amax(np.abs(pres_strain_model_girimaji.real)) > tol ):

        #     line2, = ax.plot( pres_strain_model_girimaji.real,
        #                       z,
        #                       'm',
        #                       label=r"Press. Strain model (anis.)",
        #                       linewidth=1.5 )
        #     line2.set_dashes([2, 2, 10, 2])

            # ax.plot( pres_strain_model_girimaji.real,
            #          z,
            #          'ms',                    
            #          markevery=uniform_val, 
            #          #markerfacecolor='none',
            #          markersize=marker_sz,
            #          label=r"Press. Strain model (anis.)")
        #
        if ( np.amax(np.abs(laplacian.real)) > tol ):
            ax.plot( laplacian.real,
                     z,
                     'm',
                     label=r"Molecular Diff.", # "Laplacian"
                     linewidth=1.5 )
        
        if ( np.amax(np.abs(dissipation.real)) > tol ):
            ax.plot( -dissipation.real,
                     z,
                     'y',
                     label=r"Dissipation",
                    linewidth=1.5 )
        #
        if ( np.amax(np.abs(grav_prod.real)) > tol ):
            ax.plot( grav_prod.real,
                     z,
                     'm-.',
                     #color='grey',
                     #markevery=3,
                     #markerfacecolor='none',
                     label=r"Buoyant Prod.",
                     linewidth=1.5 )


        # Sum up pres_strain and pres_diff
        # if ( np.amax(np.abs(pres_strain.real+pres_diffus.real)) > tol ):
        #     ax.plot( pres_strain.real+pres_diffus.real,
        #              z,
        #              'k--',
        #              label=r"Ps + Pdif",
        #              linewidth=1.5 )


        # Pres. strain rxy from LRR (Wilcox book)
        # if ( np.amax(np.abs(self.rxy_pres_strain_model_lrr_)) > tol ):
        #     ax.plot( self.rxy_pres_strain_model_lrr_.real,
        #              z,
        #              'b--',
        #              label=r"self.rxy_pres_strain_model_lrr_",
        #              linewidth=1.5 )

        # Pres. strain rxy from LRR (Wilcox book) ===> ONLY STRAIN
        # if ( np.amax(np.abs(self.rxy_pres_strain_model_lrr_ONLY_STRAIN)) > tol ):
        #     ax.plot( self.rxy_pres_strain_model_lrr_ONLY_STRAIN.real,
        #              z,
        #              'g:',
        #              label=r"self.rxy_pres_strain_model_lrr_ONLY_STRAIN",
        #              linewidth=1.5 )

        # Plot ke_dist
        # if ( np.amax(np.abs(self.ke_dist)) > tol ):
        #     ax.plot( self.ke_dist.real,
        #              z,
        #              'b-.',
        #              label=r"self.ke_dist",
        #              linewidth=1.5 )

        # Plot pressure amplitude
        # if ( np.amax(np.abs(self.solver.peig)) > tol ):
        #     #ax.plot( np.abs(self.solver.peig),
        #     ax.plot( np.multiply(np.abs(self.solver.peig), self.solver.bsfl.Up),
        #              z,
        #              'c-.',
        #              label=r"np.abs(self.peig)",
        #              linewidth=1.5 )
            
        # Sum up pres_strain and pres_diff models
        # if ( np.amax(np.abs(pres_strain_model.real+pres_diffus_model.real)) > tol ):



        #     # Scale model
        #     max_ps_plus_pd = np.max(np.abs(pres_strain.real+pres_diffus.real))
        #     max_ps_plus_pd_mod = np.max(np.abs(pres_strain_model.real+pres_diffus_model.real+self.ryy.real))
        #     scale_ps_pd = max_ps_plus_pd/max_ps_plus_pd_mod
        #     label_ps_pd_mod = "Ps + Pdif model (%s) + Ryy" % str('{:.2e}'.format(scale_ps_pd))    

        #     line3, = ax.plot( 1.5*(pres_strain_model.real+pres_diffus_model.real+np.multiply(self.ryy.real, self.solver.bsfl.Up)),#*scale_ps_pd,
        #                       z,
        #                       'k',
        #                       #markevery=uniform_val,
        #                       #markersize=marker_sz,
        #                       label=label_ps_pd_mod,
        #                       linewidth=1.5 )
        #     line3.set_dashes([6,2])


        # # Plot -Rxx
        # if ( np.amax(np.abs(self.rxx)) > tol ):
        #     line4, = ax.plot( -self.rxx.real,
        #                       z,
        #                       'r',
        #                       label="-Rxx",
        #                       linewidth=1.5 )
        #     line4.set_dashes([2,2,2,2])


        DU = 0.5*( self.solver.bsfl.U[-1]-self.solver.bsfl.U[0] )
        
        # Plot Ryy*DU/l_taylor_fst
        # if ( np.amax(np.abs(self.ryy)) > tol ):
            
        #     line5, = ax.plot( self.ryy.real*DU/self.l_taylor_fst,
        #                       z,
        #                       'r',
        #                       label="Ryy*DU/l_taylor_fst",
        #                       linewidth=1.5 )
        #     line5.set_dashes([4,4,4,4])


        # Plot Ryy*Up
        # if ( np.amax(np.abs(self.ryy)) > tol ):
            
        #     line5, = ax.plot( np.multiply(self.ryy.real, self.solver.bsfl.Up),
        #                       z,
        #                       'r',
        #                       label="Ryy*Up",
        #                       linewidth=1.5 )
        #     line5.set_dashes([4,4,4,4])


            
        # # Plot dRxydy

        # Upp = np.matmul(self.solver.map.D2, self.solver.bsfl.U)
        # if ( np.amax(np.abs(self.rxy)) > tol ):
            
        #     line5, = ax.plot( -0.103*np.multiply( np.matmul(self.solver.map.D1, self.rxy.real), Upp),
        #                       z,
        #                       'grey',
        #                       label="Upp*dRxy/dy",
        #                       linewidth=1.5 )
        #     line5.set_dashes([4,4,4,4])

        #     line5, = ax.plot( 0.51*np.multiply( np.matmul(self.solver.map.D1, self.rxy.real), self.solver.bsfl.U),
        #                       z,
        #                       'grey',
        #                       label="U*dRxy/dy",
        #                       linewidth=1.5 )
        #     line5.set_dashes([1,1,1,1])

        # # Plot Rxy
        # if ( np.amax(np.abs(self.rxy)) > tol ):
        #     line6, = ax.plot( self.rxy.real,
        #                       z,
        #                       'r',
        #                       label="Rxy",
        #                       linewidth=1.5 )
        #     line6.set_dashes([1,1])

        # Plot Rxy+Ryy
        # if ( np.amax(np.abs(self.rxy+self.ryy)) > tol ):
        #     #####Up = np.matmul(self.solver.map.D1, self.solver.bsfl.U)
        #     line7, = ax.plot( 4.8*np.multiply(self.rxy.real, DU)/self.l_taylor_fst + np.multiply(self.ryy.real, DU)/self.l_taylor_fst,
        #                       z,
        #                       'g',
        #                       label="self.rxy.real*DU/self.l_taylor_fst+self.ryy.real*DU/self.l_taylor_fst",
        #                       linewidth=1.5 )
        #     line7.set_dashes([2,2,2,2])

        # # Plot Rxy+Ryy
        # if ( np.amax(np.abs(self.rxy+self.ryy)) > tol ):
        #     line7, = ax.plot( 1.1*pres_strain_model.real+self.ryy.real*DU/self.l_taylor_fst,
        #                       z,
        #                       'g',
        #                       label="1.1*pres_strain_model.real+self.ryy.real*DU/self.l_taylor_fst",
        #                       linewidth=1.5 )
        #     line7.set_dashes([2,2,2,2])

            
        hp, lp = ax.get_legend_handles_labels()

        if ( len(lp) > 0 ):
            title_lab = "%s" % str_bal #str('{:.2e}'.format(scale_pd))
            ax.set_xlabel(title_lab, fontsize=20)
            ax.set_ylabel(r'z', fontsize=20)
            
            plt.gcf().subplots_adjust(left=0.145)
            plt.gcf().subplots_adjust(bottom=0.145)
            
            #plt.xscale("log")
            
            zmin, zmax = mod_util.FindResonableZminZmax(lhs, z)

            # if ( len(lp) > 4 ):
            #     if ( len(lp)%2 == 0 ):
            #         ncolumns = len(lp)//2
            #     else:
            #         ncolumns = len(lp)//2+1
            # else:
            #     ncolumns = 4

            ncolumns = 4

            zmin = -4
            zmax = 4
            #ax.set_title('Eigenvalue spectrum')
            #ax.set_xlim([-10, 10])
            #ax.set_ylim(1.*zmin, 1.*zmax)
            ax.set_ylim(1.*zmin, 1.*zmax)
            ax.legend(loc='upper right', bbox_to_anchor=(1.14, 1.18),   # box with upper-right corner at (x, y)
            #ax.legend(loc='lower left', bbox_to_anchor=(0.4, 1.15),   # box with lower-left corner at (x, y)
                      ncol=1, fancybox=True, shadow=True, fontsize=10, framealpha = 0.8)
            plt.show()

    def plot_reynolds_stresses(self, rxx, ryy, rzz, rxy, rxz, ryz, z, str_bal):

        # Plot
        ax = plt.gca()

        if ( np.amax(np.abs(rxx.real)) > tol ):
            ax.plot( rxx.real,
                     z,
                     'k',
                     label=r"$R_{xx}$",
                     linewidth=1.5 )
        #
        if ( np.amax(np.abs(ryy.real)) > tol ):
            line1, = ax.plot( ryy.real,
                             z,
                             'r',
                             label=r"$R_{yy}$",
                             linewidth=1.5 )
            line1.set_dashes([4,4])
            # ax.plot( ryy.real,
            #          z,
            #          'r--',
            #          label=r"$R_{yy}$",
            #          linewidth=1.5 )
        #
        if ( np.amax(np.abs(rzz.real)) > tol ):
            ax.plot( rzz.real,
                     z,
                     'b',
                     label=r"$R_{zz}$",
                     linewidth=1.5 )

        #
        if ( np.amax(np.abs(rxy.real)) > tol ):
            ax.plot( rxy.real,
                     z,
                     'gs',
                     markevery=uniform_val,
                     #markerfacecolor='none',
                     markersize=marker_sz,
                     label=r"$R_{xy}$",
                     linewidth=1.5 )
        #
        if ( np.amax(np.abs(ryz.real)) > tol ):
            line, = ax.plot( ryz.real,
                             z,
                             'y-',
                             label=r"$R_{yz}$",
                             linewidth=1.5 )
            line.set_dashes([8, 4, 2, 4, 2, 4])
        #
        if ( np.amax(np.abs(rxz.real)) > tol ):
            ax.plot( rxz.real,
                     z,
                     ':',
                     color='orange',
                     label=r"$R_{xz}$",
                     linewidth=1.5 )
        #
        hp, lp = ax.get_legend_handles_labels()

        if ( len(lp) > 0 ):
            title_lab = "%s" % str_bal #str('{:.2e}'.format(scale_pd))
            ax.set_xlabel(title_lab, fontsize=20)
            ax.set_ylabel(r'z', fontsize=20)
            
            plt.gcf().subplots_adjust(left=0.145)
            plt.gcf().subplots_adjust(bottom=0.145)
            
            zmin, zmax = mod_util.FindResonableZminZmax(rxz, z)

            if ( len(lp) > 4 ):
                if ( len(lp)%2 == 0 ):
                    ncolumns = len(lp)//2
                else:
                    ncolumns = len(lp)//2+1
            else:
                ncolumns = 4

            #ax.set_title('Eigenvalue spectrum')
            #ax.set_xlim([-10, 10])
            ax.set_ylim(zmin, zmax)
            ax.legend(loc='upper center', bbox_to_anchor=(0.48, 1.15),   # box with lower left corner at (x, y)
                      ncol=ncolumns, fancybox=True, shadow=True, fontsize=10, framealpha = 0.8)
            plt.show()

    def compute_disturbance_dissipation_without_viscosity(self):

        # I solve non-dimensional, there use 1/Re for dissipation
        # But for length scales (integral and Taylor) I do not need to include viscosity here
        visc = 1/self.solver.Re
        testing = 0

        self.get_eigenfcts_derivatives()
        
        if testing == 1:
            print("")
            print("---------------> testing in compute_disturbance_dissipation")
            print("")

            nz = len(self.solver.ueig)
            
            self.dudx = np.random.rand(nz) + 1j*np.random.rand(nz)
            self.dudy = np.random.rand(nz) + 1j*np.random.rand(nz)
            self.dudz = np.random.rand(nz) + 1j*np.random.rand(nz)
        
            self.dvdx = np.random.rand(nz) + 1j*np.random.rand(nz)
            self.dvdy = np.random.rand(nz) + 1j*np.random.rand(nz)
            self.dvdz = np.random.rand(nz) + 1j*np.random.rand(nz)
            
            self.dwdx = np.random.rand(nz) + 1j*np.random.rand(nz)
            self.dwdy = np.random.rand(nz) + 1j*np.random.rand(nz)
            self.dwdz = np.random.rand(nz) + 1j*np.random.rand(nz)

        # Compute sub-groupings
        sub1 = self.dudy+self.dvdx
        sub2 = self.dvdz+self.dwdy
        sub3 = self.dudz+self.dwdx

        div  = self.dudx+self.dvdy+self.dwdz
    
        term1 = 2.*( self.compute_inner_prod(self.dudx, self.dudx) + self.compute_inner_prod(self.dvdy, self.dvdy) + self.compute_inner_prod(self.dwdz, self.dwdz) )
        term2 = -2./3.*self.compute_inner_prod(div, div)
        term3 = self.compute_inner_prod(sub1, sub1) + self.compute_inner_prod(sub2, sub2) + self.compute_inner_prod(sub3, sub3)
        
        epsilon_check = visc*( term1 + term2 + term3 )
        
        ########################################
        # Various components of the dissipation
        ########################################
        
        # 1) visc*(du_i/dx_j)*(du_i/dx_j)
        epsilon_tilde_dist = self.compute_inner_prod(self.dudx, self.dudx) + self.compute_inner_prod(self.dudy, self.dudy) + self.compute_inner_prod(self.dudz, self.dudz) \
            + self.compute_inner_prod(self.dvdx, self.dvdx) + self.compute_inner_prod(self.dvdy, self.dvdy) + self.compute_inner_prod(self.dvdz, self.dvdz) \
            + self.compute_inner_prod(self.dwdx, self.dwdx) + self.compute_inner_prod(self.dwdy, self.dwdy) + self.compute_inner_prod(self.dwdz, self.dwdz)
        epsilon_tilde_dist = visc*epsilon_tilde_dist
        
        # 2) visc*(du_i/dx_j)*(du_j/dx_i)
        epsilon_2tilde_dist = self.compute_inner_prod(self.dudx, self.dudx) + self.compute_inner_prod(self.dvdy, self.dvdy) + self.compute_inner_prod(self.dwdz, self.dwdz) \
            + 2.*self.compute_inner_prod(self.dudy, self.dvdx) + 2.*self.compute_inner_prod(self.dudz, self.dwdx) + 2.*self.compute_inner_prod(self.dvdz, self.dwdy)
        epsilon_2tilde_dist = visc*epsilon_2tilde_dist
        
        # 3) incompressible dissipation
        epsilon_incomp_dist = epsilon_tilde_dist + epsilon_2tilde_dist
        
        # 4) compressible part of dissipation
        epsilon_comp_dist = visc*term2
        
        # 5) full dissipation
        epsilon_dist = epsilon_incomp_dist + epsilon_comp_dist 
                
        if ( np.max(np.abs(epsilon_dist.imag)) > tol ):
            sys.exit("Disturbance dissipation should be real!")
        else:
            epsilon_dist = epsilon_dist.real
            
        # Integrate in vertical direction
        epsilon_int = mod_util.trapezoid_integration(epsilon_dist, self.solver.map.z)
        epsilon_int_check = mod_util.trapezoid_integration(epsilon_check, self.solver.map.z)
        if ( np.abs(epsilon_int.real-epsilon_int_check.real) > tol ):
            print("Check dissipation integral ----> epsilon_int, epsilon_check = ", epsilon_int, epsilon_int_check)
            sys.exit("Check dissipation!!!!!")
    
        #Plot_dissip_compos = 1
        #if (Plot_dissip_compos==1):
        #    self.plot_dissipation_components(epsilon_dist, epsilon_tilde_dist, self.ke_dist, self.l_taylor)

        self.epsilon_tilde_dist = epsilon_tilde_dist
        self.epsilon_dist = epsilon_dist

    def plot_dissipation_components(self):
        
        if ( not hasattr(self, 'l_taylor') ):
            self.compute_taylor_and_integral_scales()
                
        # GET ymin-ymax
        z = self.solver.map.z
        zmin, zmax = mod_util.FindResonableZminZmax(self.epsilon_dist, z)
        
        ptn = plt.gcf().number + 1
        
        f = plt.figure(ptn)
        ax = plt.subplot(111)
        
        # total dissipation
        plt.plot(np.real(self.epsilon_dist), z, 'k', linestyle='-', linewidth=1.5, label=r"total dissipation")
        # pseudo dissipation
        plt.plot(np.real(self.epsilon_tilde_dist), z, 'r', linestyle='-', linewidth=1.5, label="pseudo dissipation")
        # disturbance kinetic energy
        plt.plot(np.real(self.ke_dist), z, 'b', linestyle='-', linewidth=1.5, label="ke dist")
        # taylor length scale
        plt.plot(np.real(self.l_taylor), z, 'c', linestyle='-', linewidth=1.5, label="Taylor length scale")

        plt.xlabel('dissipation/tke/Taylor', fontsize=16)
        plt.ylabel('z', fontsize=16)
        plt.gcf().subplots_adjust(left=0.16)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.ylim([-4, 4])
        ax.legend(loc='upper center', bbox_to_anchor=(0.455, 1.17), fancybox=True, shadow=True, fontsize=10, framealpha=0.8)
        
        f.show()

        input("dissipation components")


    def compute_taylor_and_integral_scales(self):

        if ( not hasattr(self, 'ke_dist') ):
            self.get_disturbance_kinetic_energy()

        if ( not hasattr(self, 'epsilon_dist') ):
            self.get_eigenfcts_derivatives()
            self.compute_disturbance_dissipation_without_viscosity()

        visc = 1 # (nu_dim/nu_ref)
        
        # Compute Taylor scale (non-dimensional)
        #self.l_taylor = np.sqrt(15*visc*np.divide(self.ke_dist, self.epsilon_dist))
        self.l_taylor = np.sqrt(15*np.divide(self.ke_dist, self.epsilon_dist*self.solver.Re))
        
        # Integral length scale: https://en.wikipedia.org/wiki/Taylor_microscale, + see Daniel model.pdf
        # This is called turbulence length scale S by Schwartzkopf
        self.l_integr = np.divide(self.ke_dist**(1.5), self.epsilon_dist*self.solver.Re )*self.solver.Re

        #
        # Compute integral length scale by dimensionalizing non-dim ke and eps
        #
        
        # Time scale
        #t_ref  = bsfl_ref.Lref/bsfl_ref.Uref
        
        #ke_dim_check = ke*bsfl_ref.Uref**2.
        #eps_dim_check = eps*bsfl_ref.Uref**2./t_ref
        
        #l_integr_dim_check = np.divide(ke_dim_check**(1.5), eps_dim_check)
    
        # ------------------------------------------------
        # Plot taylor microscale and integral length scale
        # ------------------------------------------------
        self.plot_single_quantity(self.l_taylor, self.solver.map.z, 'Taylor length scale')
        self.plot_single_quantity(self.l_integr, self.solver.map.z, 'Integral length scale')
        
        #input("length scales")

    def plot_single_quantity(self, data_in, z, str_in):

        # Plot
        ax = plt.gca()
        ax.plot( data_in,
                 z,
                 'k',
                 linewidth=1.5 )
        
        title_lab = "%s" % str_in
        ax.set_xlabel(title_lab, fontsize=20)
        ax.set_ylabel(r'z', fontsize=20)
        
        plt.gcf().subplots_adjust(left=0.145)
        plt.gcf().subplots_adjust(bottom=0.145)
        
        zmin, zmax = mod_util.FindResonableZminZmax(data_in, z)

        #ax.set_title('Eigenvalue spectrum')
        #ax.set_xlim([-10, 10])
        ax.set_ylim(zmin, zmax)
        #ax.legend(loc='upper center', bbox_to_anchor=(0.48, 1.15),   # box with lower left corner at (x, y)
        #          ncol=ncolumns, fancybox=True, shadow=True, fontsize=10, framealpha = 0.8)
        plt.show()

    def SolvePoissonEquation1D(self):

        # Solve Linear System
        D2_with_bc = self.solver.map.D2
        
        D2_with_bc[0,:] = 0.0
        D2_with_bc[0,0] = 1.0
        
        D2_with_bc[-1,:] = 0.0
        D2_with_bc[-1,-1] = 1.0
        
        # Add diagonal elements
        #for i in range(0, nz):
        #    D2_with_bc[i,i] = D2_with_bc[i,i] #-alpha**2.
            
        vec_rhs = -2.*self.solver.bsfl.Up
        vec_rhs[0] = 0.
        vec_rhs[-1] = 0.
        
        SOL = linalg.solve(D2_with_bc, vec_rhs)

        sys.exit("Debugging poisson")

        # Plot
        ax = plt.gca()

        # 
        ax.plot( np.real(self.solver.peig),
                 self.solver.map.z,
                 'k--',
                 label=r"self.solver.peig",
                 linewidth=1.5 )

        ax.plot( np.real(SOL),
                 self.solver.map.z,
                 'k--',
                 label=r"SOL",
                 linewidth=1.5 )

        ax.set_xlabel(r'p', fontsize=20)
        ax.set_ylabel(r'z', fontsize=20)
        
        plt.gcf().subplots_adjust(left=0.145)
        plt.gcf().subplots_adjust(bottom=0.145)
            
        #ax.set_title('Eigenvalue spectrum')
        #ax.set_xlim([-10, 10])
        #ax.set_ylim(zmin, zmax)
        ax.legend(loc='lower left', bbox_to_anchor=(0.4, 1.15),   # box with lower-left corner at (x, y)
        ncol=1, fancybox=True, shadow=True, fontsize=10, framealpha = 0.8)
        plt.show()

    def compute_disturbance_dissipation_without_viscosity_non_dimensional_LOCAL_FREESTREAM(self, ueig, veig, weig, alpha, beta, zgrid):

        # Create non-dimensional differentiation matrix
        nz = len(zgrid)
        dz = zgrid[1]-zgrid[0]
        
        D1 = np.zeros((nz, nz), dp)
        D1[0,0] = -1./dz
        D1[0,1] = 1./dz
        
        for ii in range(1, nz-1):
            D1[ii,ii+1] = 1./(2.*dz)
            D1[ii,ii-1] = -1./(2.*dz)
            
        D1[-1,-1] = 1./dz
        D1[-1,-2] = -1./dz

        # Eigenfunctions derivatives
        ia = 1j*alpha
        ib = 1j*beta
        
        dudx = ia*ueig
        dudy = ib*ueig
        dudz = np.matmul(D1, ueig)
        
        dvdx = ia*veig
        dvdy = ib*veig
        dvdz = np.matmul(D1, veig)
        
        dwdx = ia*weig
        dwdy = ib*weig
        dwdz = np.matmul(D1, weig)
        
        sub1 = dudy+dvdx
        sub2 = dvdz+dwdy
        sub3 = dudz+dwdx
        
        div = dudx+dvdy+dwdz
        
        term1 = 2.*( self.compute_inner_prod(dudx, dudx) + self.compute_inner_prod(dvdy, dvdy) + self.compute_inner_prod(dwdz, dwdz) )
        term2 = -2./3.*self.compute_inner_prod(div, div)
        term3 = self.compute_inner_prod(sub1, sub1) + self.compute_inner_prod(sub2, sub2) + self.compute_inner_prod(sub3, sub3)
        
        ########################################
        # Various components of the dissipation
        ########################################
        
        # 1) visc*(du_i/dx_j)*(du_i/dx_j)
        epsilon_tilde_dist = self.compute_inner_prod(dudx, dudx) + self.compute_inner_prod(dudy, dudy) + self.compute_inner_prod(dudz, dudz) \
            + self.compute_inner_prod(dvdx, dvdx) + self.compute_inner_prod(dvdy, dvdy) + self.compute_inner_prod(dvdz, dvdz) \
            + self.compute_inner_prod(dwdx, dwdx) + self.compute_inner_prod(dwdy, dwdy) + self.compute_inner_prod(dwdz, dwdz)
        #epsilon_tilde_dist = visc*epsilon_tilde_dist
        
        # 2) visc*(du_i/dx_j)*(du_j/dx_i)
        epsilon_2tilde_dist = self.compute_inner_prod(dudx, dudx) + self.compute_inner_prod(dvdy, dvdy) + self.compute_inner_prod(dwdz, dwdz) \
            + 2.*self.compute_inner_prod(dudy, dvdx) + 2.*self.compute_inner_prod(dudz, dwdx) + 2.*self.compute_inner_prod(dvdz, dwdy)
        #epsilon_2tilde_dist = visc*epsilon_2tilde_dist
        
        # 3) incompressible dissipation
        epsilon_incomp_dist = epsilon_tilde_dist + epsilon_2tilde_dist
        
        # 4) compressible part of dissipation
        #epsilon_comp_dist = visc*term2
        epsilon_comp_dist = term2
        
        # 5) full dissipation
        epsilon_dist = epsilon_incomp_dist + epsilon_comp_dist 
        
        return epsilon_dist

class PostSingleFluid(PostProcess):
    def __init__(self, solver):
        """
        Constructor of class PostSingleFluid: That is the new one with shear-layer in z
        """
        super(PostSingleFluid, self).__init__(solver)

    def get_model_terms(self):

        # Constants
        C1 = 1.
        Cr1 = 1.
        
        if ( np.amax( np.abs(self.solver.bsfl.V) ) > 0. ):
             sys.exit("Check this: some modifications required for 3d baseflows!!")             

        if ( not hasattr(self, 'l_taylor') ):
            self.compute_taylor_and_integral_scales()

        if ( not hasattr(self, 'ke_dist') ):
            self.get_disturbance_kinetic_energy()

        # Compute freestream value of length scale
        self.compute_freestream_length_scales_non_dimensional()

        if ( np.min(np.abs(self.l_taylor-self.l_taylor_fst)) > 1.e-5 ):
            print("np.min(np.abs(self.l_taylor-self.l_taylor_fst)) = ", np.min(np.abs(self.l_taylor-self.l_taylor_fst)))
            sys.exit("Check self.l_taylor_fst (single fluid shear in z)!!!!!")

        # Set Taylor microscale
        if (taylor_opt==1):
            l_taylor = self.l_taylor_fst
        else:
            l_taylor = self.l_taylor

        if ( np.isnan(np.max(l_taylor)) ):
            sys.exit("l_taylor is nan ===> adjust domain height and re-run 111")

        l_taylor_nondim2 = np.multiply(l_taylor, l_taylor)

        # # Pressure strain model terms
        self.rxx_pres_strain_model = 4./3.*Cr1*np.multiply(self.rxz, self.solver.bsfl.Up)
        self.ryy_pres_strain_model = -2./3.*Cr1*np.multiply(self.rxz, self.solver.bsfl.Up)
        self.rzz_pres_strain_model = -2./3.*Cr1*np.multiply(self.rxz, self.solver.bsfl.Up)

        self.rxy_pres_strain_model = 0.0*self.solver.bsfl.U
        self.rxz_pres_strain_model = Cr1*np.multiply(self.rzz, self.solver.bsfl.Up)

        # C2 = 0.4
        # alpha_hat = 0.8 #(8.+C2)/11.
        # beta_hat = -0.10 #(8.*C2-2.)/11.
        # gam_hat = 0. #(60.*C2-4.)/55.
        
        # self.rxy_pres_strain_model_lrr_ = alpha_hat*np.multiply(self.ryy, self.solver.bsfl.Up) \
        #     + beta_hat*np.multiply(self.rxx, self.solver.bsfl.Up) \
        #     - 0.25*gam_hat*np.multiply(self.ke_dist, self.solver.bsfl.Up)
        # self.rxy_pres_strain_model_lrr_ = self.rxy_pres_strain_model_lrr_*0.77

        # # Strain only using LRR
        # C2 = 0.4
        # alpha_hat = -1.5 #2. #(8.+C2)/11.
        # beta_hat = 1.#-1. #(8.*C2-2.)/11.
        # gam_hat = 2. #(60.*C2-4.)/55.
        
        # self.rxy_pres_strain_model_lrr_ONLY_STRAIN = alpha_hat*np.multiply(self.ryy, self.solver.bsfl.Up) \
        #     + beta_hat*np.multiply(self.rxx, self.solver.bsfl.Up) \
        #     - 0.25*gam_hat*np.multiply(self.ke_dist, self.solver.bsfl.Up)
        
        # Set to zero 
        self.rxx_pres_strain_model_girimaji = 0.0*self.rxx_pres_strain_model
        self.ryy_pres_strain_model_girimaji = 0.0*self.rxx_pres_strain_model
        self.rzz_pres_strain_model_girimaji = 0.0*self.rxx_pres_strain_model

        self.rxy_pres_strain_model_girimaji = 0.0*self.rxx_pres_strain_model
        self.rxz_pres_strain_model_girimaji = 0.0*self.rxx_pres_strain_model
        
        # Pressure diffusion model terms
        self.rxx_pres_diffus_model = 0.0*self.rxx_pres_strain_model

        # I added a minus here
        self.ryy_pres_diffus_model = 0.0*self.rzz_pres_strain_model
        self.rzz_pres_diffus_model = -4.*C1*np.matmul(self.solver.map.D1, np.multiply(np.multiply(self.drxzdz, self.solver.bsfl.Up), l_taylor_nondim2))

        self.rxy_pres_diffus_model = 0.0*C1*np.matmul(self.solver.map.D1, np.multiply(np.multiply(self.drxxdz, self.solver.bsfl.Up), l_taylor_nondim2))
        self.rxz_pres_diffus_model = C1*np.matmul(self.solver.map.D1, np.multiply(np.multiply(self.drxxdz, self.solver.bsfl.Up), l_taylor_nondim2))


        #
        # Output
        #

        z = self.solver.map.z

        # # Rxx equation: terms
        # term1_rxx_out = 4./3.*np.multiply(self.rxy, self.solver.bsfl.Up)
        # # np.multiply(self.rxx, self.solver.bsfl.Up) == this term does not work very well, it cannot capture freestream very well
        # term2_rxx_out = self.rxx # Adding this term based on intuition & a-posteriori optimization & the fact that in freestream it should look like Rxx
        # term3_rxx_out = self.ryy 
        # term4_rxx_out = self.rxy 

        # # Rxx equation: write data
        # data_out = np.column_stack([z, np.real(term1_rxx_out), np.real(term2_rxx_out), np.real(term3_rxx_out), np.real(term4_rxx_out), np.real(self.rxx_pres_strain)])
        # datafile_path = "/Users/aph/Desktop/pres_strain_vars_z_fz_gz_hz_iz_exact_single_fluid_rxx.dat" 
        # np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])

        # # Ryy equation: terms
        # term1_ryy_out = (-2./3.)*np.multiply(self.rxy, self.solver.bsfl.Up)
        # term2_ryy_out = self.rxx
        # term3_ryy_out = self.ryy
        # term4_ryy_out = self.rxy

        # # Ryy equation: write data
        # data_out = np.column_stack([z, np.real(term1_ryy_out), np.real(term2_ryy_out), np.real(term3_ryy_out), np.real(term4_ryy_out), np.real(self.ryy_pres_strain)])
        # datafile_path = "/Users/aph/Desktop/pres_strain_vars_z_fz_gz_hz_iz_exact_single_fluid_ryy.dat" 
        # np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])

        # # Rxy equation: terms
        # term1_rxy_out = np.multiply(self.ryy, self.solver.bsfl.Up)
        # term2_rxy_out = self.rxx
        # term3_rxy_out = self.ryy
        # term4_rxy_out = self.rxy

        # # Rxy equation: write data
        # data_out = np.column_stack([z, np.real(term1_rxy_out), np.real(term2_rxy_out), np.real(term3_rxy_out), np.real(term4_rxy_out), np.real(self.rxy_pres_strain)])
        # datafile_path = "/Users/aph/Desktop/pres_strain_vars_z_fz_gz_hz_iz_exact_single_fluid_rxy.dat" 
        # np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])

    def compute_freestream_length_scales_non_dimensional(self):

        #
        # THIS IS FOR SHEAR-LAYER IN Z
        #
        
        z = self.solver.map.z
        kx = self.solver.alpha[0]
        ky = self.solver.beta[0]
        omega = self.solver.omega_max

        if ( np.amax( np.abs(self.solver.bsfl.V) ) > 0. ):
             sys.exit("Check this: so far only for V=0")             

        Re = self.solver.Re
        
        # Create local grid
        nz = 1001
        zmin = z[-1]-z[-1]/20
        zmax = z[-1]
        zgrid = np.linspace(zmin, zmax, nz)

        #
        # Matrix entries
        #
        Mat = np.zeros((6, 6), dpc)
        #
        Mat[0,1] = 1.
        #
        Mat[1,0] = kx**2 + ky**2 + 1j*kx*Re -1j*omega*Re
        Mat[1,5] = 1j*kx*Re
        #
        Mat[2,3] = 1.
        #
        Mat[3,2] = (kx**2+ky**2) + 1j*kx*Re -1j*omega*Re
        Mat[3,5] = 1j*ky*Re
        #
        Mat[4,0] = -1j*kx
        Mat[4,2] = -1j*ky
        #
        Mat[5,1] = -1j*kx/Re
        Mat[5,3] = -1j*ky/Re
        Mat[5,4] = 1j*omega-1j*kx-1/Re*(kx**2+ky**2)

        #
        # Compute eigenvalues
        #
        D, V = linalg.eig(Mat)        
        eigvals = D
        
        tols = 1e-10
        
        # find indices of eigenvalues with negative real parts
        idx_array = np.zeros(3, i4)
        
        ct = 0
        
        for i in range(0, np.size(Mat, 0)):
            if ( eigvals[i].real < 0.0 and np.abs(eigvals[i].real) > tols ):
                idx_array[ct] = i
                ct = ct + 1

        Lam1 = eigvals[idx_array[0]]
        Lam2 = eigvals[idx_array[1]]
        Lam3 = eigvals[idx_array[2]]
        
        uvec = np.zeros(nz,dpc)
        vvec = np.zeros(nz,dpc)
        wvec = np.zeros(nz,dpc)
        pvec = np.zeros(nz,dpc)
        
        ke = np.zeros(nz, dp)

        # str = ['u','v','w','p'];
        # vars = [0 2 4 5];
    
        for k in range(0, nz):
            
            zloc = zgrid[k]
            
            q = V[:,idx_array[0]]*np.exp(Lam1*zloc) + \
                V[:,idx_array[1]]*np.exp(Lam2*zloc) + \
                V[:,idx_array[2]]*np.exp(Lam3*zloc)
            
            uvec[k] = q[0]
            vvec[k] = q[2]
            wvec[k] = q[4]
            pvec[k] = q[5]
            
            # u -> index 0
            uu = self.compute_inner_prod(q[0], q[0])
            # v -> index 2
            vv = self.compute_inner_prod(q[2], q[2])
            # w -> index 4
            ww = self.compute_inner_prod(q[4], q[4])
            
            # Compute disturbance kinetic energy
            ke[k] = 0.5*( uu + vv + ww ).real

        # Get no viscosity non-dimensional dissipation 
        eps_no_visc = self.compute_disturbance_dissipation_without_viscosity_non_dimensional_LOCAL_FREESTREAM(uvec, vvec, wvec, kx, ky, zgrid)
                
        # Compute the length scales
        l_taylor_fst = np.sqrt(np.divide(15.*ke, eps_no_visc))
        if ( np.abs(l_taylor_fst[-20]-l_taylor_fst[-21]) > 1.0e-7 ):
            sys.exit("Something weird for Freestream Taylor scale (Single Fluid)")

        self.l_taylor_fst = l_taylor_fst[-20].real
        
        print("Freestream Taylor scale (Single Fluid, Shear in z): ", self.l_taylor_fst)

        
class PostRayleighTaylor(PostProcess):
    def __init__(self, solver):
        """
        Constructor of class PostRayleighTaylor
        """
        super(PostRayleighTaylor, self).__init__(solver)

    def get_model_terms(self):

        #print("")
        #print("RT model terms")

        # Constants
        B = 0.088 #1.
        Cr1 = 1./4.
        
        if ( not hasattr(self, 'l_taylor') ):
            self.compute_taylor_and_integral_scales()

        if ( not hasattr(self, 'ke_dist') ):
            self.get_disturbance_kinetic_energy()

        self.compute_freestream_length_scales_non_dimensional()

        if ( np.min(np.abs(self.l_taylor-self.l_taylor_fst)) > 1.e-5 ):
            print("np.min(np.abs(self.l_taylor-self.l_taylor_fst)) = ", np.min(np.abs(self.l_taylor-self.l_taylor_fst)))
            sys.exit("Check self.l_taylor_fst!!!!!")

        # Set Taylor microscale
        if (taylor_opt==1):
            l_taylor = self.l_taylor_fst
        else:
            l_taylor = self.l_taylor

        if ( np.isnan(np.max(l_taylor)) ):
            sys.exit("l_taylor is nan ===> adjust domain height and re-run 222")

        l_taylor_nondim2 = np.multiply(l_taylor, l_taylor)

        # Baseflow density (non-dimensional)
        Rho = self.solver.bsfl.Rho_nd

        # Terms related to a-equation
        az  = np.divide(self.compute_inner_prod(self.solver.reig, self.solver.weig), Rho)
        dazdz = np.matmul(self.solver.map.D1, az)
        dpdz_bsfl = -Rho/self.solver.Fr**2*1

        # Pressure strain model terms
        self.rxx_pres_strain_model = 2./3.*Cr1*np.multiply(az, dpdz_bsfl)
        self.ryy_pres_strain_model = 0.0*self.rxx
        self.rzz_pres_strain_model = 4./3.*Cr1*az/self.solver.Fr**2.
        self.rxy_pres_strain_model = 0.0*self.rxx

        # Cr1__
        # For Reh = 1000
        #Cr1__ = -0.15
        #Cr1_star = 0.25

        # For Reh = 100
        #Cr1__ = -0.2
        #Cr1_star = 0.12

        # Trying to get something for both Reh=100 and Reh=1000
        Cr1__ = -0.175
        Cr1_star = 0.19
        
        # For Reh = 100
        #Cr1__ = -0.23
        #Cr1_star = 0.13

        # For Reh = 10
        #Cr1__ = -5./20.
        #Cr1_star = 0.6/20. 
        
        d2azdz2 = np.matmul(self.solver.map.D2, az)
        self.rzz_pres_strain_model_dani_new = -4./3.*Cr1__*az/self.solver.Fr**2. + 2./3.*Cr1_star*l_taylor_nondim2*d2azdz2

        data_out = np.column_stack([self.solver.map.z, np.real(-4./3.*az/self.solver.Fr**2.), np.real(2./3.*l_taylor_nondim2*d2azdz2), np.real(self.rzz_pres_strain)])
        datafile_path = "/Users/aph/Desktop/pres_strain_vars_y_fz_gz_exact.dat" 
        np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e'])

        # Pressure strain model from Girimaji (anisotropic)
        C5 = 1.3
        C1_star = -2.*C5 #-3
        
        ke_dist_no_bc = self.ke_dist
        ke_dist_no_bc[0] = ke_dist_no_bc[1]
        ke_dist_no_bc[-1] = ke_dist_no_bc[-2]
        bzz = np.real( np.divide( self.rzz, 2.*ke_dist_no_bc ) ) - 1./3.

        self.rzz_pres_strain_model_girimaji = np.multiply( (C1_star*bzz + 4./3.*C5), np.real(az))/self.solver.Fr**2.
        self.rxx_pres_strain_model_girimaji = 0.0*self.rzz_pres_strain_model_girimaji
        self.ryy_pres_strain_model_girimaji = 0.0*self.rzz_pres_strain_model_girimaji
        self.rxy_pres_strain_model_girimaji = 0.0*self.rzz_pres_strain_model_girimaji

        # Pressure diffusion model terms
        self.rxx_pres_diffus_model = 0.0*self.rxx
        self.ryy_pres_diffus_model = 0.0*self.rxx
        self.rxy_pres_diffus_model = 0.0*self.rxx

        # AHA: I added a minus sign here!!!
        self.rzz_pres_diffus_model = -2./self.solver.Fr**2.*B*np.matmul(self.solver.map.D1, np.multiply(l_taylor_nondim2, dazdz))

        # Plot pressure strain exact and models
        plot_label = r"Press. Strain $R_{zz}$-eq."
        self.plot_rzz_eq_pres_strain_exact_model(plot_label, self.rzz_pres_strain, self.rzz_pres_strain_model, self.rzz_pres_strain_model_dani_new)

    def plot_rzz_eq_pres_strain_exact_model(self, plot_label, rzz_pres_strain, rzz_pres_strain_model, rzz_pres_strain_model_dani_new):

        #########################################
        # Rzz Pressure strain models comparison #
        #########################################
        
        # GET ymin-ymax
        y = self.solver.map.z
        ymin, ymax = mod_util.FindResonableZminZmax(rzz_pres_strain, y)
        
        ptn = plt.gcf().number + 1
        
        f = plt.figure(ptn)
        ax = plt.subplot(111)

        # Exact pressure strain
        plt.plot(np.real(rzz_pres_strain), y, 'k', linestyle='-', linewidth=1.5, label=r"Exact")
        # Pressure strain model (homogeneous)
        plt.plot(np.real(rzz_pres_strain_model), y, 'ks', markerfacecolor='none', markevery=uniform_val, markersize=marker_sz, label="Press. Strain model (homogeneous)")
        # Pressure strain model (inhomogeneous)
        plt.plot(np.real(rzz_pres_strain_model_dani_new), y, 'r+', markerfacecolor='none', markevery=uniform_val, markersize=marker_sz, label="Press. Strain model (inhomogeneous)")
        #
        plt.xlabel(plot_label, fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.gcf().subplots_adjust(left=0.16)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.ylim([-4, 4])
        #ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.15), ncol=1, fancybox=True, shadow=True, fontsize=14)
        #ax.legend(loc='upper center', bbox_to_anchor=(0.75, 1.15), ncol=1, fancybox=True, shadow=True, fontsize=14)
        ax.legend(loc='upper center', bbox_to_anchor=(0.455, 1.17), fancybox=True, shadow=True, fontsize=10, framealpha=0.8)
        
        f.show()

        #data_out = np.column_stack([y, np.real(rzz_pres_strain)])
        #datafile_path = "/Users/aph/Desktop/pres_strain_exact.dat" 
        #np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e'])

        input("Rzz Press. Strain (Exact+Models)")

    def plot_az_eq_pres_dissip_exact_model(self, plot_label, az_eq_pres_dissip, az_eq_pres_dissip_model, az_eq_pres_dissip_mod_daniel_new):

        ##########################################
        # Pressure dissipation models comparison #
        ##########################################
        
        # GET ymin-ymax
        y = self.solver.map.z
        ymin, ymax = mod_util.FindResonableZminZmax(az_eq_pres_dissip, y)
        
        ptn = plt.gcf().number + 1
        
        f = plt.figure(ptn)
        ax = plt.subplot(111)

        # Exact pressure dissipation
        plt.plot(np.real(az_eq_pres_dissip), y, 'k', linestyle='-', linewidth=1.5, label=r"Exact")
        # Pressure dissipation model (homogeneous)
        plt.plot(np.real(az_eq_pres_dissip_model), y, 'ks', markerfacecolor='none', markevery=uniform_val, markersize=marker_sz, label="Press. Dissipation model (homogeneous)")
        # Pressure dissipation model (inhomogeneous)
        plt.plot(np.real(az_eq_pres_dissip_mod_daniel_new), y, 'r+', markerfacecolor='none', markevery=uniform_val, markersize=marker_sz, label="Press. Dissipation model (inhomogeneous)")
        #
        plt.xlabel(plot_label, fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.gcf().subplots_adjust(left=0.16)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.ylim([-4, 4])
        ax.legend(loc='upper center', bbox_to_anchor=(0.455, 1.17), fancybox=True, shadow=True, fontsize=10, framealpha=0.8)
        
        f.show()

        input("Press. Dissipation (Exact+Models)")

    def compute_freestream_length_scales_non_dimensional(self):

        y = self.solver.map.z
        kx = self.solver.alpha[0]
        ky = self.solver.beta[0]
        omega = self.solver.omega_max

        Re = self.solver.Re
        Sc = self.solver.Sc
        Fr = self.solver.Fr

        Pem_inv = Re*Sc
        
        # Create local grid
        nz = 1001
        zmin = y[-1]-y[-1]/20
        zmax = y[-1]
        zgrid = np.linspace(zmin, zmax, nz)        

        #
        # Matrix entries
        #
        Mat = np.zeros((9, 9), dpc)
        #
        Mat[0,1] = 1.
        #
        Mat[1,0] = kx**2 + ky**2 -1j*omega*Re
        Mat[1,6] = 1j*kx*Re
        #
        Mat[2,3] = 1.
        #
        Mat[3,2] = kx**2 + ky**2 -1j*omega*Re
        Mat[3,6] = 1j*ky*Re
        #
        Mat[4,5] = 1.
        #
        Mat[5,1] = -1j*kx
        Mat[5,3] = -1j*ky
        #
        Mat[6,1] = -1j*kx/Re
        Mat[6,3] = -1j*ky/Re
        Mat[6,4] = 1j*omega-(kx**2+ky**2)/Re                
        Mat[6,7] = -1/Fr**2
        #
        Mat[7,8] = 1.
        #
        Mat[8,7] = kx**2 + ky**2 -1j*omega*Pem_inv

        #
        # Compute eigenvalues
        #
        D, V = linalg.eig(Mat)        
        eigvals = D
        
        tols = 1e-10
        
        # find indices of eigenvalues with negative real parts
        idx_array = np.zeros(4, i4)
        
        ct = 0
        
        for i in range(0, np.size(Mat, 0)): #i=1:size(Mat,1)
            if ( eigvals[i].real < 0.0 and np.abs(eigvals[i].real) > tols ):
                idx_array[ct] = i
                ct = ct + 1

        Lam1 = eigvals[idx_array[0]]
        Lam2 = eigvals[idx_array[1]]
        Lam3 = eigvals[idx_array[2]]
        Lam4 = eigvals[idx_array[3]]
        
        uvec = np.zeros(nz,dpc)
        vvec = np.zeros(nz,dpc)
        wvec = np.zeros(nz,dpc)
        pvec = np.zeros(nz,dpc)
        rvec = np.zeros(nz,dpc)
        
        ke = np.zeros(nz, dp)
        
        # str = ['u','v','w','p','r']
        # vars = [0 2 4 6 7]
        
        for k in range(0, nz): #k=1:nz
            
            zloc = zgrid[k]
            
            q = V[:,idx_array[0]]*np.exp(Lam1*zloc) + \
                V[:,idx_array[1]]*np.exp(Lam2*zloc) + \
                V[:,idx_array[2]]*np.exp(Lam3*zloc) + \
                V[:,idx_array[3]]*np.exp(Lam4*zloc)
            
            uvec[k] = q[0]
            vvec[k] = q[2]
            wvec[k] = q[4]
            pvec[k] = q[6]
            rvec[k] = q[7]
            
            # u -> index 0
            uu = self.compute_inner_prod(q[0], q[0])
            # v -> index 2
            vv = self.compute_inner_prod(q[2], q[2])
            # w -> index 4
            ww = self.compute_inner_prod(q[4], q[4])
            
            # Compute disturbance kinetic energy
            ke[k] = 0.5*( uu + vv + ww ).real

        # Get no viscosity non-dimensional dissipation 
        eps_no_visc = self.compute_disturbance_dissipation_without_viscosity_non_dimensional_LOCAL_FREESTREAM(uvec, vvec, wvec, kx, ky, zgrid)
                
        # Compute the length scales
        l_taylor_fst = np.sqrt(np.divide(15.*ke, eps_no_visc))
        if ( np.abs(l_taylor_fst[-20]-l_taylor_fst[-21]) > 1.0e-7 ):
            sys.exit("Something weird for Freestream Taylor scale")

        self.l_taylor_fst = l_taylor_fst[-20].real
        
        #l_integr_fst = np.divide(ke_dim**(3./2.), eps_dim)
        #eps_dim_r_ke_dim = np.divide(eps_dim, ke_dim)
        #eps_r_ke = eps_dim_r_ke_dim*tref
        #eps_dim_r_ke_dim_fst = eps_dim_r_ke_dim[-20]
        #eps_r_ke_fst = eps_r_ke[-20]
        #l_integr_fst = l_integr_fst[-20].real
        
        print("Freestream Taylor scale (RT): ", self.l_taylor_fst)
        
    
    #def compute_a_eq_ener_bal_boussi_Sc_1_NEW(ueig, veig, weig, peig, reig, D1, D2, y, alpha, beta, omega_i, bsfl, bsfl_ref, mob, rt_flag, KE_dist, epsilon_dist, epsil_tild_min_2tild_dist, iarr):
    def get_balance_az_equation_boussinesq_sc_1(self, omega_i):

        y = self.solver.map.z
        Re = self.solver.Re
        Fr = self.solver.Fr
        ny = len(y)

        Rho = self.solver.bsfl.Rho_nd
        Rhop = self.solver.bsfl.Rhop_nd
        
        div = ( self.dudx + self.dvdy + self.dwdz )

        if ( not hasattr(self, 'dudx') ):
            self.get_eigenfcts_derivatives()

        if ( not hasattr(self, 'l_taylor') ):
            self.compute_taylor_and_integral_scales()

        if ( not hasattr(self, 'ke_dist') ):
            self.get_disturbance_kinetic_energy()

        self.compute_freestream_length_scales_non_dimensional()

        if ( np.min(np.abs(self.l_taylor-self.l_taylor_fst)) > 1.e-5 ):
            print("np.min(np.abs(self.l_taylor-self.l_taylor_fst)) = ", np.min(np.abs(self.l_taylor-self.l_taylor_fst)))
            sys.exit("Check self.l_taylor_fst (single fluid)!!!!!")

        # Set Taylor microscale
        if (taylor_opt==1):
            l_taylor = self.l_taylor_fst
        else:
            l_taylor = self.l_taylor

        if ( np.isnan(np.max(l_taylor)) ):
            sys.exit("l_taylor is nan ===> adjust domain height and re-run 333")

        l_taylor_nondim2 = np.multiply(l_taylor, l_taylor)
        
        # LHS for a-equation
        self.az_eq_lhs = 2*omega_i*self.compute_inner_prod(self.solver.reig, self.solver.weig)

        #
        # RHS: viscous terms
        #

        # Compute Laplacian of rho'w'
        self.az_eq_lap = 2.0*( self.compute_inner_prod(self.drdx, self.dwdx) + self.compute_inner_prod(self.drdy,self.dwdy) + self.compute_inner_prod(self.drdz, self.dwdz) ) + \
            self.compute_inner_prod(self.solver.reig, self.d2wdx2) + self.compute_inner_prod(self.solver.reig, self.d2wdy2) + self.compute_inner_prod(self.solver.reig, self.d2wdz2) + \
            self.compute_inner_prod(self.solver.weig, self.d2rdx2) + self.compute_inner_prod(self.solver.weig, self.d2rdy2) + self.compute_inner_prod(self.solver.weig, self.d2rdz2) 
        self.az_eq_lap = self.az_eq_lap/Re
        
        # Compute "dissipation" of a-equation
        self.az_eq_dissip = 2.*( self.compute_inner_prod(self.drdx, self.dwdx) + self.compute_inner_prod(self.drdy, self.dwdy) + self.compute_inner_prod(self.drdz, self.dwdz) )
        self.az_eq_dissip = self.az_eq_dissip/Re

        #
        # RHS: inviscid terms
        #

        # Pressure diffusion
        RHS_a_t11 = -self.compute_inner_prod(self.solver.reig, self.dpdz)-self.compute_inner_prod(self.solver.peig, self.drdz)
        # Pressure dissipation
        RHS_a_t12 = self.compute_inner_prod(self.solver.peig, self.drdz)

        # Stratified production
        RHS_a_t2 = -np.multiply(Rhop, self.compute_inner_prod(self.solver.weig, self.solver.weig))
        
        RHS_a_t3 = -self.compute_inner_prod(self.solver.reig, self.solver.reig)/Fr**2.
        
        self.az_eq_rhs  = RHS_a_t11 + RHS_a_t12 + RHS_a_t2 + RHS_a_t3 + self.az_eq_lap - self.az_eq_dissip
        
        energy_bal_az_Eq  = np.amax(np.abs(self.az_eq_lhs-self.az_eq_rhs))
        print("Energy balance (RT) a-equation (BOUSSINESQ ----> ONLY FOR Sc=1):", energy_bal_az_Eq)

        ####################
        # Model Terms
        ####################
        dpdz_bsfl = -Rho/Fr**2*1
        rho2    = np.multiply(Rho, Rho)
        
        az = np.divide(self.compute_inner_prod(self.solver.reig, self.solver.weig), Rho)
        b  = np.divide(self.compute_inner_prod(self.solver.reig, self.solver.reig), rho2 )
        dbdz = np.matmul(self.solver.map.D1, b)

        # Added negative sign here
        Cc1 = -1
        self.az_eq_pres_dissipation_model = Cc1*1./3.*np.multiply(b, dpdz_bsfl)

        G1 = 0.10755
        self.az_eq_pres_diffusion_model = -G1*np.matmul(self.solver.map.D1, np.multiply(l_taylor_nondim2, dbdz))
        
        # Use Girimaji for the pressure dissipation term
        C5 = 5.5
        C4 = -0.5*C5
        
        bzz = np.real(np.divide(self.rzz, 2.*self.ke_dist)) - 1./3.
        self.az_eq_pres_dissip_mod_az_eq_girimaji = np.multiply( (C4*bzz + 1./3.*C5), np.multiply(Rho, b))/Fr**2.

        # New pressure dissipation model from Daniel
        #E2 = 0.1 # For Reh=100 with l_taylor_fst (taylor_opt=1)
        #Cn1 = 0.8

        #E2 = 0.3 # For Reh=1000 with l_taylor_fst (taylor_opt=1)
        #Cn1 = 0.4

        #E2 = 0.05 # For Reh=1000 with l_taylor (not l_taylor_fst)
        #Cn1 = 0.6

        #Cn1 = 1.
        #E2 = 0.2

        Cn1 = 1.
        E2 = 0.18

        d2bdz2 = np.matmul(self.solver.map.D2, b)
        self.az_eq_pres_dissip_mod_daniel_new = Cn1*( b/3. + 2./3*E2*np.multiply( d2bdz2, l_taylor_nondim2 ) )/Fr**2.

        # Plot pressure dissipation exact and models
        plot_label = r"Press. Dissipation $a_{z}$-eq."
        self.plot_az_eq_pres_dissip_exact_model(plot_label, RHS_a_t12, self.az_eq_pres_dissipation_model, self.az_eq_pres_dissip_mod_daniel_new)

        # idx = np.argmax(np.abs(self.az_eq_pres_dissip_mod_daniel_new))
        # print("idx = ", idx)

        # qz = RHS_a_t12
        # fz = b/(3.*Fr**2.)
        # gz = 2./3*np.multiply( d2bdz2, l_taylor_nondim2 )/Fr**2
        # acoef_check = np.divide( (qz-fz), gz)
        # print("acoef_check[ny//2] = ", acoef_check[ny//2])
        # print("acoef_check[idx] = ", acoef_check[idx])
                
        ####################
        # PLOTS
        ####################
        
        # GET ymin-ymax
        ymin, ymax = mod_util.FindResonableZminZmax(self.az_eq_rhs, y)
        
        ptn = plt.gcf().number + 1
        
        f = plt.figure(ptn)
        ax = plt.subplot(111)
        #
        flag_lr_a = 0
        if (flag_lr_a==1):
            plt.plot(np.real(self.az_eq_lhs), y, 'r', linestyle='-', linewidth=1.5, label=r"Lhs")
            plt.plot(np.real(self.az_eq_rhs), y, 'k', linestyle='--', dashes=(5, 5), linewidth=1.5, label=r"Rhs")
        #
        # pressure diffusion
        plt.plot(np.real(RHS_a_t11), y, 'b', linestyle='-', linewidth=1.5, label=r"Press. Diff.")
        # pressure diffusion model
        max_pd = np.max(np.abs(RHS_a_t11)); max_pd_model = np.max(np.abs(self.az_eq_pres_diffusion_model)); scale_pd = max_pd/max_pd_model
        #label_pd_mod = "Press. Diff. model (%s)" % str('{:.2e}'.format(scale_pd))
        label_pd_mod = "Press. Diff. model"
        plt.plot(np.real(self.az_eq_pres_diffusion_model), y, 'bs', markerfacecolor='none', markevery=uniform_val, markersize=marker_sz, label=label_pd_mod)
        #plt.plot(np.real(self.az_eq_pres_diffusion_model)*scale_pd, y, 'bs', markerfacecolor='none', markevery=uniform_val, markersize=marker_sz, label=label_pd_mod)

        # az
        plt.plot(np.real(az), y, 'rd', markerfacecolor='none', markevery=uniform_val, markersize=marker_sz, label=r"az")

        print("np.min(RHS_a_t12) = ", np.min(RHS_a_t12))
        
        # pressure dissipation
        plt.plot(np.real(RHS_a_t12), y, 'cs', markerfacecolor='none', markevery=uniform_val, markersize=marker_sz, label=r"Press. Dissip.")
        # pressure dissipation model
        #plt.plot(np.real(self.az_eq_pres_dissip_mod_az_eq_girimaji), y, 'c', linestyle='-', linewidth=1.5, label="Anisotropic pres. diss. model")
        plt.plot(np.real(self.az_eq_pres_dissip_mod_daniel_new), y, 'c', linestyle='-', linewidth=1.5, label="Press. Dissip. model (inhomog.)")
        
        #
        plt.plot(np.real(RHS_a_t2), y, 'g', linewidth=1.5, label=r"Stratified prod.")
        plt.plot(np.real(RHS_a_t3), y, 'y', linewidth=1.5, label=r"Buoyant prod.") #$-\overline{\rho'\rho'}g/Fr^2$")
        plt.plot(np.real(self.az_eq_lap),    y, 'm', linestyle='-.', linewidth=1.5, label=r"Molecular Diff.") # "Laplacian"
        plt.plot(np.real(-self.az_eq_dissip), y, 'm', linewidth=1.5, label=r"Dissipation")    
        #
        #
        plt.xlabel(r"a-eq. balance (Sc=1)", fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.gcf().subplots_adjust(left=0.16)
        plt.gcf().subplots_adjust(bottom=0.15)
        #plt.xlim([-0.12, 0.06])
        plt.ylim([-4, 4])
        #plt.ylim([ymin, ymax])
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.455, 1.17), ncol=2, fancybox=True, shadow=True, fontsize=9.5, framealpha=0.8)
        
        f.show()
                
        input("Boussinesq a-equation balance...")
        
# class PostPoiseuille(PostProcess):
#     def __init__(self):
#         """
#         Constructor of class PostPoiseuille
#         """
#         super(PostPoiseuille, self).__init__(*args, **kwargs)
    












class PostSingleFluid_OLD(PostProcess):
    def __init__(self, solver):
        """
        Constructor of class PostSingleFluid: That is the old one with shear-layer in y
        """

        print("")
        print("OLD SINGLE FLUID POST-PROC")
        print("")
        
        super(PostSingleFluid_OLD, self).__init__(solver)

    def get_eigenfcts_derivatives(self):

        print("")
        print("OLD SINGLE FLUID POST-PROC DERIVATIVES")
        print("")

        #
        # For single fluid the vertical direction is z
        #

        #self.SolvePoissonEquation1D()

        D1 = self.solver.map.D1
        D2 = self.solver.map.D2
        
        ia = 1j*self.solver.alpha[0]
        ib = 1j*self.solver.beta[0]

        ia2 = ( ia )**2.
        ib2 = ( ib )**2.

        iab = ia*ib

        # u-derivatives
        self.dudx = ia*self.solver.ueig
        self.dudy = np.matmul(D1, self.solver.ueig)
        self.dudz = ib*self.solver.ueig

        self.d2udx2 = ia2*self.solver.ueig
        self.d2udy2 = np.matmul(D2, self.solver.ueig)
        self.d2udz2 = ib2*self.solver.ueig
        self.d2udxdy = ia*np.matmul(D1, self.solver.ueig)

        # v-derivatives
        self.dvdx = ia*self.solver.veig
        self.dvdy = np.matmul(D1, self.solver.veig)
        self.dvdz = ib*self.solver.veig

        self.d2vdx2 = ia2*self.solver.veig
        self.d2vdy2 = np.matmul(D2, self.solver.veig)
        self.d2vdz2 = ib2*self.solver.veig
        self.d2vdxdy = ia*np.matmul(D1, self.solver.veig)
        
        # w-derivatives
        self.dwdx = ia*self.solver.weig
        self.dwdy = np.matmul(D1, self.solver.weig)
        self.dwdz = ib*self.solver.weig

        self.d2wdx2 = ia2*self.solver.weig
        self.d2wdy2 = np.matmul(D2, self.solver.weig)
        self.d2wdz2 = ib2*self.solver.weig

        # p-derivatives
        self.dpdx = ia*self.solver.peig
        self.dpdy = np.matmul(D1, self.solver.peig)
        self.dpdz = ib*self.solver.peig

        self.d2pdx2 = ia2*self.solver.peig
        self.d2pdy2 = np.matmul(D2, self.solver.peig)
        self.d2pdz2 = ib2*self.solver.peig

        # rho-derivatives: not relevant for single fluid
        self.drdx = ia*self.solver.reig
        self.drdy = np.matmul(D1, self.solver.reig)
        self.drdz = ib*self.solver.reig
                
        self.d2rdx2 = ia2*self.solver.reig
        self.d2rdy2 = np.matmul(D2, self.solver.reig)
        self.d2rdz2 = ib2*self.solver.reig

        #print("")
        #print("np.max(np.abs( (d2dx2 + d2dy2)p' + 2*1j*alpha*dU/dy )) = ", np.max(np.abs(self.d2pdx2 + self.d2pdy2 + 2.*np.multiply(self.dvdx, self.solver.bsfl.Up))))

    def get_model_terms(self):

        print("")
        print("OLD SINGLE FLUID POST-PROC MODEL TERMS")
        print("")

        # Constants
        C1 = 1.
        Cr1 = 1.

        if ( not hasattr(self, 'l_taylor') ):
            self.compute_taylor_and_integral_scales()

        if ( not hasattr(self, 'ke_dist') ):
            self.get_disturbance_kinetic_energy()

        self.compute_freestream_length_scales_non_dimensional()

        if ( np.min(np.abs(self.l_taylor-self.l_taylor_fst)) > 1.e-5 ):
            print("np.min(np.abs(self.l_taylor-self.l_taylor_fst)) = ", np.min(np.abs(self.l_taylor-self.l_taylor_fst)))
            sys.exit("Check self.l_taylor_fst (single fluid)!!!!!")

        # Set Taylor microscale
        if (taylor_opt==1):
            l_taylor = self.l_taylor_fst
        else:
            l_taylor = self.l_taylor

        if ( np.isnan(np.max(l_taylor)) ):
            sys.exit("l_taylor is nan ===> adjust domain height and re-run 444")

        l_taylor_nondim2 = np.multiply(l_taylor, l_taylor)

        # Pressure strain model terms
        self.rxx_pres_strain_model = 4./3.*Cr1*np.multiply(self.rxy, self.solver.bsfl.Up)
        self.ryy_pres_strain_model = -2./3.*Cr1*np.multiply(self.rxy, self.solver.bsfl.Up)
        self.rzz_pres_strain_model = -2./3.*Cr1*np.multiply(self.rxy, self.solver.bsfl.Up)
        self.rxy_pres_strain_model = Cr1*np.multiply(self.ryy, self.solver.bsfl.Up)

        # Add Rxx term to Rxx equation, Ryy term to Ryy equation
        # add_normal_terms = 1
        # if ( add_normal_terms == 1 ):
        #     Cnxx = 1.
        #     self.rxx_pres_strain_model = self.rxx_pres_strain_model + Cnxx*np.multiply(self.rxx, self.solver.bsfl.Up)
        #     Cnyy = 1.
        #     self.ryy_pres_strain_model = self.ryy_pres_strain_model + Cnyy*np.multiply(self.ryy, self.solver.bsfl.Up)

        z = self.solver.map.z
        
        # Rxx equation: terms
        term1_rxx_out = 4./3.*np.multiply(self.rxy, self.solver.bsfl.Up)
        # np.multiply(self.rxx, self.solver.bsfl.Up) == this term does not work very well, it cannot capture freestream very well
        term2_rxx_out = self.rxx # Adding this term based on intuition & a-posteriori optimization & the fact that in freestream it should look like Rxx
        term3_rxx_out = self.ryy 
        term4_rxx_out = self.rxy 

        # Rxx equation: write data
        data_out = np.column_stack([z, np.real(term1_rxx_out), np.real(term2_rxx_out), np.real(term3_rxx_out), np.real(term4_rxx_out), np.real(self.rxx_pres_strain)])
        datafile_path = "/Users/aph/Desktop/pres_strain_vars_z_fz_gz_hz_iz_exact_single_fluid_rxx.dat" 
        np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])

        # Ryy equation: terms
        term1_ryy_out = (-2./3.)*np.multiply(self.rxy, self.solver.bsfl.Up)
        term2_ryy_out = self.rxx
        term3_ryy_out = self.ryy
        term4_ryy_out = self.rxy

        # Ryy equation: write data
        data_out = np.column_stack([z, np.real(term1_ryy_out), np.real(term2_ryy_out), np.real(term3_ryy_out), np.real(term4_ryy_out), np.real(self.ryy_pres_strain)])
        datafile_path = "/Users/aph/Desktop/pres_strain_vars_z_fz_gz_hz_iz_exact_single_fluid_ryy.dat" 
        np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])

        # Rxy equation: terms
        term1_rxy_out = np.multiply(self.ryy, self.solver.bsfl.Up)
        term2_rxy_out = self.rxx
        term3_rxy_out = self.ryy
        term4_rxy_out = self.rxy

        # Rxy equation: write data
        data_out = np.column_stack([z, np.real(term1_rxy_out), np.real(term2_rxy_out), np.real(term3_rxy_out), np.real(term4_rxy_out), np.real(self.rxy_pres_strain)])
        datafile_path = "/Users/aph/Desktop/pres_strain_vars_z_fz_gz_hz_iz_exact_single_fluid_rxy.dat" 
        np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])

        C2 = 0.4
        alpha_hat = 0.8 #(8.+C2)/11.
        beta_hat = -0.10 #(8.*C2-2.)/11.
        gam_hat = 0. #(60.*C2-4.)/55.
        
        self.rxy_pres_strain_model_lrr_ = alpha_hat*np.multiply(self.ryy, self.solver.bsfl.Up) \
            + beta_hat*np.multiply(self.rxx, self.solver.bsfl.Up) \
            - 0.25*gam_hat*np.multiply(self.ke_dist, self.solver.bsfl.Up)
        self.rxy_pres_strain_model_lrr_ = self.rxy_pres_strain_model_lrr_*0.77

        # Strain only using LRR
        C2 = 0.4
        alpha_hat = -1.5 #2. #(8.+C2)/11.
        beta_hat = 1.#-1. #(8.*C2-2.)/11.
        gam_hat = 2. #(60.*C2-4.)/55.
        
        self.rxy_pres_strain_model_lrr_ONLY_STRAIN = alpha_hat*np.multiply(self.ryy, self.solver.bsfl.Up) \
            + beta_hat*np.multiply(self.rxx, self.solver.bsfl.Up) \
            - 0.25*gam_hat*np.multiply(self.ke_dist, self.solver.bsfl.Up)
        
        # Set to zero 
        self.rxx_pres_strain_model_girimaji = 0.0*self.rxx_pres_strain_model
        self.ryy_pres_strain_model_girimaji = 0.0*self.rxx_pres_strain_model
        self.rzz_pres_strain_model_girimaji = 0.0*self.rxx_pres_strain_model
        self.rxy_pres_strain_model_girimaji = 0.0*self.rxx_pres_strain_model

        # Pressure diffusion model terms
        self.rxx_pres_diffus_model = 0.0*self.rxx_pres_strain_model

        term_tmp1 = np.multiply(self.drxydy, self.solver.bsfl.Up)

        # I added a minus here
        self.ryy_pres_diffus_model = -4.*C1*np.matmul(self.solver.map.D1, np.multiply(term_tmp1, l_taylor_nondim2))
        self.rzz_pres_diffus_model = 0.0*self.rzz_pres_strain_model

        term_tmp2 = np.multiply(self.drxxdy, self.solver.bsfl.Up)
        self.rxy_pres_diffus_model = C1*np.matmul(self.solver.map.D1, np.multiply(term_tmp2, l_taylor_nondim2))

    def compute_freestream_length_scales_non_dimensional(self):

        print("")
        print("OLD SINGLE FLUID POST-PROC FREESTREAM LENGTH SCALES")
        print("")
 
        #
        # THIS WAS FOR SHEAR-LAYER IN Y
        #
        y = self.solver.map.z
        kx = self.solver.alpha[0]
        kz = self.solver.beta[0]
        omega = self.solver.omega_max

        Re = self.solver.Re
        
        # Create local grid
        ny = 1001
        ymin = y[-1]-y[-1]/20
        ymax = y[-1]
        ygrid = np.linspace(ymin, ymax, ny)

        #
        # Matrix entries
        #
        Mat = np.zeros((6, 6), dpc)
        #
        Mat[0,1] = 1.
        #
        Mat[1,0] = kx**2 + kz**2 + 1j*kx*Re -1j*omega*Re
        Mat[1,3] = 1j*kx*Re
        #
        Mat[2,0] = -1j*kx
        Mat[2,4] = -1j*kz
        #
        Mat[3,1] = -1j*kx/Re
        Mat[3,2] = 1j*omega-1j*kx - 1/Re*(kx**2+kz**2)
        Mat[3,5] = -1j*kz/Re
        #
        Mat[4,5] = 1.
        #
        Mat[5,3] = 1j*kz*Re
        Mat[5,4] = (kx**2+kz**2) + 1j*kx*Re -1j*omega*Re

        #
        # Compute eigenvalues
        #
        D, V = linalg.eig(Mat)        
        eigvals = D
        
        tols = 1e-10
        
        # find indices of eigenvalues with negative real parts
        idx_array = np.zeros(3, i4)
        
        ct = 0
        
        for i in range(0, np.size(Mat, 0)):
            if ( eigvals[i].real < 0.0 and np.abs(eigvals[i].real) > tols ):
                idx_array[ct] = i
                ct = ct + 1

        Lam1 = eigvals[idx_array[0]]
        Lam2 = eigvals[idx_array[1]]
        Lam3 = eigvals[idx_array[2]]
        
        uvec = np.zeros(ny,dpc)
        vvec = np.zeros(ny,dpc)
        wvec = np.zeros(ny,dpc)
        pvec = np.zeros(ny,dpc)
        
        ke = np.zeros(ny, dp)
        
        # str = ['u','v','w','p']
        # vars = [0 2 4 3]
        
        for k in range(0, ny):
            
            yloc = ygrid[k]
            
            q = V[:,idx_array[0]]*np.exp(Lam1*yloc) + \
                V[:,idx_array[1]]*np.exp(Lam2*yloc) + \
                V[:,idx_array[2]]*np.exp(Lam3*yloc)
            
            uvec[k] = q[0]
            vvec[k] = q[2]
            wvec[k] = q[4]
            pvec[k] = q[3]
            
            # u -> index 0
            uu = self.compute_inner_prod(q[0], q[0])
            # v -> index 2
            vv = self.compute_inner_prod(q[2], q[2])
            # w -> index 4
            ww = self.compute_inner_prod(q[4], q[4])
            
            # Compute disturbance kinetic energy
            ke[k] = 0.5*( uu + vv + ww ).real

        # Get no viscosity non-dimensional dissipation 
        eps_no_visc = self.compute_disturbance_dissipation_without_viscosity_non_dimensional_LOCAL_FREESTREAM(uvec, vvec, wvec, kx, kz, ygrid)
                
        # Compute the length scales
        l_taylor_fst = np.sqrt(np.divide(15.*ke, eps_no_visc))
        if ( np.abs(l_taylor_fst[-20]-l_taylor_fst[-21]) > 1.0e-7 ):
            sys.exit("Something weird for Freestream Taylor scale (Single Fluid)")

        self.l_taylor_fst = l_taylor_fst[-20].real
        
        print("Freestream Taylor scale (Single Fluid): ", self.l_taylor_fst)

    def compute_disturbance_dissipation_without_viscosity_non_dimensional_LOCAL_FREESTREAM(self, ueig, veig, weig, alpha, beta, ygrid):

        print("")
        print("OLD SINGLE FLUID POST-PROC DISSIPATION FREESTREAM LENGTH SCALES")
        print("")

        #
        # THIS WAS FOR SHEAR-LAYER IN Y
        #
        # Create non-dimensional differentiation matrix
        ny = len(ygrid)
        dy = ygrid[1]-ygrid[0]
        
        D1 = np.zeros((ny, ny), dp)
        D1[0,0] = -1./dy
        D1[0,1] = 1./dy
        
        for ii in range(1, ny-1):
            D1[ii,ii+1] = 1./(2.*dy)
            D1[ii,ii-1] = -1./(2.*dy)
            
        D1[-1,-1] = 1./dy
        D1[-1,-2] = -1./dy

        # Eigenfunctions derivatives
        ia = 1j*alpha
        ib = 1j*beta
        
        dudx = ia*ueig
        dudy = np.matmul(D1, ueig)
        dudz = ib*ueig
        
        dvdx = ia*veig
        dvdy = np.matmul(D1, veig)
        dvdz = ib*veig
        
        dwdx = ia*weig
        dwdy = np.matmul(D1, weig)
        dwdz = ib*weig
        
        sub1 = dudy+dvdx
        sub2 = dvdz+dwdy
        sub3 = dudz+dwdx
        
        div = dudx+dvdy+dwdz
        
        term1 = 2.*( self.compute_inner_prod(dudx, dudx) + self.compute_inner_prod(dvdy, dvdy) + self.compute_inner_prod(dwdz, dwdz) )
        term2 = -2./3.*self.compute_inner_prod(div, div)
        term3 = self.compute_inner_prod(sub1, sub1) + self.compute_inner_prod(sub2, sub2) + self.compute_inner_prod(sub3, sub3)
        
        ########################################
        # Various components of the dissipation
        ########################################
        
        # 1) visc*(du_i/dx_j)*(du_i/dx_j)
        epsilon_tilde_dist = self.compute_inner_prod(dudx, dudx) + self.compute_inner_prod(dudy, dudy) + self.compute_inner_prod(dudz, dudz) \
            + self.compute_inner_prod(dvdx, dvdx) + self.compute_inner_prod(dvdy, dvdy) + self.compute_inner_prod(dvdz, dvdz) \
            + self.compute_inner_prod(dwdx, dwdx) + self.compute_inner_prod(dwdy, dwdy) + self.compute_inner_prod(dwdz, dwdz)
        #epsilon_tilde_dist = visc*epsilon_tilde_dist
        
        # 2) visc*(du_i/dx_j)*(du_j/dx_i)
        epsilon_2tilde_dist = self.compute_inner_prod(dudx, dudx) + self.compute_inner_prod(dvdy, dvdy) + self.compute_inner_prod(dwdz, dwdz) \
            + 2.*self.compute_inner_prod(dudy, dvdx) + 2.*self.compute_inner_prod(dudz, dwdx) + 2.*self.compute_inner_prod(dvdz, dwdy)
        #epsilon_2tilde_dist = visc*epsilon_2tilde_dist
        
        # 3) incompressible dissipation
        epsilon_incomp_dist = epsilon_tilde_dist + epsilon_2tilde_dist
        
        # 4) compressible part of dissipation
        #epsilon_comp_dist = visc*term2
        epsilon_comp_dist = term2
        
        # 5) full dissipation
        epsilon_dist = epsilon_incomp_dist + epsilon_comp_dist 
        
        return epsilon_dist

    def get_reynolds_stresses_related_quantities(self):

        print("")
        print("OLD SINGLE FLUID POST-PROC REYNOLDS STRESSES RELATED QUANTITIES")
        print("")

        print("")
        print("Computation of Reynolds stresses ====> I did not include the constant term pi/alpha or ...")
        print("")

        Re = self.solver.Re

        if ( not hasattr(self, 'dudx') ):
            self.get_eigenfcts_derivatives()

        self.rxx = self.compute_inner_prod(self.solver.ueig, self.solver.ueig)
        self.ryy = self.compute_inner_prod(self.solver.veig, self.solver.veig)
        self.rzz = self.compute_inner_prod(self.solver.weig, self.solver.weig)
        
        self.rxy = self.compute_inner_prod(self.solver.ueig, self.solver.veig)
        self.rxz = self.compute_inner_prod(self.solver.ueig, self.solver.weig)
        self.ryz = self.compute_inner_prod(self.solver.veig, self.solver.weig)

        self.drxxdx = 2.*self.compute_inner_prod(self.solver.ueig, self.dudx)
        self.drxxdy = 2.*self.compute_inner_prod(self.solver.ueig, self.dudy)
        self.drxxdz = 2.*self.compute_inner_prod(self.solver.ueig, self.dudz)
        #
        self.d2rxxdxdy = 2.*( self.compute_inner_prod(self.dudx, self.dudy)
                              +self.compute_inner_prod(self.solver.ueig, self.d2udxdy)
                             ) 

        self.d2rxxdy2 = 2.*( self.compute_inner_prod(self.dudy, self.dudy)
                             +self.compute_inner_prod(self.solver.ueig, self.d2udy2)
                             ) 

        self.dryydx = 2.*self.compute_inner_prod(self.solver.veig, self.dvdx)
        self.dryydy = 2.*self.compute_inner_prod(self.solver.veig, self.dvdy)
        self.dryydz = 2.*self.compute_inner_prod(self.solver.veig, self.dvdz)
        #
        self.d2ryydy2 = 2.*( self.compute_inner_prod(self.dvdy, self.dvdy)
                             +self.compute_inner_prod(self.solver.veig, self.d2vdy2)
                            ) 

        self.drzzdx = 2.*self.compute_inner_prod(self.solver.weig, self.dwdx)
        self.drzzdy = 2.*self.compute_inner_prod(self.solver.weig, self.dwdy)
        self.drzzdz = 2.*self.compute_inner_prod(self.solver.weig, self.dwdz)

        self.drxydx = self.compute_inner_prod(self.solver.ueig, self.dvdx) + self.compute_inner_prod(self.solver.veig, self.dudx)
        self.drxydy = self.compute_inner_prod(self.solver.ueig, self.dvdy) + self.compute_inner_prod(self.solver.veig, self.dudy)
        self.drxydz = self.compute_inner_prod(self.solver.ueig, self.dvdz) + self.compute_inner_prod(self.solver.veig, self.dudz)
        #
        self.d2rxydx2 = ( 2.*self.compute_inner_prod(self.dudx, self.dvdx)
                          +self.compute_inner_prod(self.solver.ueig, self.d2vdx2)
                          +self.compute_inner_prod(self.solver.veig, self.d2udx2)
                         ) 
        #
        self.d2rxydy2 = ( 2.*self.compute_inner_prod(self.dudy, self.dvdy)
                          +self.compute_inner_prod(self.solver.ueig, self.d2vdy2)
                          +self.compute_inner_prod(self.solver.veig, self.d2udy2)
                         ) 
        #
        self.d2rxydxdy = ( self.compute_inner_prod(self.dudy, self.dvdx)
                           +self.compute_inner_prod(self.solver.ueig, self.d2vdxdy)
                           +self.compute_inner_prod(self.dudx, self.dvdy)
                           +self.compute_inner_prod(self.solver.veig, self.d2udxdy)
                         ) 

        #
        # Build left-hand-sides for Reynolds stress equations
        #
        
        # Normal terms
        self.rxx_lhs = 2.*self.omega_i*self.rxx.real
        self.ryy_lhs = 2.*self.omega_i*self.ryy.real
        self.rzz_lhs = 2.*self.omega_i*self.rzz.real

        # Cross terms
        self.rxy_lhs = 2.*self.omega_i*self.rxy.real
        self.rxz_lhs = 2.*self.omega_i*self.rxz.real
        self.ryz_lhs = 2.*self.omega_i*self.ryz.real

        #
        # Rxx equation
        #
        self.rxx_lap = 2.*( self.compute_inner_prod(self.dudx, self.dudx)
                            +self.compute_inner_prod(self.dudy, self.dudy)
                            +self.compute_inner_prod(self.dudz, self.dudz)
                           )/Re
        
        self.rxx_lap = self.rxx_lap + 2.*( self.compute_inner_prod(self.solver.ueig, self.d2udx2)
                                           +self.compute_inner_prod(self.solver.ueig, self.d2udy2)
                                           +self.compute_inner_prod(self.solver.ueig, self.d2udz2)
                                          )/Re
        
        self.rxx_dissip = 2.*( self.compute_inner_prod(self.dudx, self.dudx)
                               +self.compute_inner_prod(self.dudy, self.dudy)
                               +self.compute_inner_prod(self.dudz, self.dudz)
                              )/Re

        self.rxx_advection = -np.multiply(self.solver.bsfl.U, self.drxxdx) -np.multiply(self.solver.bsfl.W, self.drxxdz)
        self.rxx_production = -2.*np.multiply(self.rxy, self.solver.bsfl.Up)

        self.rxx_pres_diffus = -2.* ( self.compute_inner_prod(self.solver.ueig, self.dpdx) + self.compute_inner_prod(self.solver.peig, self.dudx) )
        self.rxx_pres_strain = 2.*self.compute_inner_prod(self.solver.peig, self.dudx)

        self.rxx_grav_prod = 0.0*self.compute_inner_prod(self.solver.peig, self.dudx)

        #
        # Ryy equation
        #
        self.ryy_lap = 2.*( self.compute_inner_prod(self.dvdx, self.dvdx)
                            +self.compute_inner_prod(self.dvdy, self.dvdy)
                            +self.compute_inner_prod(self.dvdz, self.dvdz)
                           )/Re
        
        self.ryy_lap = self.ryy_lap + 2.*( self.compute_inner_prod(self.solver.veig, self.d2vdx2)
                                           +self.compute_inner_prod(self.solver.veig, self.d2vdy2)
                                           +self.compute_inner_prod(self.solver.veig, self.d2vdz2)
                                          )/Re
        
        self.ryy_dissip = 2.*( self.compute_inner_prod(self.dvdx, self.dvdx)
                               +self.compute_inner_prod(self.dvdy, self.dvdy)
                               +self.compute_inner_prod(self.dvdz, self.dvdz)
                              )/Re

        self.ryy_advection = -np.multiply(self.solver.bsfl.U, self.dryydx) -np.multiply(self.solver.bsfl.W, self.dryydz)

        self.ryy_production = 0.0*self.ryy_advection

        self.ryy_pres_diffus = -2.*( self.compute_inner_prod(self.solver.veig, self.dpdy) + self.compute_inner_prod(self.solver.peig, self.dvdy) )
        self.ryy_pres_strain = 2.*self.compute_inner_prod(self.solver.peig, self.dvdy)

        self.ryy_grav_prod = 0.0*self.compute_inner_prod(self.solver.peig, self.dudx)

        #
        # Rxy equation
        #
        self.rxy_lap = 2.*( self.compute_inner_prod(self.dudx, self.dvdx)
                            +self.compute_inner_prod(self.dudy, self.dvdy)
                            +self.compute_inner_prod(self.dudz, self.dvdz)
                           )/Re
        
        self.rxy_lap = self.rxy_lap + ( self.compute_inner_prod(self.solver.ueig, self.d2vdx2)
                                        +self.compute_inner_prod(self.solver.ueig, self.d2vdy2)
                                        +self.compute_inner_prod(self.solver.ueig, self.d2vdz2)
                                       )/Re

        self.rxy_lap = self.rxy_lap + ( self.compute_inner_prod(self.solver.veig, self.d2udx2)
                                        +self.compute_inner_prod(self.solver.veig, self.d2udy2)
                                        +self.compute_inner_prod(self.solver.veig, self.d2udz2)
                                       )/Re

        self.rxy_dissip = 2.*( self.compute_inner_prod(self.dudx, self.dvdx)
                               +self.compute_inner_prod(self.dudy, self.dvdy)
                               +self.compute_inner_prod(self.dudz, self.dvdz)
                              )/Re

        self.rxy_advection = -np.multiply(self.solver.bsfl.U, self.drxydx) -np.multiply(self.solver.bsfl.W, self.drxydz)
        self.rxy_production = -np.multiply(self.ryy, self.solver.bsfl.Up)

        self.rxy_pres_diffus = -( self.compute_inner_prod(self.solver.ueig, self.dpdy)
                               +self.compute_inner_prod(self.solver.peig, self.dudy)
                               +self.compute_inner_prod(self.solver.veig, self.dpdx)
                               +self.compute_inner_prod(self.solver.peig, self.dvdx)
                              )
                                                                                                          
        self.rxy_pres_strain = ( self.compute_inner_prod(self.solver.peig, self.dudy)
                                 +self.compute_inner_prod(self.solver.peig, self.dvdx)
                                )

        print("max(v*dpdx) = ", np.max(self.compute_inner_prod(self.solver.veig, self.dpdx)))
        print("min(v*dpdx) = ", np.min(self.compute_inner_prod(self.solver.veig, self.dpdx)))
        
        print("max(p*dvdx) = ", np.max(self.compute_inner_prod(self.solver.peig, self.dvdx)))
        print("min(p*dvdx) = ", np.min(self.compute_inner_prod(self.solver.peig, self.dvdx)))

        self.rxy_grav_prod = 0.0*self.compute_inner_prod(self.solver.peig, self.dudx)
        
        #
        # Rzz equation
        #
        
        self.rzz_lap = 2.*( self.compute_inner_prod(self.dwdx, self.dwdx)
                            +self.compute_inner_prod(self.dwdy, self.dwdy)
                            +self.compute_inner_prod(self.dwdz, self.dwdz)
                           )/Re
        
        self.rzz_lap = self.rzz_lap + 2.*( self.compute_inner_prod(self.solver.weig, self.d2wdx2)
                                           +self.compute_inner_prod(self.solver.weig, self.d2wdy2)
                                           +self.compute_inner_prod(self.solver.weig, self.d2wdz2)
                                          )/Re
        
        self.rzz_dissip = 2.*( self.compute_inner_prod(self.dwdx, self.dwdx)
                               +self.compute_inner_prod(self.dwdy, self.dwdy)
                               +self.compute_inner_prod(self.dwdz, self.dwdz)
                              )/Re

        self.rzz_advection = -np.multiply(self.solver.bsfl.U, self.drzzdx) -np.multiply(self.solver.bsfl.W, self.drzzdz)
        self.rzz_production = -2.*np.multiply(self.ryz, self.solver.bsfl.Wp)

        self.rzz_pres_diffus = -2.*( self.compute_inner_prod(self.solver.weig, self.dpdz) + self.compute_inner_prod(self.solver.peig, self.dwdz) )
        self.rzz_pres_strain = 2.*self.compute_inner_prod(self.solver.peig, self.dwdz)

        # This is only for RT
        if ( self.solver.Fr != -10. ):
            self.rzz_grav_prod = -2.*self.compute_inner_prod(self.solver.reig, self.solver.weig)/self.solver.Fr**2.
        else:
            self.rzz_grav_prod = 0.0*self.compute_inner_prod(self.solver.reig, self.solver.weig)

        # Write data
        pdudx = self.compute_inner_prod(self.solver.peig, self.dudx)
        pdvdy = self.compute_inner_prod(self.solver.peig, self.dvdy)
        pdwdz = self.compute_inner_prod(self.solver.peig, self.dwdz)
        
        udpdx = self.compute_inner_prod(self.solver.ueig, self.dpdx)
        vdpdy = self.compute_inner_prod(self.solver.veig, self.dpdy)
        wdpdz = self.compute_inner_prod(self.solver.weig, self.dpdz)
        
        ddxpu = pdudx + udpdx
        ddypv = pdvdy + vdpdy
        ddzpw = pdwdz + wdpdz
        
        pdrdx = self.compute_inner_prod(self.solver.peig, self.drdx)
        pdrdy = self.compute_inner_prod(self.solver.peig, self.drdy)
        pdrdz = self.compute_inner_prod(self.solver.peig, self.drdz)

        rdpdx = self.compute_inner_prod(self.solver.reig, self.dpdx)
        rdpdy = self.compute_inner_prod(self.solver.reig, self.dpdy)
        rdpdz = self.compute_inner_prod(self.solver.reig, self.dpdz)
        
        ddxpr = pdrdx + rdpdx
        ddypr = pdrdy + rdpdy
        ddzpr = pdrdz + rdpdz
        
        save_path = "./"
        fname     = "Mean_Pressure_Related_Quantities_LST_OLD_SINGLE_FLUID.txt"
        fullname  = os.path.join(save_path, fname)
        fhandel   = open(fullname,'w')
        fhandel.write('VARIABLES = "z" "pdudx" "pdvdy" "pdwdz" "udpdx" "vdpdy" "wdpdz" "ddxpu" "ddypv" "ddzpw" "pdrdx" "pdrdy" "pdrdz" "ddxpr" "ddypr" "ddzpr"\n')
        np.savetxt(fhandel, np.transpose([ self.solver.map.z, pdudx.real, pdvdy.real, pdwdz.real, udpdx.real, vdpdy.real, wdpdz.real, ddxpu.real, \
                                           ddypv.real, ddzpw.real, pdrdx.real, pdrdy.real, pdrdz.real, ddxpr.real, ddypr.real, ddzpr.real ]), \
                      fmt="%25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e")
        fhandel.close()
