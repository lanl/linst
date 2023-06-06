import numpy as np
import unittest

import gauss_lobatto as mgl
import baseflow as mbf
import build_matrices as mbm
import mapping as mma


class TestPoiseuille(unittest.TestCase):

    def __str__(self):

        return f"""      
Unit test(s) for Poiseuille
===========================        
"""

    def test_poiseuille(self):
        """Test Poiseuille case of Kirchner (2000)
        """
        print(str(self))
        cheb = mgl.GaussLobatto(size=351)
        map = mma.MapVoid(sinf=100, cheb=cheb, l=5.0)
        bsfl = mbf.PlanePoiseuille(y=map.y)

        solver = mbm.Poiseuille(
            map=map,
            Re=10000,
            bsfl=bsfl,
        )

        solver.solve(alpha=np.linspace(1.0,1.0,1), beta=0.0, omega_guess=0.2375264888204682+0.0037396706229799*1j)

        self.assertAlmostEqual(solver.omega_max, 0.2375264888204682+0.0037396706229799*1j)

class TestShearLayer(unittest.TestCase):

    def __str__(self):

        return f"""
Unit test(s) for Shear-Layer
============================
"""

    def test_shear_layer(self):
        """Test Shear-Layer case of Michalke
        """
        print(str(self))
        cheb = mgl.GaussLobatto(size=351)
        map = mma.MapShearLayer(sinf=150, cheb=cheb, l=10.)
        bsfl = mbf.HypTan(y=map.y)

        solver = mbm.Poiseuille(
            map=map,
            Re=1.e50,
            bsfl=bsfl,
        )

        solver.solve(alpha=np.linspace(0.4446,0.4446,1), beta=0., omega_guess=0.2223+0.09485*1j)

        self.assertAlmostEqual(solver.omega_max, 0.2223+0.09485*1j, places=5)

        ###################################
        #   Reference data from Michalke  #
        ###################################
        #alpha_michalke = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.4446, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
        #omega_i_michalke = np.array([0.0, 0.04184, 0.06975, 0.08654, 0.09410, 0.09485, 0.09376, 0.08650, 0.07305, 0.05388, 0.02942, 0.0], dtype=float)
        #omega_r_michalke = 0.5*alpha_michalke # cr=0.5, see Michalke JFM 1964

class TestRayleighTaylor(unittest.TestCase):

    def __str__(self):

        return f"""     
Unit test(s) for Rayleigh-Taylor
================================
"""

    def test_rayleigh_taylor(self):
        """Cross-validate eigenvalue by comparing eigenvalue obtained with Boussinesq and Sandoval RT solvers
        for a small Atwood number At=0.01
        """
        print(str(self))
        # Setup and solver with Boussinesq RT solver 
        cheb = mgl.GaussLobatto(size=351)
        map = mma.MapShearLayer(sinf=100, cheb=cheb, l=5.0)
        bsfl = mbf.RTSimple(y=map.y, At=0.01)
        
        solver = mbm.Boussinesq(
            map=map,
            Re=100,
            Fr=1.0,
            Sc=1.0,
            bsfl=bsfl,
        )

        solver.solve(alpha=np.linspace(0.9,0.9,1), beta=0.9, omega_guess=0.0+0.056263*1j)
        omega_boussinesq = solver.omega_max

        # Setup and solver with Sandoval RT solver
        cheb = mgl.GaussLobatto(size=351)
        map = mma.MapShearLayer(sinf=100, cheb=cheb, l=5.0)
        bsfl = mbf.RTSimple(y=map.y, At=0.01)

        solver = mbm.Sandoval(
            map=map,
            Re=100,
            Fr=1.0,
            Sc=1.0,
            bsfl=bsfl,
        )
        
        solver.solve(alpha=np.linspace(0.9,0.9,1), beta=0.9, omega_guess=0.0+0.056263*1j)
        omega_sandoval = solver.omega_max
        
        self.assertAlmostEqual(omega_boussinesq, omega_sandoval, places=5)
        
