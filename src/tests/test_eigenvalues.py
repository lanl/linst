import numpy as np
import unittest

import class_gauss_lobatto as mgl
import class_baseflow as mbf
import class_build_matrices as mbm
import class_mapping as mma


class TestPoiseuille(unittest.TestCase):
    def test_poiseuille(self):
        """Test Poiseuille case of Kirchner (2000)
        """
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
