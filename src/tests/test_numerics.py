
import math
import unittest
import numpy as np

from numpy import testing as nptest

from module_utilities import trapezoid_integration
from module_utilities import get_mid_idx

import class_gauss_lobatto as mgl

class TestDifferentiationMatrix(unittest.TestCase):

    def test_first_derivative(self):
        pass
        """Test first derivative on sine function on [1,-1]
        """
        cheb = mgl.GaussLobatto(size=351)

        sine_exact = np.sin(cheb.xc)
        cosine_exact = np.cos(cheb.xc)

        D1 = cheb.DM[:,:,0]
        
        dsinedx_num = np.matmul(D1, sine_exact)

        nptest.assert_allclose(cosine_exact, dsinedx_num, atol=1e-10)

        #print("np.amax(np.abs(cosine_exact-dsinedx_num)) = ", np.amax(np.abs(cosine_exact-dsinedx_num)))
        
        #print("cheb.xc = ", cheb.xc)
        #print("cheb.DM = ", cheb.DM)

    def test_second_derivative(self):
        pass
        """Test second derivative on sine function on [1,-1]
        """
        cheb = mgl.GaussLobatto(size=351)

        sine_exact = np.sin(cheb.xc)
        cosine_exact = np.cos(cheb.xc)

        D2 = cheb.DM[:,:,1]
        
        d2sinedx2_num = np.matmul(D2, sine_exact)

        nptest.assert_allclose(sine_exact, -d2sinedx2_num, atol=2e-6)

        #print("np.amax(np.abs(sine_exact+d2sinedx2_num)) = ", np.amax(np.abs(sine_exact+d2sinedx2_num)))

class TestTrapIntegration(unittest.TestCase):


    def test_sine_zero_2pi(self):
        """
        Test the value of sine function integration
        """
        ny  = 2001
        y   = np.linspace(0, 2*math.pi, ny)
        fct = np.sin(y)

        res = trapezoid_integration(fct, y)

        #print("res = ", res)

        nptest.assert_allclose(res, 0, atol=1e-6) # cannot use a rel tol here I think because ref. val is zero


    def test_sine_zero_pi(self):
        """
        Test the value of sine function integration
        """
        ny  = 2001
        y   = np.linspace(0, math.pi, ny)
        fct = np.sin(y)

        res = trapezoid_integration(fct, y)

        #print("res = ", res)

        nptest.assert_allclose(res, 2, atol=1e-6) # cannot use a rel tol here I think because ref. val is zero




class TestMidIndexValue(unittest.TestCase):


    def test_mid_index_val(self):
        """
        Test the value of mid-index
        Note: only for odd numbers
        """
        ny  = 5
        res = get_mid_idx(ny)

        self.assertEqual(res, 2) # remember python is zero based

        ny  = 7
        res = get_mid_idx(ny)

        self.assertEqual(res, 3)



# if __name__ == '__main__':
#     unittest.main()
