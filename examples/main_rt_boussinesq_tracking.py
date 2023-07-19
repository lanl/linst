import sys
import math
import time
import warnings
import numpy as np
import gauss_lobatto as mgl
import baseflow as mbf
import build_matrices as mbm
import mapping as mma

import matplotlib
matplotlib.use('TkAgg') #----> Specify the backend
import matplotlib.pyplot as plt

import module_utilities as mod_util

# Create instance for class GaussLobatto
cheb = mgl.GaussLobatto(size=351)
#map = mma.MapShearLayer(sinf=12, cheb=cheb, l=3.0)
map = mma.MapShearLayer(sinf=500, cheb=cheb, l=10.0)
bsfl = mbf.RTSimple(y=map.y, At=0.05)

solver = mbm.Boussinesq(
    map=map,
    Re=1000,
    Fr=1.0,
    Sc=1.0,
    bsfl=bsfl,
    )

solver.solve(alpha=np.linspace(1., 5., 100), beta=0., omega_guess=0.0+0.163475*1j)
#solver.plot_eigvals()
#plt.show()
#solver.write_eigvals()

q_eigvect = solver.identify_eigenvector_from_target()
solver.get_normalized_eigvects(q_eigvect, bsfl.rt_flag)
solver.plot_eigvects(bsfl.rt_flag)
solver.write_eigvects_out_new()

solver.write_stab_banana()
solver.write_baseflow_out(solver)

