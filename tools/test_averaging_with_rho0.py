
import numpy as np

rho0=1.5

nx = 101

rhop = np.random.rand(nx)

print("rhop = ", rhop)

rho_bar1 = np.average(rho0+rhop)

rho_bar2 = rho0 + np.average(rhop)

print("rho_bar1, rho_bar2 = ", rho_bar1, rho_bar2)


