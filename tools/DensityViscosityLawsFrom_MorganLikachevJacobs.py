
import matplotlib
import matplotlib.pyplot as plt


from scipy.special import erf
import numpy as np

grav  = 9.81
rho1  = 1.0
mu1   = 1.e-2

delta = 0.0002 # see Morgan, Likhachev, Jacobs Fig. 17 legend

z = np.linspace(-10,10,1000001)

mu2  = 3.*mu1 # that way I get Amu = 0.5
rho2 = 3.*rho1 # that way I get Atw = 0.5

rho0 = 0.5*( rho1 + rho2 )
mu0 = 0.5*( mu1 + mu2 )

Atw = (rho2-rho1)/(rho2+rho1)
Amu = (mu2-mu1)/(mu2+mu1)

print("Atw, Amu = ", Atw, Amu)

mu = mu0*( 1. + Amu*erf(z/delta) )

rho = rho0*( 1. + Atw*erf(z/delta) )

nu0 = mu0/rho0

Tscale = (nu0/grav**2.)**(1./3.)
Lscale = (nu0**2./grav)**(1./3.)

nu = np.divide(mu, rho)

print("min(nu), max(nu) = ",np.min(nu), np.max(nu))

print('The time scale is %5.3e, the length scale is %5.3e' % (Tscale, Lscale))

print("delta* = ", delta/Lscale)

ptn = plt.gcf().number

f = plt.figure(ptn)
plt.plot(rho, z, 'bs', markerfacecolor='none', label="density")
plt.xlabel('density')
plt.ylabel('z')
plt.legend(loc="upper right")
f.show()

ptn = ptn+1

f = plt.figure(ptn)
plt.plot(mu, z, 'bs', markerfacecolor='none', label="viscosity")
plt.xlabel('viscosity')
plt.ylabel('z')
plt.legend(loc="upper right")
f.show()

ptn = ptn+1

f = plt.figure(ptn)
plt.plot(nu, z, 'bs', markerfacecolor='none', label="viscosity")
plt.xlabel('kinematic viscosity')
plt.ylabel('z')
plt.legend(loc="upper right")
f.show()

k_nondim = np.array([0.0625, 0.125, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0], dtype=float)
k_dim = k_nondim/Lscale

print("k_nondim = ", k_nondim)
print("k_dim = ", k_dim)

input()
