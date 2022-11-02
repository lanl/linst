
import matplotlib
import matplotlib.pyplot as plt


from scipy.special import erf
import numpy as np

r1 = 1.
r2 = 1.5

delta = 0.1

z = np.linspace(-10,10,101)

r = r1/2. + (r2-r1)/2.*erf(z/delta)


ptn = plt.gcf().number

print("ptn = ", ptn)

f = plt.figure(ptn)
plt.plot(r, z, 'bs', markerfacecolor='none', label="Grid pts")
plt.xlabel('rho')
plt.ylabel('z')
plt.legend(loc="upper right")
f.show()

input()
