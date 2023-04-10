
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg') #----> Specify the backend

from scipy.special import erf
import numpy as np

import math

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '14'
plt.rc('font', family='serif')

pi = math.pi

ymin = -4*pi
ymax = -ymin

ny = 97

y = np.linspace(ymin, ymax, ny)

delta = 1.25

fct_erf = erf(y/delta)

fct_erf_sym = erf(-y/delta)

fct_erf_sym_remove_pt = fct_erf_sym[1:]

print("y[0] = ", y[0])
print("y[ny] = ", y[ny-1])
print("y[ny-1] = ", y[ny-2])

ynew = y[1:]+2*ymax
#ynew = y+2*ymax


print("ynew[-1] = ", ynew[-1])

print("4*pi, 6*pi, 8*pi, 10*pi, 12*pi = ", 4*pi, 6*pi, 8*pi, 10*pi, 12*pi)

print("len(fct_erf+fct_erf_sym) = ", len(fct_erf+fct_erf_sym))

print("np.amax(np.abs(fct_erf+fct_erf_sym)) = ", np.amax(np.abs(fct_erf+fct_erf_sym)))


# start from full domain

nz = (ny-1)*2+1
zmin = -8.*pi
zmax = -zmin

z = np.linspace(zmin, zmax, nz) + 4.*pi

print("y = ", y)
print("z = ", z)

f = plt.figure(1)
plt.plot(fct_erf, y, 'bs-', markerfacecolor='none', label="tanh(y)")
plt.plot(fct_erf_sym_remove_pt, ynew, 'r+-', markerfacecolor='none', label="tanh(y)")

plt.xlabel('fct', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.legend(loc="best")

#plt.xlim([0, 2])
#plt.ylim([-20, 20])

plt.gcf().subplots_adjust(left=0.16)
plt.gcf().subplots_adjust(bottom=0.13)

f.show()

input("Check pic")
