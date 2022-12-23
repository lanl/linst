
import matplotlib
import matplotlib.pyplot as plt

from scipy.special import erf
import numpy as np

import math

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '14'
plt.rc('font', family='serif')

ymin = -20
ymax = -ymin

y = np.linspace(-20, 20, 1001)

delta = 1.25#1.3

fct_1 = 0.5*( 1.0 + np.tanh(y) )
fct_2 = 0.5*( 1.0 + erf(y/delta) )

pi = math.pi

dfct_1dy = 0.5*( 1.0 - np.tanh(y)**2. )
dfct_2dy = 0.5*( 2.0*np.exp(-y**2./delta**2.)/( math.sqrt(pi)*delta ) )

f = plt.figure(1)
plt.plot(fct_1, y, 'b-', markerfacecolor='none', label="tanh(y)")
plt.plot(fct_2, y, 'rs', markerfacecolor='none', label="erf(y/delta)")
plt.xlabel('fct', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.legend(loc="best")
f.show()
#plt.xlim([0, 2])
#plt.ylim([-20, 20])

plt.gcf().subplots_adjust(left=0.16)
plt.gcf().subplots_adjust(bottom=0.13)

# Derivatives
                 
f = plt.figure(2)
plt.plot(dfct_1dy, y, 'b-', markerfacecolor='none', label="d/dy[ tanh(y) ]")
plt.plot(dfct_2dy, y, 'rs', markerfacecolor='none', label="d/dy[ erf(y/delta) ]")
plt.xlabel('fct', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.legend(loc="best")
f.show()
#plt.xlim([0, 2])
#plt.ylim([-20, 20])

plt.gcf().subplots_adjust(left=0.16)
plt.gcf().subplots_adjust(bottom=0.13)

                 

input("Hello")
