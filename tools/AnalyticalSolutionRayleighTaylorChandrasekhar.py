
import matplotlib
import matplotlib.pyplot as plt

from scipy.special import erf
import numpy as np

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '14'
plt.rc('font', family='serif')

print("")
print("Chandrasekhar analytical stability solution of R-T")
print("==================================================")

rho1  = 1.0
rho2  = 3.*rho1 # that way I get Atw = 0.5

alpha1 = rho1/(rho1+rho2)
alpha2 = rho2/(rho1+rho2)

# alpha2-alpha1 = (rho2-rho1)/(rho1+rho2) = Atwood number
print("")
print("alpha1 + alpha2 (answer should be 1) = ",alpha1+alpha2)
print("Atwood number: alpha2 - alpha1 (answer should be 0.5) = ",alpha2-alpha1)

ymin = 1
ymax = 100
eps  = 1.e-14
npts = 10001

y = np.linspace(ymin+eps, ymax, npts)

# Two equivalent ways to compute Q
Q  = (y-1.0)/(alpha2-alpha1)*( y**3. + (1.+4*alpha1*alpha2)*y**2 +(3.-8.*alpha1*alpha2)*y - (1.-4*alpha1*alpha2)  )
Q2 = 1/(alpha2-alpha1)*( y**4. + 4*alpha1*alpha2*y**3. + 2*(1-6*alpha1*alpha2)*y**2 - 4*(1-3*alpha1*alpha2)*y + 1 - 4*alpha1*alpha2 )

diff = np.abs(Q2-Q)

print("")
print("Comparing two different ways to compute Q:")
print("max(diff) = ", np.max(diff))

# Compute k and n (see Chandrasekhar page 444)

k = Q**(-1./3.)
n = (y**2.-1)*Q**(-2./3.)

#print("k=", k)
#print("n=", n)

idx_sorted = np.argsort(k)
k_sorted   = k[idx_sorted]
n_sorted   = n[idx_sorted] 

#print("k_sorted=",k_sorted)
#print("n_sorted=",n_sorted)

# Data from eigenvalue solver
k_solver = np.array([0.0625, 0.125, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2], dtype=float)
n_solver = np.array([0.15916831, 0.21856004, 0.27011286, 0.28346694, 0.25362894, 0.21671608, 0.18441644, 0.15858822, 0.13828825, 0.12221673], dtype=float)


ptn = plt.gcf().number

f = plt.figure(ptn)
plt.plot(k_sorted, n_sorted, 'b-', markerfacecolor='none', label="Chandrasekhar")
plt.plot(k_solver, n_solver, 'rs', markerfacecolor='none', label="Current")
plt.xlabel('k', fontsize=20)
plt.ylabel('n', fontsize=20)
plt.legend(loc="best")
f.show()
plt.xlim([0, 2])
plt.ylim([0, 0.5])

plt.gcf().subplots_adjust(left=0.16)
plt.gcf().subplots_adjust(bottom=0.13)

input("")

data_out = np.column_stack([k_sorted, n_sorted])
datafile_path = "./Chandrasekhar_exact_rayleigh_taylor_ratio2.txt" #+ str(ny) + ".dat" 
np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e'])

