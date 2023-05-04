
import matplotlib
matplotlib.use('TkAgg') #----> Specify the backend
import matplotlib.pyplot as plt

from scipy.special import erf
import numpy as np

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '14'
plt.rc('font', family='serif')

print("")
print("Chandrasekhar analytical stability solution of R-T")
print("==================================================")

At   = 0.5
rho1 = 1.0

rho2 = rho1*(1.+At)/(1.-At) #3.*rho1 # that way I get Atw = 0.5

# 
alpha1 = rho1/(rho1+rho2)
alpha2 = rho2/(rho1+rho2)

# alpha2-alpha1 = (rho2-rho1)/(rho1+rho2) = Atwood number
print("")
print("alpha1 + alpha2 (answer should be 1) = ",alpha1+alpha2)
print("Atwood number: alpha2 - alpha1 (answer should be 0.5) = ",alpha2-alpha1)

ymin = 1
ymax = 50#100
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
print("ptn = ", ptn)

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


# DIMENSIONAL GROWTH RATES AND WAVENUMBERS
ptn = ptn + 1

g = 9.81
nu = 0.1019367991845056 #1.1073617295175051e-06 #1.2742099898063203e-05 #1.2262500044390254e-06

print("")
print("Reference gravity and kinematic viscosity set to: ", g, nu)

Lscale = (g/nu**2.)**(-1./3.)
Tscale = (g**2./nu)**(-1./3.)

kdim2 = k_sorted/Lscale
ndim2 = n_sorted/Tscale

kdim = k_sorted*(g/nu**2.)**(1./3.)
ndim = n_sorted*(g**2./nu)**(1./3.)

Idx_max_ndim = np.argmax(ndim)

#print("")
#print("ndim[Idx_max_ndim-1], ndim[Idx_max_ndim], ndim[Idx_max_ndim+1] = ",\
#      ndim[Idx_max_ndim-1], ndim[Idx_max_ndim], ndim[Idx_max_ndim+1])

print("")
print("Wavenumber with max. growth rate, k = ", kdim[Idx_max_ndim])
print("")

f = plt.figure(ptn)
plt.plot(kdim, ndim, 'b', label="Chandrasekhar (dimensional)")
plt.plot(kdim2, ndim2, 'r--', label="Chandrasekhar (dimensional)")
plt.xlabel('k (dimensional)', fontsize=20)
plt.ylabel('n (dimensional)', fontsize=20)
plt.legend(loc="best")
f.show()
plt.xlim([0, 5.*kdim[Idx_max_ndim]])
#plt.ylim([0, 0.5])

plt.gcf().subplots_adjust(left=0.16)
plt.gcf().subplots_adjust(bottom=0.13)



input("End of program")


WriteOut = False

if (WriteOut):
    data_out = np.column_stack([k_sorted, n_sorted])
    datafile_path = "./Chandrasekhar_exact_rayleigh_taylor_ratio2.txt" #+ str(ny) + ".dat" 
    np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e'])

