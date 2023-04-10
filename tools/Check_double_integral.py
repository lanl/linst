
import numpy as np
from scipy import integrate

alpha = 1.0
beta = 1.0

# this assumes alpha=1, beta=1
fuv1 = lambda y, x: np.cos(x)*np.cos(y)*np.sin(x)*np.sin(y)
fuv2 = lambda y, x: np.cos(x)*np.cos(y)*np.sin(y)*np.cos(x)
fuv3 = lambda y, x: np.sin(x)*np.cos(y)*np.sin(x)*np.sin(y)
fuv4 = lambda y, x: np.sin(x)*np.cos(y)*np.sin(y)*np.cos(x)

fuw1 = lambda y, x: np.cos(x)**2*np.cos(y)**2
fuw2 = lambda y, x: np.cos(x)*np.cos(y)*np.sin(x)*np.cos(y)
fuw3 = lambda y, x: np.sin(x)*np.cos(y)*np.cos(x)*np.cos(y)
fuw4 = lambda y, x: np.sin(x)**2*np.cos(y)**2

int_uv1 = integrate.dblquad(fuv1, 0, 2*np.pi, 0, 2*np.pi)
int_uv2 = integrate.dblquad(fuv2, 0, 2*np.pi, 0, 2*np.pi)
int_uv3 = integrate.dblquad(fuv3, 0, 2*np.pi, 0, 2*np.pi)
int_uv4 = integrate.dblquad(fuv4, 0, 2*np.pi, 0, 2*np.pi)

int_uw1 = integrate.dblquad(fuw1, 0, 2*np.pi, 0, 2*np.pi)
int_uw2 = integrate.dblquad(fuw2, 0, 2*np.pi, 0, 2*np.pi)
int_uw3 = integrate.dblquad(fuw3, 0, 2*np.pi, 0, 2*np.pi)
int_uw4 = integrate.dblquad(fuw4, 0, 2*np.pi, 0, 2*np.pi)

print("")
print("pi**2/(alpha*beta) = ", np.pi**2/(alpha*beta))

print("")
print("int_uv1 = ", int_uv1)
print("int_uv2 = ", int_uv2)
print("int_uv3 = ", int_uv3)
print("int_uv4 = ", int_uv4)

print("")
print("int_uw1 = ", int_uw1)
print("int_uw2 = ", int_uw2)
print("int_uw3 = ", int_uw3)
print("int_uw4 = ", int_uw4)
print("")
