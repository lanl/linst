
#from scipy.fft import fft, fftfreq
import numpy as np
from numpy.fft import fft, fftfreq, rfft, rfftfreq, fftshift

import matplotlib

from matplotlib import pyplot as plt

matplotlib.use('TkAgg') #----> Specify the backend

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex
i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

M = 6

N = 2**M

Nf = int(N/2+1)

print("Nf = ", Nf)

kx = ky = fftfreq(N, 1./N).astype(int)
kz = kx[:Nf].copy()
kz[-1] *= -1

print("kx = ", kx)

print("")

print("kz = ", kz)

